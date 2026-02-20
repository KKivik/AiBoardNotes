from dotenv import load_dotenv
import os
import torch
from torch.distributed.tensor.parallel import loss_parallel
from torch.nn import functional as F
import torch.nn as nn
from tqdm import trange

from prepare_data import ds_train, ds_test, Tkn, dl_train, dl_test

load_dotenv()

# MODEL WEIGHTS
DIM_IMAGE_EMBEDDING = 512
DIM_ATTENTION = 256
D_PIC = 400
N_HEAD_ATTENTION = 64

# PICTURE
N_CONTEXT = 768  # num of patched in picture
W = int(os.getenv("W"))
H = int(os.getenv("H"))
KERNEL = 20

# TEXT PARAMETERS
MAX_LEN_OF_TEXT_CONTEXT = 188 + 1  # 187 - max len of tokens sequence (mean text in utf-8). 1 - special token for start of sequence
DIM_TEXT_EMBEDDING = 128
DIM_TEXT_ATTENTION = 128
N_HEAD_LATEX_ATTENTION = 32
VOCAB_SIZE = 303

# CROSS-MECHANISM PARAMETERS
DIM_CROSS_EMBEDDING = 512
N_HEAD_CROSS_ATTENTION = 32
DIM_CROSS_ATTENTION = 256

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def precompute_theta_2D_per_frequencies(head_size: int, theta=10000.0):
    m_x = torch.arange(0, W // KERNEL).repeat(H // KERNEL).to(device)
    m_y = torch.arange(0, H // KERNEL).repeat_interleave(W // KERNEL).to(device)
    theta_pairs_numerator = torch.arange(0, head_size // 2, 2).float()
    theta = 1.0 / (theta ** (2 * theta_pairs_numerator / head_size)).to(device)
    freqs_x = torch.outer(m_x, theta).float()
    freqs_y = torch.outer(m_y, theta).float()
    freqs_complex_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_complex_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return freqs_complex_x, freqs_complex_y  # (T, head_size // 2)

    # theta_pairs_numerator = torch.arange(0, head_size, 2).float() # i of each pair in emb
    # theta = 1.0 / (theta ** (2 * theta_pairs_numerator / head_size)).to(device)
    # m = torch.arange(m, device=device)
    # freqs = torch.outer(m, theta).float()
    # freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # convert to complex space based on Euler's formula e ^ (it)
    # return freqs_complex #(T, head_size // 2)


@torch.no_grad()
def apply_rotary_2D_embeddings(x: torch.Tensor, freqs_complex_x: torch.Tensor, freqs_complex_y: torch.Tensor):
    v_x, v_y = torch.chunk(x, 2, dim=-1)  # each: (B, T, head_size // 2)
    v_x_complex = torch.view_as_complex(v_x.float().reshape(*v_x.shape[:-1], -1, 2))  # (B, T, head_size // 4)
    v_y_complex = torch.view_as_complex(v_y.float().reshape(*v_y.shape[:-1], -1, 2))  # (B, T, head_size // 4)
    freqs_complex_x = freqs_complex_x.unsqueeze(0)
    freqs_complex_y = freqs_complex_y.unsqueeze(0)
    v_x_rotated = v_x_complex * freqs_complex_x
    v_y_rotated = v_y_complex * freqs_complex_y
    v_x_out = torch.view_as_real(v_x_rotated)  # (B, T, head_size // 4, 2)
    v_y_out = torch.view_as_real(v_y_rotated)  # (B, T, head_size // 4, 2)
    v_x_out = v_x_out.reshape(*v_x.shape)
    v_y_out = v_y_out.reshape(*v_y.shape)
    x_out = torch.cat([v_x_out, v_y_out], dim=-1)
    return x_out.type_as(x).to(device)

    # x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # (B, T, head_size // 2)
    # freqs_complex = freqs_complex.unsqueeze(0) # (1, T, head_size // 2)
    # x_rotated = x_complex * freqs_complex
    # x_out = torch.view_as_real(x_rotated) #(B, T, head_size // 2, 2)
    # x_out = x_out.reshape(*x.shape)
    # return x_out.type_as(x).to(device)


@torch.no_grad()
def precompute_theta_1D_per_frequencies(head_size: int, theta=10000.0, m=MAX_LEN_OF_TEXT_CONTEXT):
    theta_pairs_numerator = torch.arange(0, head_size, 2).float()  # i of each pair in emb
    theta = 1.0 / (theta ** (2 * theta_pairs_numerator / head_size)).to(device)
    m = torch.arange(m, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs),
                                freqs)  # convert to complex space based on Euler's formula e ^ (it)
    return freqs_complex  # (T, head_size // 2)


@torch.no_grad()
def apply_rotary_1D_embeddings(x: torch.Tensor, freqs_complex):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B, T, head_size // 2)
    freqs_complex = freqs_complex.unsqueeze(0)  # (1, T, head_size // 2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)  # (B, T, head_size // 2, 2)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class AttentionHead(nn.Module):
    def __init__(self, head_size, freqs_complex_x: torch.Tensor,
                 freqs_complex_y: torch.Tensor):  # patch dim - dim of embedding space, head_size - dim of head space
        super().__init__()
        self.freqs_complex_x, self.freqs_complex_y = freqs_complex_x, freqs_complex_y
        self.key = nn.Linear(DIM_IMAGE_EMBEDDING, head_size, bias=False)
        self.query = nn.Linear(DIM_IMAGE_EMBEDDING, head_size, bias=False)
        self.value = nn.Linear(DIM_IMAGE_EMBEDDING, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape  # (batch, count of elements, count of features)
        k = self.key(x)  # (B, T, head_size)
        k = apply_rotary_2D_embeddings(k, self.freqs_complex_x, self.freqs_complex_y)

        q = self.query(x)  # (B, T, head_size)
        q = apply_rotary_2D_embeddings(q, self.freqs_complex_x, self.freqs_complex_y)

        v = self.value(x)  # (B, T, head_size)
        wei = (q @ k.transpose(-2, -1) / (C ** 0.5))  # (B, T, T)
        softwei = F.softmax(wei, dim=-1)
        out = softwei @ v
        return out  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs_complex_x, self.freqs_complex_y = precompute_theta_2D_per_frequencies(DIM_ATTENTION)
        self.heads = nn.ModuleList(
            [AttentionHead(DIM_ATTENTION, self.freqs_complex_x, self.freqs_complex_y) for i in range(N_HEAD_ATTENTION)])
        self.proj = nn.Linear(N_HEAD_ATTENTION * DIM_ATTENTION, DIM_IMAGE_EMBEDDING)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out  # (B, T, DIM_IMAGE_EMBEDDING)


class FeedForward(nn.Module):
    def __init__(self, embedding_in, embedding_out=None):
        if embedding_out == None:
            embedding_out = embedding_in
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_in, embedding_in * 4),
            nn.ReLU(),
            nn.Linear(embedding_in * 4, embedding_out)
        )

    def forward(self, x):
        return self.net(x)


class ResViTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = MultiHeadAttention()
        self.mlp = FeedForward(DIM_IMAGE_EMBEDDING)
        self.ln1 = nn.LayerNorm(DIM_IMAGE_EMBEDDING)
        self.ln2 = nn.LayerNorm(DIM_IMAGE_EMBEDDING)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x  # (B, T, DIM_IMAGE_EMBEDDING)


class AttentionHead_Latex_withMask(nn.Module):
    def __init__(self, head_size,
                 freqs_complex: torch.Tensor):  # patch dim - dim of embedding space, head_size - dim of head space
        super().__init__()
        self.freqs_complex = freqs_complex
        self.key = nn.Linear(DIM_TEXT_EMBEDDING, head_size, bias=False)
        self.query = nn.Linear(DIM_TEXT_EMBEDDING, head_size, bias=False)
        self.value = nn.Linear(DIM_TEXT_EMBEDDING, head_size, bias=False)

    def forward(self, x_text_emb, x_latex_mask):
        B, T, C = x_text_emb.shape  # (batch, count of elements, count of features)
        k = self.key(x_text_emb)  # (B, T, head_size)
        k = apply_rotary_1D_embeddings(k, self.freqs_complex)

        q = self.query(x_text_emb)  # (B, T, head_size)
        q = apply_rotary_1D_embeddings(q, self.freqs_complex)

        v = self.value(x_text_emb)  # (B, T, head_size)
        wei = (q @ k.transpose(-2, -1) / (C ** 0.5))  # (B, T, T)
        trill = torch.tril(torch.ones((T, T))).to(device)
        wei = wei.masked_fill(trill == 0, float("-inf"))  # trial mask
        wei = wei.masked_fill(x_latex_mask.unsqueeze(1) == 0, float("-inf"))  # padding mask (by rows)
        softwei = F.softmax(wei, dim=-1)
        out = softwei @ v
        return out  # (B, T, head_size)


class MultiHeadAttention_LaTeX(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs_complex = precompute_theta_1D_per_frequencies(DIM_TEXT_ATTENTION)
        self.heads = nn.ModuleList([AttentionHead_Latex_withMask(DIM_TEXT_ATTENTION, self.freqs_complex) for i in
                                    range(N_HEAD_LATEX_ATTENTION)])
        self.proj = nn.Linear(N_HEAD_LATEX_ATTENTION * DIM_TEXT_ATTENTION, DIM_TEXT_EMBEDDING)

    def forward(self, x_text_emb, x_latex_mask):
        out = torch.cat([h(x_text_emb, x_latex_mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out  # (B, T, DIM_TEXT_EMBEDDING)


class ResLaTeXBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = MultiHeadAttention_LaTeX()
        self.mlp = FeedForward(DIM_TEXT_EMBEDDING)
        self.ln1 = nn.LayerNorm(DIM_TEXT_EMBEDDING)
        self.ln2 = nn.LayerNorm(DIM_TEXT_EMBEDDING)

    def forward(self, X):  # x_text_emb, x_latex_mask
        x_text_emb, x_latex_mask = X
        x = x_text_emb + self.att(self.ln1(x_text_emb), x_latex_mask)
        x = x_text_emb + self.mlp(self.ln2(x))
        return x  # (B, T, DIM_TEXT_EMBEDDING)


class Cross_AttentionHead_withMask(nn.Module):
    def __init__(self, head_size, freqs_complex_latex, freqs_complex_image_x,
                 freqs_complex_image_y):  # patch dim - dim of embedding space, head_size - dim of head space
        super().__init__()
        self.freqs_complex = freqs_complex_latex
        self.freqs_complex_image_x = freqs_complex_image_x
        self.freqs_complex_image_y = freqs_complex_image_y

        self.key = nn.Linear(DIM_IMAGE_EMBEDDING, head_size, bias=False)  # (B, T1, head_size)
        self.query = nn.Linear(DIM_TEXT_EMBEDDING, head_size, bias=False)  # (B, T2, head_size)
        self.value = nn.Linear(DIM_IMAGE_EMBEDDING, head_size, bias=False)  # (B, T1, head_size)

    def forward(self, x_image, x_text_emb, x_latex_mask):
        B, T, C = x_image.shape  # (batch, count of elements, count of features)

        k = self.key(x_image)  # (B, T_k, head_size)
        k = apply_rotary_2D_embeddings(k, self.freqs_complex_image_x, self.freqs_complex_image_y)

        q = self.query(x_text_emb)  # (B, T_q, head_size)
        q = apply_rotary_1D_embeddings(q, self.freqs_complex)

        v = self.value(x_image)  # (B, T_k, head_size)
        wei = (q @ k.transpose(-2, -1) / (C ** 0.5))  # (B, T_q, T_k)

        # if torch.isinf(wei).all(dim=-1).any():
        #     print("FULLY MASKED ROW FOUND")
        softwei = F.softmax(wei, dim=-1)
        out = softwei @ v
        return out  # (B, T_q, head_size)


class Cross_MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs_complex_latex = precompute_theta_1D_per_frequencies(DIM_CROSS_ATTENTION)
        self.freqs_complex_image_x, self.freqs_complex_image_y = precompute_theta_2D_per_frequencies(
            DIM_CROSS_ATTENTION)

        self.heads = nn.ModuleList([Cross_AttentionHead_withMask(DIM_CROSS_ATTENTION, self.freqs_complex_latex,
                                                                 self.freqs_complex_image_x, self.freqs_complex_image_y)
                                    for i in range(N_HEAD_CROSS_ATTENTION)])
        self.proj = nn.Linear(N_HEAD_CROSS_ATTENTION * DIM_CROSS_ATTENTION, DIM_CROSS_EMBEDDING)

    def forward(self, x_image, x_latex, x_latex_mask):
        out = torch.cat([h(x_image, x_latex, x_latex_mask) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out  # (B, T_latex, DIM_CROSS_EMBEDDING)


class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = Cross_MultiHeadAttention()
        self.mlp = FeedForward(DIM_CROSS_EMBEDDING)
        self.ln1 = nn.LayerNorm(DIM_IMAGE_EMBEDDING)
        self.ln2 = nn.LayerNorm(DIM_TEXT_EMBEDDING)

    def forward(self, X):
        x_latex, x_image, x_latex_mask = X
        # x_latex (B, 190, DIM_TEXT_EMBEDDING)
        # x_image (B, T=768, DIM_IMAGE, DIM_IMAGE_EMBEDDING)
        x = self.att(self.ln1(x_image), self.ln2(x_latex), x_latex_mask)  # (B, T_latex=190, DIM_CROSS_EMBEDDING)
        x = x + self.mlp(x)
        return x  # (B, T_latex, DIM_CROSS_EMBEDDING)


class FormulaAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_to_emb = nn.Linear(D_PIC, DIM_IMAGE_EMBEDDING)
        self.VIT_transformers = nn.Sequential(
            ResViTBlock(),
            ResViTBlock()
        )
        self.latex_transformers = nn.Sequential(
            ResLaTeXBlock()
        )
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE + 1, DIM_TEXT_EMBEDDING)
        self.cross_attention = nn.Sequential(
            CrossAttentionBlock()
        )
        self.final_mlp = FeedForward(DIM_CROSS_EMBEDDING, VOCAB_SIZE + 1)

    def forward(self, X):
        x_image, x_latex, y = X
        x_image = x_image.to(device)
        x_latex = x_latex.to(device)
        y = y.to(device)

        x_image_emb = self.image_to_emb(x_image)
        x_image_VIT = self.VIT_transformers(x_image_emb)  # (B, T_image, DIM_IMAGE_EMBEDDING)

        # x_latex:
        # [[301, 178, 204, 303, 303, .., 303]]     [302] - not here, because it's x_latex, not target

        x_latex_mask = Tkn.mask_padding(x_latex).to(device)
        x_text_emb = self.token_embedding_table(x_latex)  # (B, T, DIM_TEXT_EMBEDDING)
        x_processed_latex = self.latex_transformers((x_text_emb, x_latex_mask))  # (B, T_latex, DIM_TEXT_EMBEDDING)
        logits = self.cross_attention(
            (x_processed_latex, x_image_VIT, x_latex_mask))  # (B, T_latex, DIM_CROSS_EMBEDDING)
        logits = self.final_mlp(logits)  # (B, T_latex, vocab_size)

        B, T_latex, voc = logits.shape
        logits = logits.view(B * T_latex, voc)
        targets = y.view(B * T_latex)
        loss = F.cross_entropy(logits, targets, ignore_index=Tkn.padding)

        return logits, loss


# ---- TRAIN ----
model = FormulaAI().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
EPOCHS = 10

losses = []
cnt = 0
if __name__ == "__main__":
    for epoch in trange(EPOCHS):
        for sample in dl_train:
            logits, loss = model(sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            cnt += 1
            #if cnt % 1000 == 0:
            print(loss.item())










