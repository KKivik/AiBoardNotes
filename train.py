from dotenv import load_dotenv
import os
import torch
from torch.nn import functional as F
import torch.nn as nn
from prepare_data import ds_train, ds_test

load_dotenv()

# MODEL WEIGHTS
DIM_EMBEDDING = 256
DIM_ATTENTION = 32
D_PIC = 400
N_HEAD_ATTENTION = 8

# PICTURE
N_CONTEXT = 768 # num of patched in picture
W = int(os.getenv("W"))
H = int(os.getenv("H"))
KERNEL = 20

# TRAINING PARAMETERS




device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def precompute_theta_per_frequencies(head_size: int, theta = 10000.0):
    m_x = torch.arange(0, W // KERNEL).repeat(H // KERNEL)
    m_y = torch.arange(0, H // KERNEL).repeat_interleave(W // KERNEL - 1)
    theta_pairs_numerator = torch.arange(0, head_size // 2, 2).float()
    theta = 1.0 / (theta ** (2 * theta_pairs_numerator / head_size)).to(device)
    freqs_x = torch.outer(m_x, theta).float()
    freqs_y = torch.outer(m_y, theta).float()
    freqs_complex_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_complex_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return freqs_complex_x, freqs_complex_y #(T, head_size // 4)

    # theta_pairs_numerator = torch.arange(0, head_size, 2).float() # i of each pair in emb
    # theta = 1.0 / (theta ** (2 * theta_pairs_numerator / head_size)).to(device)
    # m = torch.arange(m, device=device)
    # freqs = torch.outer(m, theta).float()
    # freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # convert to complex space based on Euler's formula e ^ (it)
    # return freqs_complex #(T, head_size // 2)

@torch.no_grad()
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex_x: torch.Tensor, freqs_complex_y: torch.Tensor):
    v_x, v_y = torch.chunk(x, 2, dim=-1) # each: (B, T, head_size // 2)
    v_x_complex = torch.view_as_complex(v_x.float().reshape(*v_x.shape[:-1], -1, 2)) # (B, T, head_size // 4)
    v_y_complex = torch.view_as_complex(v_y.float().reshape(*v_y.shape[:-1], -1, 2)) # (B, T, head_size // 4)
    freqs_complex_x = freqs_complex_x.unsqueeze(0)
    freqs_complex_y = freqs_complex_y.unsqueeze(0)
    v_x_rotated = v_x_complex * freqs_complex_x
    v_y_rotated = v_y_complex * freqs_complex_y
    v_x_out = torch.view_as_real(v_x_rotated) # (B, T, head_size // 4, 2)
    v_y_out = torch.view_as_real(v_y_rotated) # (B, T, head_size // 4, 2)
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


class AttentionHead(nn.Module):
    def __init__(self, head_size, freqs_complex_x: torch.Tensor, freqs_complex_y: torch.Tensor): # patch dim - dim of embedding space, head_size - dim of head space
        super().__init__()
        self.freqs_complex_x, self.freqs_complex_y = freqs_complex_x, freqs_complex_y
        self.key = nn.Linear(DIM_EMBEDDING, head_size, bias=False)
        self.query = nn.Linear(DIM_EMBEDDING, head_size, bias=False)
        self.value = nn.Linear(DIM_EMBEDDING, head_size, bias=False)
    def forward(self, x):
        B, T, C = x.shape #(batch, count of elements, count of features)
        k = self.key(x) # (B, T, head_size)
        k = apply_rotary_embeddings(k, self.freqs_complex_x, self.freqs_complex_y)

        q = self.query(x) # (B, T, head_size)
        q = apply_rotary_embeddings(q, self.freqs_complex_x, self.freqs_complex_y)

        v = self.value(x) # (B, T, head_size)
        wei = (q @ k.transpose(-2, -1) / (C ** 0.5)) # (B, T, T)
        softwei = F.softmax(wei, dim=-1)
        out = softwei @ v
        return out # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_heads):
        super().__init__()
        self.freqs_complex_x, self.freqs_complex_y = precompute_theta_per_frequencies(DIM_ATTENTION)
        self.heads = nn.ModuleList([AttentionHead(DIM_ATTENTION, self.freqs_complex_x, self.freqs_complex_y) for i in range(num_of_heads)])
        self.proj = nn.Linear(N_HEAD_ATTENTION * DIM_ATTENTION, DIM_EMBEDDING)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out # (B, T, DIM_EMBEDDING)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM_EMBEDDING, DIM_EMBEDDING * 4),
            nn.ReLU(),
            nn.Linear(DIM_EMBEDDING * 4, DIM_EMBEDDING)
        )

    def forward(self, x):
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = MultiHeadAttention(N_HEAD_ATTENTION)
        self.mlp = FeedForward()
        self.ln1 = nn.LayerNorm(DIM_EMBEDDING)
        self.ln2 = nn.LayerNorm(DIM_EMBEDDING)
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_emb = nn.Linear(D_PIC, DIM_EMBEDDING)
        self.transformers = nn.Sequential(
            ResBlock(),
            ResBlock()
        )

    def forward(self, x):
        x_emb = self.w_emb(x)
        x = self.transformers(x_emb)





