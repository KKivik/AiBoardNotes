import torch
from torch.nn import functional as F
import torch.nn as nn

DIM_EMBEDDING = 256
DIM_ATTENTION = 32
D_PIC = 400

N_HEAD_ATTENTION = 8

class AttentionHead(nn.Module):
    def __init__(self, head_size): # patch dim - dim of embedding space, head_size - dim of head space
        super().__init__()
        self.key = nn.Linear(DIM_EMBEDDING, head_size, bias=False)
        self.query = nn.Linear(DIM_EMBEDDING, head_size, bias=False)
        self.value = nn.Linear(DIM_EMBEDDING, head_size, bias=False)
    def forward(self, x):
        B, T, C = x.shape #(batch, count of elements, count of features)
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = (q @ k.transpose(-2, -1) / (C ** 0.5)) # (B, T, T)
        softwei = F.softmax(wei, dim=-1)
        out = softwei @ v
        return out # (B, T, head_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(DIM_ATTENTION) for i in range(num_of_heads)])
        self.proj = nn.Linear(N_HEAD_ATTENTION * DIM_ATTENTION, DIM_EMBEDDING)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out # (B, T, N_HEAD_ATTENTION x DIM_ATTENTION)

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
    def forward(self, x):
        y = x + self.att(x)
        y = self.mlp(y)
        return x + y

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
        x = self.wha(x_emb)
        x = self.mlp(x)



