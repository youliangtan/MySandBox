import torch
import torch.nn as nn
import torch.optim as optim
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach() # detach() is used to prevent backpropagation


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        """
        heads: number of heads
        d_model: dimension of model
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads # dimension of each head
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0) # batch size
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # attention
        # formula: Head = softmax((Q * K^T) / sqrt(d_k)) * V
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        # concat heads
        output = torch.matmul(scores, v).transpose(1, 2).contiguous().view(
            bs, -1, self.d_model)
        return self.out(output) # output linear



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) # add and norm operation
        return out


class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.dropout(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


if __name__ == "__main__":
    # Example usage:
    heads = 8
    d_model = 512
    dropout = 0.1

    mha = MultiHeadAttention(heads, d_model, dropout)
    q = torch.randn(64, 10, d_model)
    k = torch.randn(64, 10, d_model)
    v = torch.randn(64, 10, d_model)

    output = mha(q, k, v)
    print(output.shape)  # Expected shape: (64, 10, 512)
