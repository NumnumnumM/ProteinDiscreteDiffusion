import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Generate sinusoidal timestep embedding.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ProteinDenoiser(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_seq_length, num_classes=20, num_ss_classes=8, dropout=0.1):
        super(ProteinDenoiser, self).__init__()
        self.d_model = d_model
        self.seq_embed = nn.Embedding(num_classes, d_model)
        self.ss_embed = nn.Embedding(num_ss_classes, d_model)
        self.pos_encode = PositionalEncoding(d_model, max_seq_length)
        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # self.recon_head = nn.Linear(d_model, num_classes)
        self.post_head = nn.Linear(d_model, num_classes)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Generate sinusoidal timestep embedding.
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, seq, t, ss=None, padding_mask=None, **kwargs):
        assert ss is not None
        assert padding_mask is not None
        seq_emb = self.seq_embed(seq)
        ss_emb = self.ss_embed(ss)
        # x = torch.cat([seq_emb, ss_emb], dim=-1)
        x = seq_emb + ss_emb
        x = self.pos_encode(x)
        t_emb = self.t_embed(self.get_timestep_embedding(t, self.d_model))
        x = x + t_emb.unsqueeze(1)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = F.relu(self.conv1(x)) + F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        # recon = self.recon_head(x)
        post = self.post_head(x)
        # post = F.softmax(self.post_head(x), dim=-1)
        padding_mask = padding_mask.unsqueeze(-1).expand_as(post)
        post = post * padding_mask
        return post