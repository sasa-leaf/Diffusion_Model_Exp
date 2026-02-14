import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """
    時間tをEmbeddingするクラス
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    時間条件付きTransformerブロック (adaLNなしの簡易版: 時間埋め込みを加算)
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )

    def forward(self, x, c):
        # 時間条件cからシフトとスケールを生成 (adaLN-Zero like)
        shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)

        # Self-Attention
        x_norm = self.norm1(x)
        x = (
            x
            + (1 + scale_msa.unsqueeze(1)) * self.attn(x_norm, x_norm, x_norm)[0]
            + shift_msa.unsqueeze(1)
        )

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleJiT(nn.Module):
    """
    シンプルなViT構造
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.IMG_SIZE
        self.patch_size = config.PATCH_SIZE
        self.in_channels = 3

        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_dim = self.in_channels * self.patch_size**2

        # Patch Embedding
        self.x_embed = nn.Linear(self.patch_dim, config.EMBED_DIM)

        # Positional Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.EMBED_DIM)
        )

        # Time Embedding
        self.t_embed = TimestepEmbedder(config.EMBED_DIM)

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(config.EMBED_DIM, config.NUM_HEADS) for _ in range(config.DEPTH)]
        )

        # Output Head
        self.final_layer = nn.Linear(config.EMBED_DIM, self.patch_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.x_embed.weight)
        nn.init.zeros_(self.x_embed.bias)
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def patchify(self, x):
        """
        (N, C, H, W) -> (N, L, Patch_Dim)
        """
        p = self.patch_size
        assert x.shape[2] == x.shape[3] and x.shape[2] % p == 0
        h = w = x.shape[2] // p
        x = x.reshape(shape=(x.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(x.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        (N, L, Patch_Dim) -> (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return x

    def forward(self, x, t):
        """
        x: (N, C, H, W) - ノイズ付き画像
        t: (N,) - タイムステップ [0, 1]
        """
        # 1. 画像をパッチ化
        x = self.patchify(x)  # (N, L, D)

        # 2. Embedding
        x = self.x_embed(x) + self.pos_embed
        t_emb = self.t_embed(t)  # (N, D)

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # 4. 出力射影
        x = self.final_layer(x)

        # 5. 画像に戻す
        x = self.unpatchify(x)
        return x
