from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Return a boolean mask with True on valid timesteps."""
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1-D, got {tuple(lengths.shape)}")
    max_len = int(max_len or lengths.max().item())
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


@dataclass
class LatentSpeakerEncoderConfig:
    patch_size: int = 4
    feat_dim: int = 64
    hidden_dim: int = 384
    num_layers: int = 4
    num_heads: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    embedding_dim: int = 192


class AttentiveStatsPooling(nn.Module):
    """Padding-aware attentive mean/std pooling for variable-length utterances."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C], mask: [B, T] with True for valid frames.
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        mean = torch.sum(weights * x, dim=1)
        var = torch.sum(weights * (x - mean.unsqueeze(1)).pow(2), dim=1).clamp_min(1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=-1)


class LatentSpeakerEncoder(nn.Module):
    """
    Student speaker encoder over VoxCPM AudioVAE latent patches.

    Input:
        audio_feats: [B, T, P, D]
        lengths: number of valid T steps for each sample.

    Output:
        L2-normalized speaker embedding [B, embedding_dim].
    """

    def __init__(self, config: LatentSpeakerEncoderConfig | None = None):
        super().__init__()
        self.config = config or LatentSpeakerEncoderConfig()
        cfg = self.config

        input_dim = cfg.patch_size * cfg.feat_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.pool = AttentiveStatsPooling(cfg.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.embedding_dim),
        )

    def forward(self, audio_feats: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if audio_feats.ndim != 4:
            raise ValueError(f"audio_feats must be [B,T,P,D], got {tuple(audio_feats.shape)}")

        bsz, time, patch, dim = audio_feats.shape
        if patch != self.config.patch_size or dim != self.config.feat_dim:
            raise ValueError(
                f"Expected patch/feat dims {(self.config.patch_size, self.config.feat_dim)}, got {(patch, dim)}"
            )

        if lengths is None:
            lengths = torch.full((bsz,), time, dtype=torch.long, device=audio_feats.device)
        else:
            lengths = lengths.to(audio_feats.device, dtype=torch.long).clamp(min=1, max=time)

        mask = lengths_to_mask(lengths, max_len=time)
        x = audio_feats.reshape(bsz, time, patch * dim)
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        pooled = self.pool(x, mask)
        emb = self.output_proj(pooled)
        return F.normalize(emb, dim=-1)

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        config = LatentSpeakerEncoderConfig(**ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["state_dict"])
        return model


def speaker_embedding_loss(student: torch.Tensor, teacher: torch.Tensor, l2_weight: float = 0.1) -> torch.Tensor:
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    cosine_loss = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
    l2_loss = F.mse_loss(student, teacher)
    return cosine_loss + l2_weight * l2_loss

