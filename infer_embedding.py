#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models import LatentSpeakerEncoder


def load_feats(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj["audio_feats"]
    feats = torch.as_tensor(obj, dtype=torch.float32)
    if feats.ndim == 3:
        feats = feats.unsqueeze(0)
    if feats.ndim != 4:
        raise ValueError(f"Expected [T,P,D] or [B,T,P,D], got {tuple(feats.shape)}")
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio_feats", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = LatentSpeakerEncoder.from_checkpoint(args.checkpoint, map_location=device).to(device).eval()
    feats = load_feats(Path(args.audio_feats)).to(device)
    lengths = torch.full((feats.size(0),), feats.size(1), dtype=torch.long, device=device)

    with torch.no_grad():
        emb = model(feats, lengths).cpu().numpy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)


if __name__ == "__main__":
    main()

