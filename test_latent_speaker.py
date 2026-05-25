#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import LatentSpeakerDataset, collate_latent_speaker
from models import LatentSpeakerEncoder, LatentSpeakerEncoderConfig, speaker_embedding_loss


def load_model(checkpoint: Path, device: torch.device) -> LatentSpeakerEncoder:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "config" in ckpt and "state_dict" in ckpt:
        config = LatentSpeakerEncoderConfig(**ckpt["config"])
        model = LatentSpeakerEncoder(config)
        model.load_state_dict(ckpt["state_dict"])
    elif "model_config" in ckpt and "model_state_dict" in ckpt:
        config = LatentSpeakerEncoderConfig(**ckpt["model_config"])
        model = LatentSpeakerEncoder(config)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise KeyError(f"Unsupported checkpoint format: {checkpoint}")
    return model.to(device).eval()


def percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return torch.quantile(values.float(), q).item()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a latent speaker encoder checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="test_data/test_manifest.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--l2_weight", type=float, default=0.1)
    parser.add_argument("--out_json", default="test_data/test_results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(Path(args.checkpoint), device)

    dataset = LatentSpeakerDataset(args.manifest, min_len=1, max_len=0, random_crop=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_latent_speaker,
        pin_memory=device.type == "cuda",
    )

    losses = []
    cosines = []
    speaker_cosines = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            feats = batch["audio_feats"].to(device)
            lengths = batch["lengths"].to(device)
            teacher = batch["teacher_embedding"].to(device)
            student = model(feats, lengths)

            batch_loss = speaker_embedding_loss(student, teacher, l2_weight=args.l2_weight)
            batch_cos = torch.nn.functional.cosine_similarity(student, teacher, dim=-1).detach().cpu()

            losses.append(batch_loss.detach().cpu())
            cosines.append(batch_cos)
            for speaker_id, cosine in zip(batch["speaker_ids"], batch_cos.tolist()):
                speaker_cosines[speaker_id].append(float(cosine))

    cosine_tensor = torch.cat(cosines) if cosines else torch.empty(0)
    speaker_means = torch.tensor(
        [sum(values) / max(1, len(values)) for values in speaker_cosines.values()],
        dtype=torch.float32,
    )
    metrics = {
        "checkpoint": str(Path(args.checkpoint)),
        "manifest": str(Path(args.manifest)),
        "chunks": int(cosine_tensor.numel()),
        "speakers": int(len(speaker_cosines)),
        "loss": torch.stack(losses).mean().item() if losses else 0.0,
        "cosine_mean": cosine_tensor.mean().item() if cosine_tensor.numel() else 0.0,
        "cosine_median": cosine_tensor.median().item() if cosine_tensor.numel() else 0.0,
        "cosine_p10": percentile(cosine_tensor, 0.10),
        "cosine_p90": percentile(cosine_tensor, 0.90),
        "speaker_cosine_mean": speaker_means.mean().item() if speaker_means.numel() else 0.0,
        "speaker_cosine_p10": percentile(speaker_means, 0.10),
        "speaker_cosine_p90": percentile(speaker_means, 0.90),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

