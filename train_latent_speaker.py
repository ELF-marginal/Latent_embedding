#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LatentSpeakerDataset, collate_latent_speaker
from models import LatentSpeakerEncoder, LatentSpeakerEncoderConfig, speaker_embedding_loss


def evaluate(model, loader, device, l2_weight: float):
    model.eval()
    losses = []
    cosines = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["audio_feats"].to(device)
            lengths = batch["lengths"].to(device)
            teacher = batch["teacher_embedding"].to(device)
            student = model(feats, lengths)
            losses.append(speaker_embedding_loss(student, teacher, l2_weight=l2_weight).detach())
            cosines.append(torch.nn.functional.cosine_similarity(student, teacher, dim=-1).mean().detach())
    if not losses:
        return {"loss": 0.0, "cosine": 0.0}
    return {
        "loss": torch.stack(losses).mean().item(),
        "cosine": torch.stack(cosines).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", default="")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=192)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_len", type=int, default=0, help="Random crop length in latent T steps; 0 disables crop.")
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--l2_weight", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LatentSpeakerDataset(args.train_manifest, min_len=args.min_len, max_len=args.max_len, random_crop=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_latent_speaker,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if args.val_manifest:
        val_ds = LatentSpeakerDataset(args.val_manifest, min_len=args.min_len, max_len=args.max_len, random_crop=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_latent_speaker,
            pin_memory=device.type == "cuda",
        )

    cfg = LatentSpeakerEncoderConfig(
        patch_size=args.patch_size,
        feat_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
    )
    model = LatentSpeakerEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"model": asdict(cfg), "train_args": vars(args)}, f, indent=2, ensure_ascii=False)

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        running = []
        running_cos = []

        for batch in progress:
            feats = batch["audio_feats"].to(device)
            lengths = batch["lengths"].to(device)
            teacher = batch["teacher_embedding"].to(device)

            student = model(feats, lengths)
            loss = speaker_embedding_loss(student, teacher, l2_weight=args.l2_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                cosine = torch.nn.functional.cosine_similarity(student, teacher, dim=-1).mean()
            running.append(loss.detach())
            running_cos.append(cosine.detach())
            progress.set_postfix(loss=f"{loss.item():.4f}", cosine=f"{cosine.item():.4f}")

        train_metrics = {
            "loss": torch.stack(running).mean().item(),
            "cosine": torch.stack(running_cos).mean().item(),
        }
        print(f"[train] epoch={epoch + 1} loss={train_metrics['loss']:.6f} cosine={train_metrics['cosine']:.6f}")

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args.l2_weight)
            print(f"[val] epoch={epoch + 1} loss={val_metrics['loss']:.6f} cosine={val_metrics['cosine']:.6f}")
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                model.save_checkpoint(save_dir / "best.pt")

        model.save_checkpoint(save_dir / "latest.pt")
        with (save_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}) + "\n")


if __name__ == "__main__":
    main()

