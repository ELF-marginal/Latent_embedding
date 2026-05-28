#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LatentSpeakerDataset, collate_latent_speaker
from models import LatentSpeakerEncoder, LatentSpeakerEncoderConfig, speaker_embedding_loss
from samplers import UtteranceGroupedBatchSampler


def save_training_checkpoint(
    path: Path,
    *,
    model: LatentSpeakerEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val: float,
    config: LatentSpeakerEncoderConfig,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_val": float(best_val),
            "train_args": vars(args),
        },
        path,
    )


def load_training_checkpoint(path: Path, model: LatentSpeakerEncoder, optimizer: torch.optim.Optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "best_val": float(ckpt.get("best_val", float("inf"))),
    }


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


def load_json_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", help="JSON config file. CLI arguments override config values.")
    pre_args, remaining = pre_parser.parse_known_args()

    defaults = {}
    if pre_args.config:
        defaults = flatten_config(load_json_config(pre_args.config))

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--train_manifest", default="")
    parser.add_argument("--val_manifest", default="")
    parser.add_argument("--save_dir", default="checkpoints/latent_spk")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_len", type=int, default=0, help="Random crop length in latent T steps; 0 disables crop.")
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--feat_cache_size", type=int, default=64, help="LRU cache size for full audio_feats tensors in Dataset.")
    parser.add_argument("--feat_cache_max_gb", type=float, default=0.0, help="Maximum RAM used by cached audio_feats; 0 disables byte limit.")
    parser.add_argument("--preload_feats_gb", type=float, default=0.0, help="Sequentially preload audio_feats into RAM up to this many GB before training.")
    parser.add_argument("--preload_embeddings", action="store_true", help="Preload teacher embeddings into RAM before training.")
    parser.add_argument("--embedding_cache_size", type=int, default=4096, help="LRU cache size for teacher embedding tensors in Dataset.")
    parser.add_argument("--group_by_utterance", action="store_true", help="Batch chunks by utterance to improve indexed full-feat cache hits.")
    parser.add_argument("--utterances_per_batch", type=int, default=16)
    parser.add_argument("--sequential_io", action="store_true", help="Sort utterances by feature path and shuffle path blocks to reduce random disk IO.")
    parser.add_argument("--io_block_size", type=int, default=2048, help="Number of utterances per shuffled IO block when --sequential_io is enabled.")
    parser.add_argument("--l2_weight", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every_steps", type=int, default=10000, help="Save a training checkpoint every N steps.")
    parser.add_argument("--save_every_epochs", type=int, default=0, help="Save a training checkpoint every N epochs; 0 disables epoch checkpoints.")
    parser.add_argument("--resume", default="", help="Path to a training checkpoint. Use 'latest' to resume from save_dir/latest_train.pt.")
    parser.add_argument("--no_auto_resume", action="store_true", help="Do not auto-resume from save_dir/latest_train.pt.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    if not args.train_manifest:
        parser.error("--train_manifest is required unless provided by --config")
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.preload_feats_gb > 0 and args.num_workers > 0:
        print(
            "[warning] --preload_feats_gb with num_workers>0 can duplicate the RAM cache in worker processes. "
            "For the 32GB in-memory mode, use --num_workers 0 unless you intentionally want per-worker caches."
        )

    train_ds = LatentSpeakerDataset(
        args.train_manifest,
        min_len=args.min_len,
        max_len=args.max_len,
        random_crop=True,
        feat_cache_size=args.feat_cache_size,
        feat_cache_max_gb=args.feat_cache_max_gb,
        embedding_cache_size=args.embedding_cache_size,
    )
    train_ds.preload_caches(
        preload_feats_gb=args.preload_feats_gb,
        preload_embeddings=args.preload_embeddings,
        sort_by_path=args.sequential_io,
    )
    train_batch_sampler = None
    if args.group_by_utterance:
        train_batch_sampler = UtteranceGroupedBatchSampler(
            train_ds.items,
            batch_size=args.batch_size,
            utterances_per_batch=args.utterances_per_batch,
            shuffle=True,
            seed=1234,
            sequential_io=args.sequential_io,
            io_block_size=args.io_block_size,
        )
    train_loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_latent_speaker,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.persistent_workers and args.num_workers > 0,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
    }
    if train_batch_sampler is not None:
        train_loader_kwargs["batch_sampler"] = train_batch_sampler
    else:
        train_loader_kwargs["batch_size"] = args.batch_size
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_ds, **train_loader_kwargs)

    val_loader = None
    if args.val_manifest:
        val_ds = LatentSpeakerDataset(
            args.val_manifest,
            min_len=args.min_len,
            max_len=args.max_len,
            random_crop=False,
            feat_cache_size=args.feat_cache_size,
            feat_cache_max_gb=args.feat_cache_max_gb,
            embedding_cache_size=args.embedding_cache_size,
        )
        val_ds.preload_caches(
            preload_feats_gb=0.0,
            preload_embeddings=args.preload_embeddings,
            sort_by_path=args.sequential_io,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_latent_speaker,
            pin_memory=device.type == "cuda",
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
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
    start_epoch = 0
    global_step = 0
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"model": asdict(cfg), "train_args": vars(args)}, f, indent=2, ensure_ascii=False)

    resume_path = None
    if args.resume:
        resume_path = save_dir / "latest_train.pt" if args.resume == "latest" else Path(args.resume)
    elif not args.no_auto_resume and (save_dir / "latest_train.pt").exists():
        resume_path = save_dir / "latest_train.pt"

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = load_training_checkpoint(resume_path, model, optimizer, device)
        start_epoch = state["epoch"]
        global_step = state["global_step"]
        best_val = state["best_val"]
        print(f"[resume] loaded {resume_path} at epoch={start_epoch}, global_step={global_step}, best_val={best_val:.6f}")

    for epoch in range(start_epoch, args.epochs):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
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
            global_step += 1

            with torch.no_grad():
                cosine = torch.nn.functional.cosine_similarity(student, teacher, dim=-1).mean()
            running.append(loss.detach())
            running_cos.append(cosine.detach())
            progress.set_postfix(step=global_step, loss=f"{loss.item():.4f}", cosine=f"{cosine.item():.4f}")

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_training_checkpoint(
                    save_dir / f"step_{global_step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val=best_val,
                    config=cfg,
                    args=args,
                )
                save_training_checkpoint(
                    save_dir / "latest_train.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val=best_val,
                    config=cfg,
                    args=args,
                )

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

        model.save_checkpoint(save_dir / "latest.pt")
        if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            save_training_checkpoint(
                save_dir / f"epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                best_val=best_val,
                config=cfg,
                args=args,
            )
        save_training_checkpoint(
            save_dir / "latest_train.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            best_val=best_val,
            config=cfg,
            args=args,
        )
        with (save_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train": train_metrics,
                        "val": val_metrics,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
