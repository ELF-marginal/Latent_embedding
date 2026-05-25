#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from prepare_student_dataset import (
    SUPPORTED_AUDIO_EXTS,
    build_rows_from_wav_root,
    main as prepare_main,
)


def read_excluded_audio_paths(manifest: Path) -> set[str]:
    if not manifest or not manifest.exists():
        return set()
    excluded = set()
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "audio" in row:
                excluded.add(str(Path(row["audio"]).resolve()))
    return excluded


def write_temp_manifest(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a held-out latent speaker test dataset.")
    parser.add_argument("--wav_root", default="/home/lqh/datasets/momo_5000h/audio")
    parser.add_argument("--num_audio", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exclude_manifest", default="")
    parser.add_argument("--out_dir", default="test_data")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--voxcpm_path", default="VoxCPMv1.5")
    parser.add_argument("--teacher_model", default="speech_eres2net_large_sv_zh-cn_3dspeaker_16k")
    parser.add_argument("--speaker_id_regex", default=r"^(?P<speaker_id>.+)_[^_]+$")
    parser.add_argument("--audio_exts", nargs="*", default=[".wav"])
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--chunk_hop", type=int, default=50)
    parser.add_argument("--min_chunk_len", type=int, default=25)
    parser.add_argument("--device", default="")
    parser.add_argument("--teacher_device", default="")
    parser.add_argument("--show_modelscope_warnings", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    wav_root = Path(args.wav_root)
    rows = build_rows_from_wav_root(
        wav_root,
        args.audio_exts or sorted(SUPPORTED_AUDIO_EXTS),
        speaker_id_regex=args.speaker_id_regex,
        speaker_id_fallback="parent",
        recursive=True,
    )

    excluded = read_excluded_audio_paths(Path(args.exclude_manifest)) if args.exclude_manifest else set()
    if excluded:
        rows = [row for row in rows if str(Path(row["audio"]).resolve()) not in excluded]

    import random

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.num_audio]
    if not rows:
        raise RuntimeError("No test audio rows selected.")

    temp_manifest = out_dir / "_selected_audio.jsonl"
    write_temp_manifest(rows, temp_manifest)

    prepare_args = [
        "prepare_student_dataset.py",
        "--input_manifest",
        str(temp_manifest),
        "--voxcpm_path",
        args.voxcpm_path,
        "--teacher_model",
        args.teacher_model,
        "--out_root",
        str(out_dir / "student_cache"),
        "--out_manifest",
        str(out_dir / "test_manifest.jsonl"),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_hop",
        str(args.chunk_hop),
        "--min_chunk_len",
        str(args.min_chunk_len),
        "--skip_existing",
    ]
    if args.num_shards > 1:
        prepare_args.extend(
            [
                "--num_shards",
                str(args.num_shards),
                "--shard_index",
                str(args.shard_index),
                "--sharded_manifest",
            ]
        )
    if args.device:
        prepare_args.extend(["--device", args.device])
    if args.teacher_device:
        prepare_args.extend(["--teacher_device", args.teacher_device])
    if args.show_modelscope_warnings:
        prepare_args.append("--show_modelscope_warnings")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = prepare_args
        prepare_main()
    finally:
        sys.argv = old_argv

    print(f"Built test data under {out_dir}")
    print(f"Manifest: {out_dir / 'test_manifest.jsonl'}")


if __name__ == "__main__":
    main()
