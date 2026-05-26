#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from prepare_student_dataset import SUPPORTED_AUDIO_EXTS, build_rows_from_wav_root


def balanced_sample(grouped: dict[str, list[dict]], target_count: int, rng: random.Random) -> list[dict]:
    speakers = list(grouped)
    rng.shuffle(speakers)
    for speaker_id in speakers:
        rng.shuffle(grouped[speaker_id])

    selected = []
    cursor = 0
    while len(selected) < target_count and speakers:
        speaker_id = speakers[cursor % len(speakers)]
        bucket = grouped[speaker_id]
        if bucket:
            selected.append(bucket.pop())
        if not bucket:
            speakers.remove(speaker_id)
            if not speakers:
                break
            cursor %= len(speakers)
        else:
            cursor += 1
    return selected


def random_sample(grouped: dict[str, list[dict]], target_count: int, rng: random.Random) -> list[dict]:
    rows = [row for bucket in grouped.values() for row in bucket]
    rng.shuffle(rows)
    return rows[:target_count]


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lines(values: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(value + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def row_key(row: dict) -> str:
    return str(Path(row["audio"]).resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Build speaker-disjoint train/test audio splits.")
    parser.add_argument("--wav_root", required=True)
    parser.add_argument("--speaker_id_regex", default=r"^(?P<speaker_id>.+)_[^_]+$")
    parser.add_argument("--speaker_id_fallback", default="parent")
    parser.add_argument("--audio_exts", nargs="*", default=sorted(SUPPORTED_AUDIO_EXTS))
    parser.add_argument("--num_train_audio", type=int, required=True)
    parser.add_argument("--num_test_audio", type=int, required=True)
    parser.add_argument("--min_files_per_speaker", type=int, default=1)
    parser.add_argument("--sample_strategy", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--extend_from", default="", help="Existing split directory to extend while preserving speaker assignment.")
    parser.add_argument("--out_dir", default="splits/momo_5000h")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    wav_root = Path(args.wav_root)
    rows = build_rows_from_wav_root(
        wav_root,
        args.audio_exts,
        speaker_id_regex=args.speaker_id_regex,
        speaker_id_fallback=args.speaker_id_fallback,
        recursive=True,
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["speaker_id"]].append(row)

    grouped = {
        speaker_id: bucket
        for speaker_id, bucket in grouped.items()
        if len(bucket) >= args.min_files_per_speaker
    }
    speakers = list(grouped)
    rng.shuffle(speakers)

    old_train_rows = []
    old_test_rows = []
    old_train_speakers = []
    old_test_speakers = []
    used_audio = set()
    if args.extend_from:
        old_dir = Path(args.extend_from)
        old_train_rows = read_jsonl(old_dir / "train_audio.jsonl")
        old_test_rows = read_jsonl(old_dir / "test_audio.jsonl")
        old_train_speakers = read_lines(old_dir / "train_speakers.txt") or sorted(
            {row["speaker_id"] for row in old_train_rows}
        )
        old_test_speakers = read_lines(old_dir / "test_speakers.txt") or sorted(
            {row["speaker_id"] for row in old_test_rows}
        )
        overlap = set(old_train_speakers) & set(old_test_speakers)
        if overlap:
            raise RuntimeError(f"Existing split has speaker overlap: {sorted(overlap)[:10]}")
        used_audio = {row_key(row) for row in old_train_rows + old_test_rows}

    locked_train = set(old_train_speakers)
    locked_test = set(old_test_speakers)
    available_speakers = [s for s in speakers if s not in locked_train and s not in locked_test]

    train_target_remaining = max(0, args.num_train_audio - len(old_train_rows))
    test_target_remaining = max(0, args.num_test_audio - len(old_test_rows))

    test_speakers = list(locked_test)
    test_capacity = 0
    new_test_speakers = []
    while available_speakers and test_capacity < test_target_remaining:
        speaker_id = available_speakers.pop()
        new_test_speakers.append(speaker_id)
        test_speakers.append(speaker_id)
        test_capacity += sum(1 for row in grouped[speaker_id] if row_key(row) not in used_audio)

    train_speakers = list(locked_train) + available_speakers
    train_grouped = {speaker_id: list(grouped[speaker_id]) for speaker_id in train_speakers}
    test_grouped = {speaker_id: list(grouped[speaker_id]) for speaker_id in test_speakers}
    for group in (train_grouped, test_grouped):
        for speaker_id, bucket in list(group.items()):
            group[speaker_id] = [row for row in bucket if row_key(row) not in used_audio]

    if args.sample_strategy == "balanced":
        new_train_rows = balanced_sample(train_grouped, train_target_remaining, rng)
        new_test_rows = balanced_sample(test_grouped, test_target_remaining, rng)
    else:
        new_train_rows = random_sample(train_grouped, train_target_remaining, rng)
        new_test_rows = random_sample(test_grouped, test_target_remaining, rng)

    train_rows = old_train_rows + new_train_rows
    test_rows = old_test_rows + new_test_rows

    train_speaker_set = sorted({row["speaker_id"] for row in train_rows})
    test_speaker_set = sorted({row["speaker_id"] for row in test_rows})
    overlap = set(train_speaker_set) & set(test_speaker_set)
    if overlap:
        raise RuntimeError(f"Speaker overlap detected: {sorted(overlap)[:10]}")

    write_jsonl(train_rows, out_dir / "train_audio.jsonl")
    write_jsonl(test_rows, out_dir / "test_audio.jsonl")
    write_lines(train_speaker_set, out_dir / "train_speakers.txt")
    write_lines(test_speaker_set, out_dir / "test_speakers.txt")

    summary = {
        "wav_root": str(wav_root),
        "total_audio": len(rows),
        "eligible_speakers": len(grouped),
        "num_train_audio": len(train_rows),
        "num_test_audio": len(test_rows),
        "old_train_audio": len(old_train_rows),
        "old_test_audio": len(old_test_rows),
        "new_train_audio": len(new_train_rows),
        "new_test_audio": len(new_test_rows),
        "num_train_speakers": len(train_speaker_set),
        "num_test_speakers": len(test_speaker_set),
        "new_test_speakers": len(new_test_speakers),
        "speaker_overlap": 0,
        "extend_from": args.extend_from,
        "sample_strategy": args.sample_strategy,
        "min_files_per_speaker": args.min_files_per_speaker,
        "seed": args.seed,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote split files to {out_dir}")


if __name__ == "__main__":
    main()
