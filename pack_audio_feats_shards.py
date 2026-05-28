#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def make_relative(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_audio_feats(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        feats = torch.from_numpy(np.load(path)).float()
    else:
        obj: Any = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj["audio_feats"]
        feats = torch.as_tensor(obj, dtype=torch.float32)
    if feats.ndim != 3:
        raise ValueError(f"{path} must contain [T,P,D] audio feats, got {tuple(feats.shape)}")
    return feats.contiguous()


def row_source_path(row: dict[str, Any]) -> str:
    storage = row.get("chunk_storage", "")
    if storage == "indexed":
        return str(row.get("source_full_audio_feats") or row["audio_feats"])
    return str(row["audio_feats"])


def iter_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Pack many small audio_feats files referenced by a manifest into fewer contiguous .npy shards."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--shard_size_gb", type=float, default=2.0)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    manifest_dir = manifest.parent
    out_dir = Path(args.out_dir)
    out_manifest = Path(args.out_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = iter_rows(manifest)
    if not rows:
        raise RuntimeError(f"No rows found in {manifest}")

    source_values = []
    seen = set()
    for row in rows:
        source = row_source_path(row)
        key = str(resolve_path(source, manifest_dir))
        if key not in seen:
            seen.add(key)
            source_values.append(source)

    dtype = np.float16 if args.dtype == "float16" else np.float32
    dtype_size = np.dtype(dtype).itemsize
    shard_max_bytes = max(1, int(args.shard_size_gb * 1024**3))

    packed_index: dict[str, tuple[Path, int, int]] = {}
    shard_arrays: list[np.ndarray] = []
    shard_entries: list[tuple[str, int, int]] = []
    shard_len = 0
    shard_bytes = 0
    shard_id = 0

    def flush_shard() -> None:
        nonlocal shard_arrays, shard_entries, shard_len, shard_bytes, shard_id
        if not shard_arrays:
            return
        shard_path = out_dir / f"audio_feats_shard_{shard_id:05d}.npy"
        merged = np.concatenate(shard_arrays, axis=0)
        np.save(shard_path, merged)
        for key, start, end in shard_entries:
            packed_index[key] = (shard_path, start, end)
        print(
            f"[pack] wrote {shard_path} shape={tuple(merged.shape)} "
            f"size={shard_bytes / 1024**3:.2f}GB files={len(shard_entries)}"
        )
        shard_arrays = []
        shard_entries = []
        shard_len = 0
        shard_bytes = 0
        shard_id += 1

    for source in tqdm(source_values, desc="packing audio_feats"):
        source_path = resolve_path(source, manifest_dir)
        key = str(source_path)
        feats = load_audio_feats(source_path)
        array = feats.numpy().astype(dtype, copy=False)
        nbytes = int(array.size * dtype_size)
        if shard_arrays and shard_bytes + nbytes > shard_max_bytes:
            flush_shard()
        start = shard_len
        end = start + int(array.shape[0])
        shard_arrays.append(array)
        shard_entries.append((key, start, end))
        shard_len = end
        shard_bytes += nbytes
    flush_shard()

    with out_manifest.open("w", encoding="utf-8") as writer:
        for row in rows:
            packed_row = dict(row)
            source = row_source_path(row)
            source_path = resolve_path(source, manifest_dir)
            shard_path, start, end = packed_index[str(source_path)]
            packed_row["original_audio_feats"] = row["audio_feats"]
            packed_row["audio_feats"] = make_relative(shard_path, out_manifest.parent)
            packed_row["packed_feat_start"] = int(start)
            packed_row["packed_feat_end"] = int(end)

            if row.get("chunk_storage") != "indexed":
                length = int(row.get("length", end - start))
                packed_row["chunk_start"] = 0
                packed_row["chunk_end"] = length
            packed_row["chunk_storage"] = "packed"
            writer.write(json.dumps(packed_row, ensure_ascii=False) + "\n")

    print(f"[pack] wrote packed manifest: {out_manifest}")
    print(f"[pack] shards={shard_id} unique_sources={len(source_values)} rows={len(rows)}")


if __name__ == "__main__":
    main()
