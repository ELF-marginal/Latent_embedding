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


def load_embedding(value: Any, manifest_dir: Path) -> np.ndarray:
    if isinstance(value, list):
        emb = np.asarray(value, dtype=np.float32)
    else:
        path = resolve_path(value, manifest_dir)
        if path.suffix == ".npy":
            emb = np.load(path).astype(np.float32, copy=False)
        else:
            obj: Any = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                for key in ("embedding", "teacher_embedding", "spk_embedding"):
                    if key in obj:
                        obj = obj[key]
                        break
            emb = torch.as_tensor(obj, dtype=torch.float32).numpy()
    emb = np.squeeze(emb).astype(np.float32, copy=False)
    if emb.ndim != 1:
        raise ValueError(f"Expected a 1-D teacher embedding, got shape {emb.shape}")
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


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
    parser.add_argument("--pack_embeddings", action="store_true", help="Pack per-row teacher embeddings into one contiguous .npy matrix.")
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

    source_pairs = []
    seen = set()
    for row in rows:
        source = row_source_path(row)
        key = str(resolve_path(source, manifest_dir))
        if key not in seen:
            seen.add(key)
            source_pairs.append((key, source))
    source_pairs.sort(key=lambda pair: pair[0])

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

    for _, source in tqdm(source_pairs, desc="packing audio_feats"):
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

    packed_embeddings_path = None
    if args.pack_embeddings:
        embeddings = []
        for row in tqdm(rows, desc="packing teacher embeddings"):
            embeddings.append(load_embedding(row["teacher_embedding"], manifest_dir))
        embedding_matrix = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
        packed_embeddings_path = out_manifest.parent / "teacher_embeddings.npy"
        packed_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(packed_embeddings_path, embedding_matrix)
        print(
            f"[pack] wrote {packed_embeddings_path} shape={tuple(embedding_matrix.shape)} "
            f"size={embedding_matrix.nbytes / 1024**3:.2f}GB"
        )

    with out_manifest.open("w", encoding="utf-8") as writer:
        for row_idx, row in enumerate(rows):
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
            if packed_embeddings_path is not None:
                packed_row["original_teacher_embedding"] = row["teacher_embedding"]
                packed_row["teacher_embedding"] = make_relative(packed_embeddings_path, out_manifest.parent)
                packed_row["teacher_embedding_index"] = int(row_idx)
            writer.write(json.dumps(packed_row, ensure_ascii=False) + "\n")

    print(f"[pack] wrote packed manifest: {out_manifest}")
    print(f"[pack] shards={shard_id} unique_sources={len(source_pairs)} rows={len(rows)}")


if __name__ == "__main__":
    main()
