from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_path(path: str | Path, manifest_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else manifest_dir / p


def _load_audio_feats(path: Path) -> torch.Tensor:
    obj: Any = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "audio_feats" not in obj:
            raise KeyError(f"{path} is a dict but has no 'audio_feats' key")
        obj = obj["audio_feats"]
    feats = torch.as_tensor(obj, dtype=torch.float32)
    if feats.ndim != 3:
        raise ValueError(f"{path} must contain [T,P,D] audio feats, got {tuple(feats.shape)}")
    return feats


def _load_embedding(value: Any, manifest_dir: Path) -> torch.Tensor:
    if isinstance(value, list):
        return torch.tensor(value, dtype=torch.float32)
    if not isinstance(value, str):
        raise TypeError("teacher_embedding must be a list or path")

    path = _resolve_path(value, manifest_dir)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("embedding", "teacher_embedding", "spk_embedding"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise KeyError(f"{path} is a dict but has no embedding key")
    return torch.as_tensor(obj, dtype=torch.float32)


class LatentSpeakerDataset(Dataset):
    def __init__(self, manifest: str | Path, min_len: int = 1, max_len: int = 0, random_crop: bool = True):
        self.manifest = Path(manifest)
        self.manifest_dir = self.manifest.parent
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.random_crop = bool(random_crop)

        self.items = []
        with self.manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "audio_feats" not in item or "teacher_embedding" not in item:
                        raise KeyError("Each row needs 'audio_feats' and 'teacher_embedding'")
                    self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        feats_path = _resolve_path(item["audio_feats"], self.manifest_dir)
        feats = _load_audio_feats(feats_path)

        if feats.size(0) < self.min_len:
            raise ValueError(f"{feats_path} has only {feats.size(0)} frames; min_len={self.min_len}")

        if self.max_len > 0 and feats.size(0) > self.max_len:
            if self.random_crop:
                start = torch.randint(0, feats.size(0) - self.max_len + 1, ()).item()
            else:
                start = 0
            feats = feats[start : start + self.max_len]

        emb = _load_embedding(item["teacher_embedding"], self.manifest_dir).flatten()
        emb = torch.nn.functional.normalize(emb, dim=0)
        return {
            "audio_feats": feats,
            "teacher_embedding": emb,
            "length": feats.size(0),
            "id": item.get("id", str(idx)),
            "speaker_id": item.get("speaker_id", ""),
            "utterance_id": item.get("utterance_id", ""),
            "chunk_index": item.get("chunk_index", -1),
        }


def collate_latent_speaker(batch):
    max_len = max(sample["length"] for sample in batch)
    patch, dim = batch[0]["audio_feats"].shape[1:]
    feats = torch.zeros(len(batch), max_len, patch, dim, dtype=torch.float32)
    lengths = torch.tensor([sample["length"] for sample in batch], dtype=torch.long)
    embeddings = torch.stack([sample["teacher_embedding"] for sample in batch], dim=0)

    for idx, sample in enumerate(batch):
        cur = sample["audio_feats"]
        feats[idx, : cur.size(0)] = cur

    return {
        "audio_feats": feats,
        "lengths": lengths,
        "teacher_embedding": embeddings,
        "ids": [sample["id"] for sample in batch],
        "speaker_ids": [sample["speaker_id"] for sample in batch],
        "utterance_ids": [sample["utterance_id"] for sample in batch],
        "chunk_indices": torch.tensor([sample["chunk_index"] for sample in batch], dtype=torch.long),
    }
