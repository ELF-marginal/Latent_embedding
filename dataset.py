from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


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
    def __init__(
        self,
        manifest: str | Path,
        min_len: int = 1,
        max_len: int = 0,
        random_crop: bool = True,
        feat_cache_size: int = 0,
        feat_cache_max_gb: float = 0.0,
        embedding_cache_size: int = 0,
    ):
        self.manifest = Path(manifest)
        self.manifest_dir = self.manifest.parent
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.random_crop = bool(random_crop)
        self.feat_cache_size = max(0, int(feat_cache_size))
        self.feat_cache_max_bytes = max(0, int(float(feat_cache_max_gb) * 1024**3))
        self.embedding_cache_size = max(0, int(embedding_cache_size))
        self._feat_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._feat_cache_bytes = 0
        self._embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

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

    def _feat_cache_enabled(self) -> bool:
        return self.feat_cache_size > 0 or self.feat_cache_max_bytes > 0

    def _put_audio_feats(self, key: str, feats: torch.Tensor) -> None:
        if not self._feat_cache_enabled():
            return

        old = self._feat_cache.pop(key, None)
        if old is not None:
            self._feat_cache_bytes -= _tensor_nbytes(old)

        self._feat_cache[key] = feats
        self._feat_cache_bytes += _tensor_nbytes(feats)
        self._trim_feat_cache()

    def _trim_feat_cache(self) -> None:
        while self.feat_cache_size > 0 and len(self._feat_cache) > self.feat_cache_size:
            _, old = self._feat_cache.popitem(last=False)
            self._feat_cache_bytes -= _tensor_nbytes(old)

        while self.feat_cache_max_bytes > 0 and self._feat_cache_bytes > self.feat_cache_max_bytes and self._feat_cache:
            _, old = self._feat_cache.popitem(last=False)
            self._feat_cache_bytes -= _tensor_nbytes(old)

    def preload_caches(
        self,
        preload_feats_gb: float = 0.0,
        preload_embeddings: bool = False,
        sort_by_path: bool = True,
    ) -> None:
        """
        Warm Dataset caches before DataLoader starts.

        For indexed manifests, many rows point to the same full audio_feats file.
        Loading unique files once in path order turns the expensive part from
        repeated random reads into a bounded sequential warmup.
        """

        preload_bytes = max(0, int(float(preload_feats_gb) * 1024**3))
        if preload_bytes > 0 and self.feat_cache_max_bytes <= 0:
            self.feat_cache_max_bytes = preload_bytes

        rows = sorted(self.items, key=lambda item: str(item.get("audio_feats", ""))) if sort_by_path else list(self.items)

        if preload_bytes > 0:
            seen_feats: set[str] = set()
            feat_paths = []
            for item in rows:
                path = _resolve_path(item["audio_feats"], self.manifest_dir)
                key = str(path)
                if key not in seen_feats:
                    seen_feats.add(key)
                    feat_paths.append(path)

            loaded = 0
            for path in tqdm(feat_paths, desc="preloading audio_feats"):
                key = str(path)
                if key in self._feat_cache:
                    continue
                feats = _load_audio_feats(path)
                nbytes = _tensor_nbytes(feats)
                if loaded + nbytes > preload_bytes and loaded > 0:
                    break
                self._put_audio_feats(key, feats)
                loaded += nbytes

            print(
                f"[preload] audio_feats cached={len(self._feat_cache)} "
                f"bytes={self._feat_cache_bytes / 1024**3:.2f}GB"
            )

        if preload_embeddings:
            seen_embeddings: set[str] = set()
            embedding_values = []
            for item in rows:
                value = item["teacher_embedding"]
                if not isinstance(value, str):
                    continue
                path = _resolve_path(value, self.manifest_dir)
                key = str(path)
                if key not in seen_embeddings:
                    seen_embeddings.add(key)
                    embedding_values.append(value)

            old_limit = self.embedding_cache_size
            self.embedding_cache_size = max(self.embedding_cache_size, len(embedding_values))
            for value in tqdm(embedding_values, desc="preloading teacher embeddings"):
                self._get_embedding(value)
            self.embedding_cache_size = max(old_limit, self.embedding_cache_size)
            print(f"[preload] teacher_embeddings cached={len(self._embedding_cache)}")

    def _get_audio_feats(self, path: Path) -> torch.Tensor:
        key = str(path)
        if self._feat_cache_enabled() and key in self._feat_cache:
            feats = self._feat_cache.pop(key)
            self._feat_cache[key] = feats
            return feats

        feats = _load_audio_feats(path)
        self._put_audio_feats(key, feats)
        return feats

    def _get_embedding(self, value: Any) -> torch.Tensor:
        if isinstance(value, str):
            path = _resolve_path(value, self.manifest_dir)
            key = str(path)
            if self.embedding_cache_size > 0 and key in self._embedding_cache:
                emb = self._embedding_cache.pop(key)
                self._embedding_cache[key] = emb
                return emb
            emb = _load_embedding(value, self.manifest_dir).flatten()
            emb = torch.nn.functional.normalize(emb, dim=0)
            if self.embedding_cache_size > 0:
                self._embedding_cache[key] = emb
                while len(self._embedding_cache) > self.embedding_cache_size:
                    self._embedding_cache.popitem(last=False)
            return emb

        emb = _load_embedding(value, self.manifest_dir).flatten()
        return torch.nn.functional.normalize(emb, dim=0)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        feats_path = _resolve_path(item["audio_feats"], self.manifest_dir)
        feats = self._get_audio_feats(feats_path)

        if "chunk_start" in item and "chunk_end" in item:
            start = int(item["chunk_start"])
            end = int(item["chunk_end"])
            feats = feats[start:end]

        if feats.size(0) < self.min_len:
            raise ValueError(f"{feats_path} has only {feats.size(0)} frames; min_len={self.min_len}")

        if self.max_len > 0 and feats.size(0) > self.max_len:
            if self.random_crop:
                start = torch.randint(0, feats.size(0) - self.max_len + 1, ()).item()
            else:
                start = 0
            feats = feats[start : start + self.max_len]

        emb = self._get_embedding(item["teacher_embedding"])
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
