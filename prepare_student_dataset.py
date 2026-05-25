#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VOXCPM_ROOT = WORKSPACE_ROOT / "VoxCPM-main"
sys.path.insert(0, str(VOXCPM_ROOT / "src"))

from voxcpm.model import VoxCPM2Model, VoxCPMModel  # noqa: E402


SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def stable_id(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.resolve().as_posix()
    stem = path.stem.replace(" ", "_")
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def find_audio_files(wav_root: Path, exts: Iterable[str]) -> list[Path]:
    ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts}
    return sorted(p for p in wav_root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set)


def load_voxcpm_model(pretrained_path: Path, device: torch.device):
    cfg_path = pretrained_path / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        arch = json.load(f).get("architecture", "voxcpm").lower()
    model_cls = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
    model = model_cls.from_local(str(pretrained_path), optimize=False, training=True)
    model = model.to(device).eval()
    model.audio_vae = model.audio_vae.to(device).eval()
    return model


def encode_audio_feats(model, wav_path: Path, device: torch.device) -> torch.Tensor:
    audio, sr = torchaudio.load(str(wav_path))
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    sample_rate = int(model.audio_vae.sample_rate)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    patch_len = int(model.patch_size * model.audio_vae.hop_length)
    if audio.size(-1) % patch_len != 0:
        pad = patch_len - audio.size(-1) % patch_len
        audio = torch.nn.functional.pad(audio, (0, pad))

    with torch.no_grad():
        z = model.audio_vae.encode(audio.to(device), sample_rate).cpu()  # [1, D, T_vae]

    latent_dim = z.size(1)
    if z.size(-1) % model.patch_size != 0:
        pad = model.patch_size - z.size(-1) % model.patch_size
        z = torch.nn.functional.pad(z, (0, pad))
    feats = z.squeeze(0).view(latent_dim, -1, model.patch_size).permute(1, 2, 0).contiguous()
    return feats  # [T, P, D]


def normalize_embedding(embedding: Any) -> np.ndarray:
    if isinstance(embedding, torch.Tensor):
        arr = embedding.detach().cpu().float().numpy()
    else:
        arr = np.asarray(embedding, dtype=np.float32)
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D embedding, got shape {arr.shape}")
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def find_embedding_in_object(obj: Any, expected_dim: int) -> Any | None:
    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
        if tensor.ndim == 1 and tensor.numel() == expected_dim:
            return tensor
        if tensor.ndim == 2 and expected_dim in tensor.shape:
            return tensor.reshape(-1, expected_dim)[0]
        return None

    if isinstance(obj, np.ndarray):
        arr = np.squeeze(obj)
        if arr.ndim == 1 and arr.shape[0] == expected_dim:
            return arr
        if arr.ndim == 2 and expected_dim in arr.shape:
            return arr.reshape(-1, expected_dim)[0]
        return None

    if isinstance(obj, dict):
        preferred = (
            "embedding",
            "emb",
            "spk_embedding",
            "speaker_embedding",
            "xvector",
            "outputs",
            "output",
        )
        for key in preferred:
            if key in obj:
                found = find_embedding_in_object(obj[key], expected_dim)
                if found is not None:
                    return found
        for value in obj.values():
            found = find_embedding_in_object(value, expected_dim)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, (int, float, np.number)) for x in obj):
            arr = np.asarray(obj, dtype=np.float32)
            if arr.ndim == 1 and arr.shape[0] == expected_dim:
                return arr
        for value in obj:
            found = find_embedding_in_object(value, expected_dim)
            if found is not None:
                return found
        return None

    return None


class ModelScopeERes2NetTeacher:
    def __init__(self, model_dir: Path, device: str, expected_dim: int = 512):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.expected_dim = int(expected_dim)
        modelscope_device = "gpu" if device.startswith("cuda") else device
        self.pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=str(model_dir),
            device=modelscope_device,
        )

    def extract(self, wav_path: Path) -> np.ndarray:
        wav = str(wav_path)
        attempts = []
        errors = []

        for call in (
            lambda: self.pipeline(wav, output_emb=True),
            lambda: self.pipeline([wav], output_emb=True),
            lambda: self.pipeline([wav, wav], output_emb=True),
            lambda: self.pipeline.preprocess(wav),
            lambda: self.pipeline(wav),
            lambda: self.pipeline([wav]),
            lambda: self.pipeline([wav, wav]),
        ):
            try:
                attempts.append(call())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        if attempts:
            for obj in attempts:
                found = find_embedding_in_object(obj, self.expected_dim)
                if found is not None:
                    return normalize_embedding(found)

        if hasattr(self.pipeline, "preprocess") and hasattr(self.pipeline, "forward"):
            try:
                data = self.pipeline.preprocess(wav)
                for output in (
                    self.pipeline.forward(data),
                    self.pipeline.forward(data, output_emb=True),
                ):
                    found = find_embedding_in_object(output, self.expected_dim)
                    if found is not None:
                        return normalize_embedding(found)
            except Exception as exc:
                errors.append(f"forward {type(exc).__name__}: {exc}")

        model = getattr(self.pipeline, "model", None)
        if model is not None:
            for method_name in ("extract_embedding", "get_embedding", "encode", "inference"):
                method = getattr(model, method_name, None)
                if method is None:
                    continue
                try:
                    output = method(wav)
                    found = find_embedding_in_object(output, self.expected_dim)
                    if found is not None:
                        return normalize_embedding(found)
                except Exception as exc:
                    errors.append(f"{method_name} {type(exc).__name__}: {exc}")

        debug = []
        for obj in attempts[-3:]:
            if isinstance(obj, dict):
                debug.append(f"dict keys={list(obj.keys())}")
            else:
                debug.append(f"{type(obj).__name__}: {str(obj)[:300]}")

        raise RuntimeError(
            "Could not extract a 512-D embedding from the ModelScope ERes2Net pipeline. "
            "If your local ModelScope version only exposes verification scores, generate "
            "teacher embeddings with 3D-Speaker's infer_sv.py and pass them through a manifest. "
            f"Recent outputs: {debug}. Recent errors: {errors[-5:]}"
        )


def read_manifest_audio_paths(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "audio" not in row:
                raise KeyError(f"Manifest row has no 'audio': {row}")
            rows.append(row)
    return rows


def build_rows_from_wav_root(wav_root: Path, exts: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for wav_path in find_audio_files(wav_root, exts):
        rows.append(
            {
                "audio": str(wav_path),
                "speaker_id": wav_path.parent.name,
            }
        )
    return rows


def resolve_audio_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a student speaker-embedding dataset: wav -> VoxCPM audio_feats + ERes2Net teacher emb."
    )
    parser.add_argument("--voxcpm_path", default="VoxCPM2", help="Local VoxCPM checkpoint directory.")
    parser.add_argument(
        "--teacher_model",
        default="speech_eres2net_large_sv_zh-cn_3dspeaker_16k",
        help="Local ERes2Net ModelScope model directory.",
    )
    parser.add_argument("--wav_root", default="dataset/train/wav", help="Directory scanned when --input_manifest is empty.")
    parser.add_argument("--input_manifest", default="", help="Optional JSONL with an 'audio' field.")
    parser.add_argument("--out_root", default="dataset/train/student_cache")
    parser.add_argument("--out_manifest", default="dataset/train/student_train.jsonl")
    parser.add_argument("--audio_exts", nargs="*", default=sorted(SUPPORTED_AUDIO_EXTS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_device", default="", help="Defaults to --device. Use 'cpu' if ModelScope rejects cuda.")
    parser.add_argument("--teacher_dim", type=int, default=512)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_teacher", action="store_true", help="Only create audio_feats; useful for debugging AudioVAE.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    voxcpm_path = resolve_audio_path(args.voxcpm_path, script_dir)
    teacher_model = resolve_audio_path(args.teacher_model, script_dir)
    wav_root = resolve_audio_path(args.wav_root, script_dir)
    out_root = resolve_audio_path(args.out_root, script_dir)
    out_manifest = resolve_audio_path(args.out_manifest, script_dir)

    feats_dir = out_root / "audio_feats"
    emb_dir = out_root / "teacher_embeddings"
    feats_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.input_manifest:
        input_manifest = resolve_audio_path(args.input_manifest, script_dir)
        rows = read_manifest_audio_paths(input_manifest)
        manifest_base = input_manifest.parent
    else:
        rows = build_rows_from_wav_root(wav_root, args.audio_exts)
        manifest_base = script_dir

    if not rows:
        raise RuntimeError("No audio files found.")

    device = torch.device(args.device)
    voxcpm = load_voxcpm_model(voxcpm_path, device)

    teacher = None
    if not args.no_teacher:
        teacher_device = args.teacher_device or args.device
        teacher = ModelScopeERes2NetTeacher(teacher_model, teacher_device, expected_dim=args.teacher_dim)

    prepared = []
    for index, row in enumerate(tqdm(rows, desc="Preparing student dataset")):
        wav_path = resolve_audio_path(row["audio"], manifest_base)
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)

        item_id = stable_id(wav_path, script_dir)
        feat_path = feats_dir / f"{item_id}.pt"
        emb_path = emb_dir / f"{item_id}.npy"

        if not (args.skip_existing and feat_path.exists()):
            feats = encode_audio_feats(voxcpm, wav_path, device)
            torch.save(
                {
                    "audio_feats": feats,
                    "source_audio": str(wav_path),
                    "patch_size": int(voxcpm.patch_size),
                    "feat_dim": int(feats.size(-1)),
                    "length": int(feats.size(0)),
                },
                feat_path,
            )

        if teacher is not None and not (args.skip_existing and emb_path.exists()):
            emb = teacher.extract(wav_path)
            np.save(emb_path, emb)

        out_row = {
            "id": item_id,
            "audio": str(wav_path),
            "audio_feats": str(feat_path),
            "length": int(torch.load(feat_path, map_location="cpu", weights_only=False)["length"]),
        }
        if "speaker_id" in row:
            out_row["speaker_id"] = row["speaker_id"]
        if teacher is not None:
            out_row["teacher_embedding"] = str(emb_path)
        prepared.append(out_row)

    with out_manifest.open("w", encoding="utf-8") as f:
        for row in prepared:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prepared)} rows to {out_manifest}")
    print(f"Audio feats: {feats_dir}")
    if teacher is not None:
        print(f"Teacher embeddings: {emb_dir}")


if __name__ == "__main__":
    main()
