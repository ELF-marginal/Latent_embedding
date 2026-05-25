#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
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


class SuppressMessageFilter(logging.Filter):
    def __init__(self, suppressed_substrings: tuple[str, ...]):
        super().__init__()
        self.suppressed_substrings = suppressed_substrings

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(text in message for text in self.suppressed_substrings)


def quiet_modelscope_logging():
    suppressed = ("The sample rate of audio is not 16000, resample it.",)
    message_filter = SuppressMessageFilter(suppressed)
    root_logger = logging.getLogger()
    root_logger.addFilter(message_filter)
    for handler in root_logger.handlers:
        handler.addFilter(message_filter)

    for name in ("modelscope", "modelscope.models", "modelscope.pipelines"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.addFilter(message_filter)
        for handler in logger.handlers:
            handler.addFilter(message_filter)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


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
    def __init__(self, model_dir: Path, device: str, expected_dim: int = 512, quiet: bool = True):
        if quiet:
            quiet_modelscope_logging()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.expected_dim = int(expected_dim)
        modelscope_device = "gpu" if device.startswith("cuda") else device
        self.pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=str(model_dir),
            device=modelscope_device,
        )
        if quiet:
            quiet_modelscope_logging()

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


def parse_speaker_id_from_path(wav_path: Path, wav_root: Path, speaker_id_regex: str, fallback: str) -> str:
    if speaker_id_regex:
        match = re.match(speaker_id_regex, wav_path.name)
        if match is None:
            match = re.match(speaker_id_regex, wav_path.stem)
        if match is None:
            raise ValueError(f"speaker_id_regex did not match filename: {wav_path.name}")
        if "speaker_id" in match.groupdict():
            return match.group("speaker_id")
        return match.group(1)

    if fallback == "parent":
        return wav_path.parent.name
    if fallback == "stem":
        return wav_path.stem
    if fallback == "prefix_before_last_underscore":
        parts = wav_path.stem.rsplit("_", 1)
        return parts[0] if len(parts) == 2 else wav_path.stem
    if fallback == "relative_parent":
        try:
            rel_parent = wav_path.parent.relative_to(wav_root)
            return rel_parent.as_posix() if str(rel_parent) != "." else wav_path.parent.name
        except ValueError:
            return wav_path.parent.name
    raise ValueError(f"Unsupported speaker_id_fallback: {fallback}")


def build_rows_from_wav_root(
    wav_root: Path,
    exts: Iterable[str],
    *,
    speaker_id_regex: str = "",
    speaker_id_fallback: str = "parent",
    recursive: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    if recursive:
        wav_paths = find_audio_files(wav_root, exts)
    else:
        ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts}
        wav_paths = sorted(p for p in wav_root.iterdir() if p.is_file() and p.suffix.lower() in ext_set)

    for wav_path in wav_paths:
        rows.append(
            {
                "audio": str(wav_path),
                "speaker_id": parse_speaker_id_from_path(
                    wav_path,
                    wav_root,
                    speaker_id_regex=speaker_id_regex,
                    fallback=speaker_id_fallback,
                ),
            }
        )
    return rows


def resolve_audio_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def get_speaker_id(row: dict[str, Any], wav_path: Path) -> str:
    speaker_id = row.get("speaker_id") or row.get("spk_id") or row.get("speaker")
    return str(speaker_id) if speaker_id is not None else wav_path.parent.name


def chunk_starts(length: int, chunk_size: int, chunk_hop: int) -> list[int]:
    if chunk_size <= 0 or length <= chunk_size:
        return [0]
    starts = list(range(0, length - chunk_size + 1, max(1, chunk_hop)))
    last_start = length - chunk_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a student speaker-embedding dataset: wav -> VoxCPM audio_feats + ERes2Net teacher emb."
    )
    parser.add_argument("--voxcpm_path", default="VoxCPMv1.5", help="Local VoxCPM checkpoint directory.")
    parser.add_argument(
        "--teacher_model",
        default="speech_eres2net_large_sv_zh-cn_3dspeaker_16k",
        help="Local ERes2Net ModelScope model directory.",
    )
    parser.add_argument("--wav_root", default="dataset/train/wav", help="Directory scanned when --input_manifest is empty.")
    parser.add_argument("--input_manifest", default="", help="Optional JSONL with an 'audio' field.")
    parser.add_argument(
        "--speaker_id_regex",
        default="",
        help="Regex applied to filename/stem. Use a named group (?P<speaker_id>...) or the first capture group.",
    )
    parser.add_argument(
        "--speaker_id_fallback",
        default="parent",
        choices=["parent", "stem", "prefix_before_last_underscore", "relative_parent"],
        help="Fallback speaker-id rule when no regex is provided.",
    )
    parser.add_argument("--non_recursive", action="store_true", help="Only scan files directly under wav_root.")
    parser.add_argument("--out_root", default="train_data/student_cache")
    parser.add_argument("--out_manifest", default="train_data/student_train.jsonl")
    parser.add_argument("--chunk_size", type=int, default=50, help="Latent T steps per saved training chunk; <=0 keeps full utterances.")
    parser.add_argument("--chunk_hop", type=int, default=50, help="Hop in latent T steps between chunks.")
    parser.add_argument("--min_chunk_len", type=int, default=25, help="Drop chunks shorter than this many latent T steps.")
    parser.add_argument("--audio_exts", nargs="*", default=sorted(SUPPORTED_AUDIO_EXTS))
    parser.add_argument("--max_audio_files", type=int, default=0, help="Limit the number of source audio files; 0 means all.")
    parser.add_argument("--shuffle_audio_files", action="store_true", help="Shuffle source audio files before applying max_audio_files.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_shards", type=int, default=1, help="Split selected audio files into N shards for parallel data prep.")
    parser.add_argument("--shard_index", type=int, default=0, help="Current shard index in [0, num_shards).")
    parser.add_argument("--sharded_manifest", action="store_true", help="Write out_manifest with a .shard{idx}-of-{N}.jsonl suffix.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher_device", default="", help="Defaults to --device. Use 'cpu' if ModelScope rejects cuda.")
    parser.add_argument("--teacher_dim", type=int, default=512)
    parser.add_argument("--show_modelscope_warnings", action="store_true", help="Show ModelScope warnings such as sample-rate resampling messages.")
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
    utt_emb_dir = out_root / "utterance_teacher_embeddings"
    spk_emb_dir = out_root / "speaker_embeddings"
    feats_dir.mkdir(parents=True, exist_ok=True)
    utt_emb_dir.mkdir(parents=True, exist_ok=True)
    spk_emb_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.input_manifest:
        input_manifest = resolve_audio_path(args.input_manifest, script_dir)
        rows = read_manifest_audio_paths(input_manifest)
        manifest_base = input_manifest.parent
    else:
        rows = build_rows_from_wav_root(
            wav_root,
            args.audio_exts,
            speaker_id_regex=args.speaker_id_regex,
            speaker_id_fallback=args.speaker_id_fallback,
            recursive=not args.non_recursive,
        )
        manifest_base = script_dir

    if not rows:
        raise RuntimeError("No audio files found.")
    if args.shuffle_audio_files:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.max_audio_files and args.max_audio_files > 0:
        rows = rows[: args.max_audio_files]
        print(f"Using {len(rows)} source audio files after max_audio_files={args.max_audio_files}")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    if args.num_shards > 1:
        before = len(rows)
        rows = rows[args.shard_index :: args.num_shards]
        print(f"Using shard {args.shard_index}/{args.num_shards}: {len(rows)} of {before} source audio files")
        if args.sharded_manifest:
            suffix = f".shard{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl"
            out_manifest = out_manifest.with_suffix(suffix)

    device = torch.device(args.device)
    voxcpm = load_voxcpm_model(voxcpm_path, device)

    teacher = None
    if not args.no_teacher:
        teacher_device = args.teacher_device or args.device
        teacher = ModelScopeERes2NetTeacher(
            teacher_model,
            teacher_device,
            expected_dim=args.teacher_dim,
            quiet=not args.show_modelscope_warnings,
        )

    items = []
    for row in rows:
        wav_path = resolve_audio_path(row["audio"], manifest_base)
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)
        item_id = stable_id(wav_path, script_dir)
        speaker_id = get_speaker_id(row, wav_path)
        items.append(
            {
                "row": row,
                "id": item_id,
                "speaker_id": speaker_id,
                "wav_path": wav_path,
            }
        )

    speaker_to_utt_embs: dict[str, list[np.ndarray]] = {}
    if teacher is not None:
        for item in tqdm(items, desc="Extracting utterance teacher embeddings"):
            utt_emb_path = utt_emb_dir / f"{item['id']}.npy"
            if args.skip_existing and utt_emb_path.exists():
                emb = normalize_embedding(np.load(utt_emb_path))
            else:
                emb = teacher.extract(item["wav_path"])
                np.save(utt_emb_path, emb)
            item["utterance_teacher_embedding"] = utt_emb_path
            speaker_to_utt_embs.setdefault(item["speaker_id"], []).append(emb)

        for speaker_id, embeddings in tqdm(speaker_to_utt_embs.items(), desc="Building speaker centroids"):
            spk_emb_path = spk_emb_dir / f"{safe_name(speaker_id)}.npy"
            if args.skip_existing and spk_emb_path.exists():
                speaker_emb = normalize_embedding(np.load(spk_emb_path))
            else:
                stacked = np.stack([normalize_embedding(emb) for emb in embeddings], axis=0)
                speaker_emb = normalize_embedding(stacked.mean(axis=0))
                np.save(spk_emb_path, speaker_emb)
            for item in items:
                if item["speaker_id"] == speaker_id:
                    item["speaker_embedding"] = spk_emb_path

    prepared = []
    skipped_short = 0
    for item in tqdm(items, desc="Encoding and chunking audio_feats"):
        row = item["row"]
        wav_path = item["wav_path"]
        item_id = item["id"]
        speaker_id = item["speaker_id"]

        full_feat_path = feats_dir / f"{item_id}_full.pt"

        if args.skip_existing and full_feat_path.exists():
            full_obj = torch.load(full_feat_path, map_location="cpu", weights_only=False)
            feats = full_obj["audio_feats"]
        else:
            feats = encode_audio_feats(voxcpm, wav_path, device)
            torch.save(
                {
                    "audio_feats": feats,
                    "source_audio": str(wav_path),
                    "patch_size": int(voxcpm.patch_size),
                    "feat_dim": int(feats.size(-1)),
                    "length": int(feats.size(0)),
                },
                full_feat_path,
            )

        total_len = int(feats.size(0))
        starts = chunk_starts(total_len, args.chunk_size, args.chunk_hop)
        for chunk_index, start in enumerate(starts):
            end = total_len if args.chunk_size <= 0 else min(total_len, start + args.chunk_size)
            chunk = feats[start:end].contiguous()
            if chunk.size(0) < args.min_chunk_len:
                skipped_short += 1
                continue

            chunk_id = f"{item_id}_chunk{chunk_index:04d}"
            chunk_path = feats_dir / f"{chunk_id}.pt"
            if not (args.skip_existing and chunk_path.exists()):
                torch.save(
                    {
                        "audio_feats": chunk,
                        "source_audio": str(wav_path),
                        "source_full_audio_feats": str(full_feat_path),
                        "speaker_id": speaker_id,
                        "chunk_index": int(chunk_index),
                        "chunk_start": int(start),
                        "chunk_end": int(end),
                        "patch_size": int(voxcpm.patch_size),
                        "feat_dim": int(chunk.size(-1)),
                        "length": int(chunk.size(0)),
                    },
                    chunk_path,
                )

            out_row = {
                "id": chunk_id,
                "utterance_id": item_id,
                "speaker_id": speaker_id,
                "audio": str(wav_path),
                "audio_feats": str(chunk_path),
                "length": int(chunk.size(0)),
                "chunk_index": int(chunk_index),
                "chunk_start": int(start),
                "chunk_end": int(end),
            }
            if teacher is not None:
                out_row["teacher_embedding"] = str(item["speaker_embedding"])
                out_row["speaker_embedding"] = str(item["speaker_embedding"])
                out_row["utterance_teacher_embedding"] = str(item["utterance_teacher_embedding"])
            prepared.append(out_row)

    with out_manifest.open("w", encoding="utf-8") as f:
        for row in prepared:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prepared)} rows to {out_manifest}")
    if skipped_short:
        print(f"Skipped {skipped_short} chunks shorter than min_chunk_len={args.min_chunk_len}")
    print(f"Audio feat chunks: {feats_dir}")
    if teacher is not None:
        print(f"Utterance teacher embeddings: {utt_emb_dir}")
        print(f"Speaker centroid embeddings: {spk_emb_dir}")


if __name__ == "__main__":
    main()
