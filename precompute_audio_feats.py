#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VOXCPM_ROOT = WORKSPACE_ROOT / "VoxCPM-main"
sys.path.insert(0, str(VOXCPM_ROOT / "src"))

from voxcpm.model import VoxCPM2Model, VoxCPMModel  # noqa: E402


def load_model(pretrained_path: str):
    cfg_path = Path(pretrained_path) / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        arch = json.load(f).get("architecture", "voxcpm").lower()
    cls = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
    return cls.from_local(pretrained_path, optimize=False, training=True)


def encode_audio(model, wav_path: Path, device: torch.device):
    audio, sr = torchaudio.load(str(wav_path))
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    sample_rate = getattr(model.audio_vae, "sample_rate", model.sample_rate)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    patch_len = model.patch_size * model.audio_vae.hop_length
    if audio.size(1) % patch_len != 0:
        pad = patch_len - audio.size(1) % patch_len
        audio = torch.nn.functional.pad(audio, (0, pad))

    with torch.no_grad():
        z = model.audio_vae.encode(audio.to(device), sample_rate).cpu()  # [1,D,T']
    latent_dim = z.size(1)
    feats = z.view(latent_dim, -1, model.patch_size).permute(1, 2, 0).contiguous()  # [T,P,D]
    return feats


def resolve_audio_path(value: str, manifest_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else manifest_dir / p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--manifest", required=True, help="JSONL with an 'audio' field.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)

    model = load_model(args.pretrained_path).to(device).eval()
    model.audio_vae.to(device).eval()

    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as writer:
        for idx, row in enumerate(tqdm(rows, desc="Encoding AudioVAE latents")):
            wav_path = resolve_audio_path(row["audio"], manifest.parent)
            feats = encode_audio(model, wav_path, device)
            feat_path = out_dir / f"{idx:08d}_feats.pt"
            torch.save({"audio_feats": feats, "source_audio": str(wav_path)}, feat_path)

            out_row = dict(row)
            out_row["audio_feats"] = str(feat_path)
            writer.write(json.dumps(out_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
