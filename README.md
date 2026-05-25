# Latent Speaker Embedding

This folder trains a student speaker encoder on VoxCPM AudioVAE latents.

The intended input is a padded batch of variable-length, patchized AudioVAE
features:

```text
audio_feats: [B, T, P, D]
lengths:     [B]
```

`T` is variable per utterance. Batches are padded on `T`, and the model uses
`lengths` to ignore padding during pooling.

The teacher target is an external speaker embedding, for example one extracted
from ERes2Net for the same audio. The student learns to map VoxCPM latents into
that teacher embedding space.

## Files

- `models.py`: `LatentSpeakerEncoder`, suitable for later use as a frozen
  `loss_spk` network.
- `prepare_student_dataset.py`: scans wav files and creates the full student
  training dataset: VoxCPM AudioVAE `audio_feats` plus ERes2Net teacher
  embeddings.
- `precompute_audio_feats.py`: converts manifest audio into VoxCPM AudioVAE
  latent patch files.
- `train_latent_speaker.py`: trains the student model from precomputed latents
  and teacher embeddings.
- `infer_embedding.py`: extracts a student embedding from one latent file.

## Expected Training Manifest

After precomputing latents and teacher embeddings, train with a JSONL manifest:

```json
{"audio_feats": "cache/000001_feats.pt", "teacher_embedding": "cache/000001_spk.npy"}
{"audio_feats": "cache/000002_feats.pt", "teacher_embedding": [0.01, -0.02, 0.3]}
```

`audio_feats` can be a `.pt` file containing either `[T, P, D]` directly or a
dict with key `audio_feats`. `teacher_embedding` can be a list, `.npy`, `.pt`, or
a dict `.pt` with key `embedding`.

## Train

First prepare the student dataset. With the current local folder layout, the
defaults point to:

- `VoxCPM2/` for the VoxCPM AudioVAE.
- `speech_eres2net_large_sv_zh-cn_3dspeaker_16k/` for the ERes2Net teacher.
- `dataset/train/wav/` for source audio.

```bash
python prepare_student_dataset.py --skip_existing
```

This writes:

```text
train_data/student_cache/audio_feats/*.pt
train_data/student_cache/teacher_embeddings/*.npy
train_data/student_train.jsonl
```

```bash
python train_latent_speaker.py ^
  --train_manifest train_data/student_train.jsonl ^
  --save_dir checkpoints/latent_spk ^
  --embedding_dim 512
```

## Later VoxCPM Integration Shape

The frozen loss network can be used like this:

```python
encoder = LatentSpeakerEncoder.from_checkpoint("latent_speaker_encoder.pt")
encoder.eval().requires_grad_(False)

emb_pred = encoder(feat_pred, pred_lengths)
emb_gt = encoder(feat_gt, gt_lengths).detach()
loss_spk = 1.0 - torch.nn.functional.cosine_similarity(emb_pred, emb_gt, dim=-1).mean()
```
