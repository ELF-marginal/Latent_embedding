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
  training dataset. It caches utterance-level ERes2Net embeddings, averages them
  into one normalized centroid per speaker, and writes chunk-level VoxCPM
  AudioVAE `audio_feats` supervised by that speaker centroid.
- `precompute_audio_feats.py`: converts manifest audio into VoxCPM AudioVAE
  latent patch files.
- `train_latent_speaker.py`: trains the student model from precomputed latents
  and teacher embeddings.
- `infer_embedding.py`: extracts a student embedding from one latent file.

## Expected Training Manifest

After precomputing latents and speaker centroids, train with a JSONL manifest.
Each row is one latent chunk, not necessarily one full utterance:

```json
{"id": "spk001_utt001_chunk0000", "speaker_id": "spk001", "audio_feats": "train_data/student_cache/audio_feats/spk001_utt001_chunk0000.pt", "teacher_embedding": "train_data/student_cache/speaker_embeddings/spk001.npy", "length": 100}
```

`audio_feats` can be a `.pt` file containing either `[T, P, D]` directly or a
dict with key `audio_feats`. `teacher_embedding` points to the speaker-level
centroid embedding: `normalize(mean(normalize(ERes2Net(wav_i))))` over all
utterances for the same speaker.

## Train

First prepare the student dataset. With the current local folder layout, the
defaults point to:

- `VoxCPM2/` for the VoxCPM AudioVAE.
- `speech_eres2net_large_sv_zh-cn_3dspeaker_16k/` for the ERes2Net teacher.
- `dataset/train/wav/` for source audio.

```bash
python prepare_student_dataset.py --skip_existing --chunk_size 50 --chunk_hop 50
```

For the flat momo 5000h wav directory, where filenames look like
`00429834_Session_960956110_0_S1_c2b6ba0df049.wav` and the speaker id is
`00429834_Session_960956110_0_S1`:

```bash
python prepare_student_dataset.py \
  --wav_root /home/lqh/datasets/momo_5000h/audio \
  --audio_exts .wav \
  --speaker_id_regex '^(?P<speaker_id>.+)_[^_]+$' \
  --out_root train_data/momo_5000h_cache \
  --out_manifest train_data/momo_5000h_train.jsonl \
  --chunk_size 50 \
  --chunk_hop 50 \
  --min_chunk_len 25 \
  --skip_existing
```

This writes:

```text
train_data/student_cache/audio_feats/*.pt
train_data/student_cache/utterance_teacher_embeddings/*.npy
train_data/student_cache/speaker_embeddings/*.npy
train_data/student_train.jsonl
```

```bash
python train_latent_speaker.py --config configs/latent_speaker_default.json
```

For the momo 5000h manifest:

```bash
python train_latent_speaker.py --config configs/latent_speaker_momo_5000h.json
```

CLI arguments override config values:

```bash
python train_latent_speaker.py ^
  --config configs/latent_speaker_default.json ^
  --batch_size 32 ^
  --epochs 50
```

Resume from the latest training checkpoint:

```bash
python train_latent_speaker.py ^
  --config configs/latent_speaker_default.json ^
  --resume latest
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
