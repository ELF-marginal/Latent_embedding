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
- `build_test_data.py`: builds a held-out `test_data/` directory on demand.
- `test_latent_speaker.py`: evaluates a trained student checkpoint on a test
  manifest.
- `build_dataset_splits.py`: creates speaker-disjoint train/test audio splits.

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

- `VoxCPMv1.5/` for the VoxCPM AudioVAE.
- `speech_eres2net_large_sv_zh-cn_3dspeaker_16k/` for the ERes2Net teacher.
- `dataset/train/wav/` for source audio.

```bash
python prepare_student_dataset.py --skip_existing --chunk_size 50 --chunk_hop 50 --chunk_storage indexed
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
  --chunk_storage indexed \
  --skip_existing
```

For a quick randomized 200-file smoke run:

```bash
python prepare_student_dataset.py \
  --wav_root /home/lqh/datasets/momo_5000h/audio \
  --audio_exts .wav \
  --speaker_id_regex '^(?P<speaker_id>.+)_[^_]+$' \
  --out_root train_data/momo_5000h_200_cache \
  --out_manifest train_data/momo_5000h_200_train.jsonl \
  --chunk_size 50 \
  --chunk_hop 50 \
  --min_chunk_len 25 \
  --chunk_storage indexed \
  --max_audio_files 200 \
  --shuffle_audio_files \
  --seed 1234 \
  --skip_existing
```

For two-GPU sharded data preparation, run one process per GPU and merge the
shard manifests afterwards:

```bash
CUDA_VISIBLE_DEVICES=0 python prepare_student_dataset.py \
  --wav_root /home/lqh/datasets/momo_5000h/audio \
  --audio_exts .wav \
  --speaker_id_regex '^(?P<speaker_id>.+)_[^_]+$' \
  --out_root train_data/momo_5000h_cache \
  --out_manifest train_data/momo_5000h_train.jsonl \
  --chunk_size 50 \
  --chunk_hop 50 \
  --min_chunk_len 25 \
  --chunk_storage indexed \
  --num_shards 2 \
  --shard_index 0 \
  --sharded_manifest \
  --skip_existing

CUDA_VISIBLE_DEVICES=1 python prepare_student_dataset.py \
  --wav_root /home/lqh/datasets/momo_5000h/audio \
  --audio_exts .wav \
  --speaker_id_regex '^(?P<speaker_id>.+)_[^_]+$' \
  --out_root train_data/momo_5000h_cache \
  --out_manifest train_data/momo_5000h_train.jsonl \
  --chunk_size 50 \
  --chunk_hop 50 \
  --min_chunk_len 25 \
  --chunk_storage indexed \
  --num_shards 2 \
  --shard_index 1 \
  --sharded_manifest \
  --skip_existing

python merge_manifests.py \
  --inputs 'train_data/momo_5000h_train.shard*-of-*.jsonl' \
  --output train_data/momo_5000h_train.jsonl
```

For a proper speaker-disjoint train/test split, first create audio split files:

```bash
python build_dataset_splits.py \
  --wav_root /home/lqh/datasets/momo_5000h/audio \
  --speaker_id_regex '^(?P<speaker_id>.+)_[^_]+$' \
  --num_train_audio 5000 \
  --num_test_audio 1000 \
  --sample_strategy balanced \
  --min_files_per_speaker 1 \
  --out_dir splits/momo_5000h \
  --seed 1234 \
  --overwrite
```

Then build train features:

```bash
python prepare_student_dataset.py \
  --input_manifest splits/momo_5000h/train_audio.jsonl \
  --out_root train_data/momo_5000h_cache \
  --out_manifest train_data/momo_5000h_train.jsonl \
  --chunk_size 50 \
  --chunk_hop 50 \
  --min_chunk_len 25 \
  --chunk_storage indexed \
  --skip_existing
```

And build test features:

```bash
python prepare_student_dataset.py \
  --input_manifest splits/momo_5000h/test_audio.jsonl \
  --out_root test_data/momo_5000h_cache \
  --out_manifest test_data/test_manifest.jsonl \
  --chunk_size 50 \
  --chunk_hop 50 \
  --min_chunk_len 25 \
  --chunk_storage indexed \
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

## Test

Build a held-out 200-file test set. This creates `test_data/` only when run; if
the directory already exists, pass `--overwrite` to replace it.

```bash
python build_test_data.py \
  --wav_root /home/lqh/datasets/momo_5000h/audio \
  --exclude_manifest train_data/momo_5000h_200_train.jsonl \
  --num_audio 200 \
  --seed 2026 \
  --overwrite
```

Evaluate a model checkpoint:

```bash
python test_latent_speaker.py \
  --checkpoint checkpoints/latent_spk_momo_5000h_200/latest.pt \
  --manifest test_data/test_manifest.jsonl \
  --batch_size 32
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
