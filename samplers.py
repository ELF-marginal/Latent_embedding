from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler


class UtteranceGroupedBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that groups chunks by utterance to improve full-feat cache hits.

    It does not require equal chunk counts per utterance. Each epoch shuffles
    utterances, shuffles chunks within each utterance, then emits batches filled
    from a small window of utterances.
    """

    def __init__(
        self,
        items: list[dict],
        batch_size: int,
        utterances_per_batch: int = 16,
        shuffle: bool = True,
        seed: int = 1234,
        drop_last: bool = False,
    ):
        self.items = items
        self.batch_size = int(batch_size)
        self.utterances_per_batch = max(1, int(utterances_per_batch))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        grouped = defaultdict(list)
        for idx, item in enumerate(items):
            utterance_id = item.get("utterance_id") or item.get("id") or str(idx)
            grouped[str(utterance_id)].append(idx)
        self.grouped = dict(grouped)
        self.utterance_ids = list(self.grouped)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        utterance_ids = list(self.utterance_ids)
        if self.shuffle:
            rng.shuffle(utterance_ids)

        batch = []
        for offset in range(0, len(utterance_ids), self.utterances_per_batch):
            window = utterance_ids[offset : offset + self.utterances_per_batch]
            chunks_by_utt = []
            for utterance_id in window:
                indices = list(self.grouped[utterance_id])
                if self.shuffle:
                    rng.shuffle(indices)
                chunks_by_utt.append(indices)

            active = True
            while active:
                active = False
                for indices in chunks_by_utt:
                    if not indices:
                        continue
                    active = True
                    batch.append(indices.pop())
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.items) // self.batch_size
        return math.ceil(len(self.items) / self.batch_size)

