import hashlib
import struct
from typing import Dict, List, Set

import numpy as np
import scipy.stats
import torch
from transformers import LogitsProcessor


class BaselineWatermarkLogitsProcessor(LogitsProcessor):
    """Simple baseline watermark using greenlist/redlist biasing."""

    def __init__(self, vocab_size: int = 50257, gamma: float = 0.5, delta: float = 1.0):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.rng = torch.Generator()
        self.secret_key = hashlib.sha256(b"baseline_watermark").digest()[:32]

    def _hash_to_seed(self, prefix: List[int]) -> int:
        buf = bytearray()
        for t in prefix:
            buf.extend(struct.pack(">I", int(t)))
        h = hashlib.sha256(self.secret_key + buf).digest()
        return int.from_bytes(h[:8], "big")

    def _get_greenlist(self, prefix: List[int]) -> Set[int]:
        seed = self._hash_to_seed(prefix)
        self.rng.manual_seed(seed % (2**64 - 1))

        greenlist_size = int(self.gamma * self.vocab_size)
        greenlist = torch.randperm(self.vocab_size, generator=self.rng)[:greenlist_size]
        return set(greenlist.tolist())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            prefix = input_ids[batch_idx].tolist()
            greenlist = self._get_greenlist(prefix)

            greenlist_mask = torch.zeros_like(scores[batch_idx], dtype=torch.bool)
            for token_id in greenlist:
                if token_id < scores.shape[-1]:
                    greenlist_mask[token_id] = True

            scores[batch_idx][greenlist_mask] += self.delta

        return scores


class BaselineWatermarkDetector:
    """Detector for baseline watermark."""

    def __init__(self, vocab_size: int = 50257, gamma: float = 0.5):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.rng = torch.Generator()
        self.secret_key = hashlib.sha256(b"baseline_watermark").digest()[:32]

    def _hash_to_seed(self, prefix: List[int]) -> int:
        buf = bytearray()
        for t in prefix:
            buf.extend(struct.pack(">I", int(t)))
        h = hashlib.sha256(self.secret_key + buf).digest()
        return int.from_bytes(h[:8], "big")

    def _get_greenlist(self, prefix: List[int]) -> Set[int]:
        seed = self._hash_to_seed(prefix)
        self.rng.manual_seed(seed % (2**64 - 1))

        greenlist_size = int(self.gamma * self.vocab_size)
        greenlist = torch.randperm(self.vocab_size, generator=self.rng)[:greenlist_size]
        return set(greenlist.tolist())

    def detect(self, token_ids: List[int], z_threshold: float = 1.96) -> Dict:
        memberships = []

        for i in range(1, len(token_ids)):
            prefix = token_ids[:i]
            greenlist = self._get_greenlist(prefix)
            token_id = token_ids[i]
            memberships.append(token_id in greenlist)

        if len(memberships) == 0:
            return {"detected": False, "z_score": 0.0, "p_value": 1.0, "green_fraction": 0.0}

        num_green = sum(memberships)
        num_total = len(memberships)
        green_fraction = num_green / num_total

        expected_green = self.gamma
        variance = num_total * expected_green * (1 - expected_green)

        if variance > 0:
            z_score = (num_green - num_total * expected_green) / np.sqrt(variance)
        else:
            z_score = 0.0

        p_value = 1 - scipy.stats.norm.cdf(z_score) if z_score > 0 else 1.0

        return {
            "detected": z_score > z_threshold,
            "z_score": z_score,
            "p_value": p_value,
            "green_fraction": green_fraction,
        }
