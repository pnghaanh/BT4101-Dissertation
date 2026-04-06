import hashlib
import math
import random
import struct
from typing import Dict, List, Set

import torch
from scipy.stats import binom
from transformers import LogitsProcessor


class PCMWatermarkLogitsProcessor(LogitsProcessor):
    """PCM Watermark generation with block-anchor color control."""

    def __init__(
        self,
        vocab_size: int = 50257, # designed for GPT-2 tokenizer, but can be adapted for others
        gamma: float = 0.5,
        delta: float = 2.0,
        block_size: int = 8,
        warmup: int = 40,
        topk_bias: int = 96,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.block_size = block_size
        self.warmup = warmup
        self.topk_bias = max(1, int(topk_bias))
        self.rng = torch.Generator()
        self.secret_key = hashlib.sha256(b"pcm_watermark").digest()[:32]
        self.pattern_bits = self._get_pattern_bits()

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

    def _build_greenlist_prefix(self, prefix: List[int]) -> List[int]:
        return prefix[-4:] if len(prefix) >= 4 else prefix

    def _get_pattern_bits(self) -> List[int]:
        key_int = int.from_bytes(self.secret_key[:4], "big")
        rng = random.Random(key_int)
        return [rng.randint(0, 1) for _ in range(self.block_size)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            prefix = input_ids[batch_idx].tolist()
            if len(prefix) < self.warmup:
                continue

            membership_idx = len(prefix) - self.warmup
            block_pos = membership_idx % self.block_size
            if block_pos == 0:
                # First token in a block becomes the anchor for later tokens in that block.
                continue

            block_start_idx = membership_idx - block_pos
            anchor_token_pos = self.warmup + block_start_idx
            if anchor_token_pos >= len(prefix):
                continue

            greenlist_prefix = self._build_greenlist_prefix(prefix)
            greenlist = self._get_greenlist(greenlist_prefix)

            anchor_token = prefix[anchor_token_pos]
            anchor_colour_green = anchor_token in greenlist
            pattern_bit = self.pattern_bits[block_pos]
            target_green = anchor_colour_green if pattern_bit == 1 else (not anchor_colour_green)

            green_mask = torch.zeros_like(scores[batch_idx], dtype=torch.bool)
            for token_id in greenlist:
                if token_id < scores.shape[-1]:
                    green_mask[token_id] = True

            if target_green:
                # Double probability mass for green tokens by adding log(2) to their logits.
                scores[batch_idx][green_mask] += math.log(2.0)
            else:
                # If target is non-green, set green token probability to zero.
                scores[batch_idx][green_mask] = -float("inf")

        return scores


class PCMWatermarkDetector:
    """PCM Watermark detector for anchor-color consistency within each block."""

    def __init__(
        self,
        vocab_size: int = 50257,
        gamma: float = 0.5,
        block_size: int = 8,
        warmup: int = 40,
        z_threshold: float = 1.96,
        block_match_threshold: float = 0.6875,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.block_size = block_size
        self.warmup = warmup
        self.z_threshold = z_threshold
        self.effective_block_size = self.block_size
        self.block_match_threshold = block_match_threshold
        self.min_block_matches = max(1, int(math.ceil(self.effective_block_size * self.block_match_threshold)))
        self.rng = torch.Generator()
        self.secret_key = hashlib.sha256(b"pcm_watermark").digest()[:32]
        self.pattern_bits = self._get_pattern_bits()
        self.p0_block_hit = sum(
            math.comb(self.effective_block_size, k)
            for k in range(self.min_block_matches, self.effective_block_size + 1)
        ) / float(2 ** self.effective_block_size)

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

    def _build_greenlist_prefix(self, token_ids: List[int], token_pos: int) -> List[int]:
        start = max(0, token_pos - 4)
        return token_ids[start:token_pos]

    def _get_pattern_bits(self) -> List[int]:
        key_int = int.from_bytes(self.secret_key[:4], "big")
        rng = random.Random(key_int)
        return [rng.randint(0, 1) for _ in range(self.block_size)]

    def detect(self, token_ids: List[int], return_debug: bool = False) -> Dict:
        consistencies = []
        green_count = 0

        for i in range(self.warmup, len(token_ids)):
            membership_idx = i - self.warmup
            block_pos = membership_idx % self.effective_block_size
            if block_pos == 0:
                continue

            block_start_idx = membership_idx - block_pos
            anchor_token_pos = self.warmup + block_start_idx
            if anchor_token_pos >= i:
                continue

            greenlist_prefix = self._build_greenlist_prefix(token_ids, i)
            greenlist = self._get_greenlist(greenlist_prefix)

            anchor_token = token_ids[anchor_token_pos]
            anchor_colour_green = anchor_token in greenlist
            token_colour_green = token_ids[i] in greenlist
            green_count += 1 if token_colour_green else 0

            pattern_bit = self.pattern_bits[block_pos]
            same_colour = token_colour_green == anchor_colour_green
            consistent = same_colour if pattern_bit == 1 else (not same_colour)
            consistencies.append(1 if consistent else 0)

        if len(consistencies) < self.effective_block_size:
            result = {"detected": False, "p_value": 1.0, "z_score": 0.0, "pattern_confidence": 0.0}
            if return_debug:
                result["consistencies"] = consistencies
                result["blocks"] = []
                result["phase"] = 0
            return result

        n_total = len(consistencies)
        consistency_sum = sum(consistencies)
        consistency_z = (consistency_sum - 0.5 * n_total) / math.sqrt(n_total * 0.25)

        best = {
            "p_value": 1.0,
            "z_score": 0.0,
            "phase": 0,
            "pattern_confidence": 0.0,
            "block_hits": 0,
            "trials": 0,
            "blocks": [],
        }

        bonf = max(1.0, math.log2(self.effective_block_size))

        for phase in range(self.effective_block_size):
            block_hits = 0
            block_sims = []
            blocks = []

            for start in range(phase, len(consistencies) - self.effective_block_size + 1, self.effective_block_size):
                block = consistencies[start : start + self.effective_block_size]
                matches = sum(block)
                sim = matches / self.effective_block_size
                block_sims.append(sim)
                blocks.append(block)
                if matches >= self.min_block_matches:
                    block_hits += 1

            trials = len(block_sims)
            if trials == 0:
                continue

            p_value = float(1.0 - binom.cdf(block_hits - 1, trials, self.p0_block_hit))
            p_value = min(1.0, p_value * bonf)

            mu = trials * self.p0_block_hit
            var = trials * self.p0_block_hit * (1.0 - self.p0_block_hit)
            block_z = (block_hits - mu) / math.sqrt(var) if var > 0 else 0.0
            confidence = sum(block_sims) / len(block_sims) if block_sims else 0.0

            if (p_value < best["p_value"]) or (
                p_value == best["p_value"] and confidence > best["pattern_confidence"]
            ):
                best = {
                    "p_value": p_value,
                    "z_score": float(block_z),
                    "phase": int(phase),
                    "pattern_confidence": float(confidence),
                    "block_hits": int(block_hits),
                    "trials": int(trials),
                    "blocks": blocks,
                }

        combined_z = max(best["z_score"], consistency_z)

        result = {
            "detected": combined_z > self.z_threshold,
            "p_value": best["p_value"],
            "z_score": combined_z,
            "pattern_confidence": best["pattern_confidence"],
            "green_fraction": float(green_count / max(1, len(consistencies))),
        }

        if return_debug:
            result["consistencies"] = consistencies
            result["blocks"] = best["blocks"]
            result["phase"] = best["phase"]
            result["block_hits"] = best["block_hits"]
            result["trials"] = best["trials"]
            result["pattern"] = self.pattern_bits
            result["consistency_z"] = consistency_z
            result["block_z"] = best["z_score"]

        return result
