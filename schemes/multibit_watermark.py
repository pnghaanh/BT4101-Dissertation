import hashlib
import random
import struct
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.stats
import torch
from transformers import LogitsProcessor


class MultibitWatermarkBase:
    """Base class for multi-bit watermarking (Yoo style colorlist partitioning)."""

    def __init__(
        self,
        vocab_size: int = 50257,
        gamma: float = 0.5,
        delta: float = 2.0,
        base: int = 2,
        message_length: int = 4,
        code_length: int = 4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.base = base

        self.original_msg_length = message_length
        self.message_length = max(message_length, code_length)

        decimal = int("1" * message_length, 2)
        self.converted_msg_length = len(self._numberToBase(decimal, base))

        assert 1 // gamma >= base, (
            f"Only {1 // gamma} chunks available with gamma={gamma}, but base is {base}"
        )

        self.message = None
        self.converted_message = None
        self.bit_position = 0

        self.rng = torch.Generator()
        self.secret_key = hashlib.sha256(b"multibit_watermark_yoo").digest()[:32]

        # Use a random binary message with the configured length by default.
        self.set_message(self._generate_random_binary_message(self.original_msg_length))

    @staticmethod
    def _generate_random_binary_message(length: int) -> str:
        length = max(1, int(length))
        return "".join(str(random.randint(0, 1)) for _ in range(length))

    @staticmethod
    def _numberToBase(n, b):
        if n == 0:
            return "0"
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return "".join(map(str, digits[::-1]))

    def _convert_binary_to_base(self, binary_msg: str):
        decimal = int(binary_msg, 2)
        converted_msg = self._numberToBase(decimal, self.base)
        converted_msg = "0" * (self.converted_msg_length - len(converted_msg)) + converted_msg
        return converted_msg

    def set_message(self, binary_msg: str = ""):
        self.message = binary_msg
        if binary_msg:
            self.converted_message = self._convert_binary_to_base(binary_msg)
            self.converted_msg_length = len(self.converted_message)
        else:
            self.converted_message = None

    def get_current_bit(self, bit_position: int) -> int:
        if self.converted_message:
            idx = (bit_position - 1) % len(self.converted_message)
            return int(self.converted_message[idx])
        return 0

    def _hash_to_seed(self, prefix: List[int]) -> int:
        buf = bytearray()
        for t in prefix:
            buf.extend(struct.pack(">I", int(t)))
        h = hashlib.sha256(self.secret_key + buf).digest()
        return int.from_bytes(h[:8], "big")

    def _get_colorlists(self, prefix: List[int]) -> List[Set[int]]:
        seed = self._hash_to_seed(prefix)
        self.rng.manual_seed(seed % (2**64 - 1))

        colorlist_size = int(self.vocab_size / self.base)
        colorlists = []
        perm = torch.randperm(self.vocab_size, generator=self.rng)

        for i in range(self.base):
            start = i * colorlist_size
            end = (i + 1) * colorlist_size if i < self.base - 1 else self.vocab_size
            colorlists.append(set(perm[start:end].tolist()))

        return colorlists

    def get_colorlist_flag(self, token_id: int, prefix: List[int]) -> Tuple[List[bool], int]:
        colorlists = self._get_colorlists(prefix)
        colorlist_flag = []
        for colorlist in colorlists[: self.base]:
            colorlist_flag.append(token_id in colorlist)
        return colorlist_flag, self.bit_position


class MultibitWatermarkLogitsProcessor(MultibitWatermarkBase, LogitsProcessor):
    """Multi-bit watermark generation stage."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            prefix = input_ids[batch_idx].tolist()
            colorlists = self._get_colorlists(prefix)
            current_bit = self.get_current_bit(self.bit_position + 1)

            target_colorlist_idx = current_bit % self.base
            target_colorlist = colorlists[target_colorlist_idx]

            mask = torch.zeros_like(scores[batch_idx], dtype=torch.bool)
            for token_id in target_colorlist:
                if token_id < scores.shape[-1]:
                    mask[token_id] = True

            scores[batch_idx][mask] += self.delta
            self.bit_position += 1

        return scores


class MultibitWatermarkDetector(MultibitWatermarkBase):
    """Multi-bit watermark detection stage."""

    def __init__(self, *args, z_threshold: float = 1.96, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_threshold = z_threshold

    def detect(
        self,
        token_ids: List[int],
        return_message: bool = True,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ) -> Dict:
        if len(token_ids) < 2:
            return {
                "detected": False,
                "z_score": 0.0,
                "p_value": 1.0,
                "pred_message": "",
            }

        position_colorlist_counts = {}
        reconstructed_symbols = []
        z_scores = []

        for pos in range(1, len(token_ids)):
            prefix = token_ids[:pos]
            token_id = token_ids[pos]
            colorlists = self._get_colorlists(prefix)
            position_idx = (pos - 1) % self.converted_msg_length + 1

            if position_idx not in position_colorlist_counts:
                position_colorlist_counts[position_idx] = [0] * self.base

            for cl_idx, colorlist in enumerate(colorlists[: self.base]):
                if token_id in colorlist:
                    position_colorlist_counts[position_idx][cl_idx] += 1
                    break

        for pos in sorted(position_colorlist_counts.keys()):
            counts = position_colorlist_counts[pos]
            total_count = sum(counts)
            if total_count == 0:
                continue

            max_count = max(counts)
            max_idx = counts.index(max_count)

            expected_count = total_count / self.base
            variance = total_count * (1.0 / self.base) * (1.0 - 1.0 / self.base)
            z_score = (max_count - expected_count) / np.sqrt(variance) if variance > 0 else 0.0

            z_scores.append(z_score)
            reconstructed_symbols.append(max_idx)

        if z_scores:
            mean_z_score = np.mean(z_scores)
            p_value = 1 - scipy.stats.norm.cdf(mean_z_score) if mean_z_score > 0 else 1.0
        else:
            mean_z_score = 0.0
            p_value = 1.0

        pred_message = ""
        if len(reconstructed_symbols) >= self.converted_msg_length:
            pred_message = "".join(str(s) for s in reconstructed_symbols[: self.converted_msg_length])

        result = {"detected": mean_z_score > self.z_threshold}
        if return_z_score:
            result["z_score"] = float(mean_z_score)
        if return_p_value:
            result["p_value"] = float(p_value)
        if return_message:
            result["pred_message"] = pred_message

        result["position_colorlist_counts"] = position_colorlist_counts
        result["reconstructed_symbols"] = reconstructed_symbols[: self.converted_msg_length]
        return result
