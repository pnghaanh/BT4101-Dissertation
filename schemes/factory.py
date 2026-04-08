from typing import Dict, Tuple

from research_final.schemes.baseline_watermark import BaselineWatermarkDetector, BaselineWatermarkLogitsProcessor
from research_final.schemes.multibit_watermark import MultibitWatermarkDetector, MultibitWatermarkLogitsProcessor
from research_final.schemes.pcm_watermark import PCMWatermarkDetector, PCMWatermarkLogitsProcessor

WATERMARK_METHODS = ["baseline", "multibit", "pcm"]


def build_processors_and_detectors(vocab_size: int) -> Tuple[Dict, Dict]:
    processors = {
        "baseline": BaselineWatermarkLogitsProcessor(vocab_size=vocab_size, gamma=0.5, delta=2.0),
        "multibit": MultibitWatermarkLogitsProcessor(
            vocab_size=vocab_size,
            gamma=0.5,
            delta=2.0,
            base=2,
        ),
        "pcm": PCMWatermarkLogitsProcessor(vocab_size=vocab_size, gamma=0.5, delta=2.0, block_size=8, warmup=40),
    }

    detectors = {
        "baseline": BaselineWatermarkDetector(vocab_size=vocab_size, gamma=0.5),
        "multibit": MultibitWatermarkDetector(
            vocab_size=vocab_size,
            gamma=0.5,
            delta=2.0,
            base=2,
            z_threshold=1.96,
        ),
        "pcm": PCMWatermarkDetector(vocab_size=vocab_size, gamma=0.5, block_size=8, warmup=40, z_threshold=2.24),
    }

    return processors, detectors
