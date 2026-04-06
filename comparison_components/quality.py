import math
from collections import Counter
from typing import List

import numpy as np
import torch


def compute_distinct_n(texts: List[str], n: int) -> float:
    ngrams = []
    for text in texts:
        words = text.split()
        if len(words) < n:
            continue
        ngrams.extend(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    if len(ngrams) == 0:
        return 0.0
    return float(len(set(ngrams)) / len(ngrams))


def compute_perplexity(model, device: str, token_ids: List[int]) -> float:
    if len(token_ids) < 2:
        return float("nan")
    input_tensor = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, labels=input_tensor)
    return float(math.exp(float(outputs.loss.item())))


def summarize_quality(model, device: str, text_map, token_id_map):
    def corpus_bleu(candidate_texts: List[str], reference_texts: List[str], max_n: int = 4) -> float:
        if not candidate_texts or not reference_texts:
            return float("nan")

        cand_lens = 0
        ref_lens = 0
        log_precisions = []

        for n in range(1, max_n + 1):
            match = 0
            total = 0
            for cand, ref in zip(candidate_texts, reference_texts):
                cand_tokens = cand.split()
                ref_tokens = ref.split()
                cand_lens += len(cand_tokens) if n == 1 else 0
                ref_lens += len(ref_tokens) if n == 1 else 0

                if len(cand_tokens) < n:
                    continue
                cand_ngrams = Counter(tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1))
                ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(max(0, len(ref_tokens) - n + 1)))

                total += sum(cand_ngrams.values())
                for ng, cnt in cand_ngrams.items():
                    match += min(cnt, ref_ngrams.get(ng, 0))

            # Add-one smoothing to avoid zeroing the whole BLEU.
            precision_n = (match + 1.0) / (total + 1.0)
            log_precisions.append(math.log(precision_n))

        if cand_lens == 0:
            return float("nan")
        if cand_lens > ref_lens:
            bp = 1.0
        else:
            bp = math.exp(1.0 - (ref_lens / max(1, cand_lens)))

        bleu = bp * math.exp(sum(log_precisions) / max_n)
        return float(bleu)

    plain_refs = text_map.get("plain", [])
    out = {}
    for method, texts in text_map.items():
        token_lists = token_id_map.get(method, [])
        perplexities = [compute_perplexity(model, device, ids) for ids in token_lists if len(ids) >= 2]
        bleu = corpus_bleu(texts, plain_refs) if method != "plain" else 1.0
        out[method] = {
            "distinct_1": compute_distinct_n(texts, 1),
            "distinct_2": compute_distinct_n(texts, 2),
            "mean_perplexity": float(np.nanmean(perplexities)) if perplexities else float("nan"),
            "bleu_vs_plain": bleu,
        }
    return out
