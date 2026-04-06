import argparse
import json
import random
import time
from typing import Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research_final.pipeline.generation import generate_with_watermark
from research_final.pipeline.data_helper import build_wikitext_prompts
from research_final.schemes.factory import WATERMARK_METHODS, build_processors_and_detectors
from research_final.comparison_components import (
    get_attack_registry,
    compute_auc,
    compute_accuracy,
    bootstrap_ci,
    summarize_quality,
    compute_calibration_stats,
    compute_efficiency_stats,
    save_summary_json,
)


def parse_args():
    p = argparse.ArgumentParser("Watermark comparison")
    p.add_argument("--model_name", default="gpt2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_prompts", type=int, default=500)
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", default="research_final/results/results.json")
    return p.parse_args()


def continuation_only(tokenizer, prompt: str, full_ids, full_text):
    prompt_ids = tokenizer.encode(prompt)
    continuation_ids = full_ids[len(prompt_ids):]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
    if not continuation_text:
        continuation_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
    return continuation_text, continuation_ids


def detector_score(detection: Dict) -> float:
    return float(detection.get("z_score", 0.0))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = build_wikitext_prompts(n_prompts=args.n_prompts)

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    vocab_size = tokenizer.vocab_size

    attacks = get_attack_registry()
    methods = list(WATERMARK_METHODS)
    quality_methods = ["plain"] + methods

    processors, detectors = build_processors_and_detectors(vocab_size=vocab_size)

    results = {m: {"clean": [], "attacked": {a: [] for a in attacks}} for m in methods}
    controls = {m: {"clean": [], "attacked": {a: [] for a in attacks}} for m in methods}
    latencies = {m: [] for m in methods}
    n_tokens = {m: [] for m in methods}
    quality_texts = {m: [] for m in quality_methods}
    quality_ids = {m: [] for m in quality_methods}

    for prompt in prompts:
        plain_text, plain_ids = generate_with_watermark(model, tokenizer, prompt, max_tokens=args.max_tokens, processor=None)
        pt, pi = continuation_only(tokenizer, prompt, plain_ids, plain_text)
        quality_texts["plain"].append(pt)
        quality_ids["plain"].append(pi)

        wm_texts = {}
        wm_ids = {}
        for method in methods:
            proc = processors[method]
            if method == "multibit":
                proc.bit_position = 0
            if method == "pcm":
                proc.anchor_color = None

            t0 = time.perf_counter()
            generation_kwargs = getattr(proc, "generation_kwargs", {})
            txt, ids = generate_with_watermark(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                processor=proc,
                **generation_kwargs,
            )
            dt = time.perf_counter() - t0
            wm_texts[method] = txt
            wm_ids[method] = ids

            results[method]["clean"].append(detectors[method].detect(ids))
            controls[method]["clean"].append(detectors[method].detect(plain_ids))

            ct, ci = continuation_only(tokenizer, prompt, ids, txt)
            quality_texts[method].append(ct)
            quality_ids[method].append(ci)

            latencies[method].append(float(dt))
            n_tokens[method].append(len(ci))

        for attack_name, attack_cfg in attacks.items():
            attacked_plain = attack_cfg["func"](plain_text, attack_cfg["strength"])
            attacked_plain_ids = tokenizer.encode(attacked_plain)

            for method in methods:
                attacked_wm = attack_cfg["func"](wm_texts[method], attack_cfg["strength"])
                attacked_wm_ids = tokenizer.encode(attacked_wm)
                results[method]["attacked"][attack_name].append(detectors[method].detect(attacked_wm_ids))
                controls[method]["attacked"][attack_name].append(detectors[method].detect(attacked_plain_ids))

    summary = {}
    for method in methods:
        clean_pos = [detector_score(r) for r in results[method]["clean"]]
        clean_neg = [detector_score(r) for r in controls[method]["clean"]]
        clean_auc = compute_auc(clean_pos, clean_neg)
        clean_acc = compute_accuracy(results[method]["clean"], controls[method]["clean"])

        attacked_auc_vals = []
        attacked_acc_vals = []
        per_attack_auc = {}
        per_attack_acc = {}
        for attack_name in attacks:
            pos = [detector_score(r) for r in results[method]["attacked"][attack_name]]
            neg = [detector_score(r) for r in controls[method]["attacked"][attack_name]]
            auc_val = compute_auc(pos, neg)
            acc_val = compute_accuracy(results[method]["attacked"][attack_name], controls[method]["attacked"][attack_name])
            attacked_auc_vals.append(auc_val)
            attacked_acc_vals.append(acc_val)
            per_attack_auc[attack_name] = float(auc_val)
            per_attack_acc[attack_name] = float(acc_val)

        auc_mean, auc_lo, auc_hi = bootstrap_ci(attacked_auc_vals, n_boot=200)
        acc_mean, acc_lo, acc_hi = bootstrap_ci(attacked_acc_vals, n_boot=200)

        summary[method] = {
            "clean": {"auc": float(clean_auc), "accuracy": float(clean_acc)},
            "attacked": {
                "auc_mean": float(auc_mean), "auc_ci_low": float(auc_lo), "auc_ci_high": float(auc_hi),
                "accuracy_mean": float(acc_mean), "accuracy_ci_low": float(acc_lo), "accuracy_ci_high": float(acc_hi),
                "auc_by_attack": per_attack_auc,
                "accuracy_by_attack": per_attack_acc,
            },
            "calibration": compute_calibration_stats(results[method]["clean"]),
            "efficiency": compute_efficiency_stats(latencies[method], n_tokens[method]),
        }

    quality = summarize_quality(model, args.device, quality_texts, quality_ids)

    payload = {
        "config": vars(args),
        "summary": summary,
        "quality": quality,
        "attacks": {k: v["strength"] for k, v in attacks.items()},
    }
    save_summary_json(args.output_json, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
