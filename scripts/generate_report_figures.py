import os
import json

import matplotlib.pyplot as plt
import numpy as np


RESULT_400 = {
    "baseline": {
        "attacked": {
            "overall": 0.9383333333333335,
            "paraphrase": 0.92825,
            "shuffle": 0.94515,
            "deletion": 0.9416,
        },
        "clean_auc": 0.9809,
        "quality": {"distinct_2": 0.5864466661655265, "perplexity": 10.400625301253562},
        "calibration": {"mean_p_value": 0.0003211580530857516, "brier_proxy": 5.073267077578311e-06},
        "efficiency": {"mean_latency": 99.9817881879583, "tokens_per_sec": 3.800856604692779},
    },
    "multibit": {
        "attacked": {
            "overall": 0.8224333333333332,
            "paraphrase": 0.7715,
            "shuffle": 0.833,
            "deletion": 0.8628,
        },
        "clean_auc": 0.9277,
        "quality": {"distinct_2": 0.5927297985644825, "perplexity": 22.24644101883995},
        "calibration": {"mean_p_value": 0.012100744423600913, "brier_proxy": 0.08921204929458457},
        "efficiency": {"mean_latency": 97.56754421204329, "tokens_per_sec": 3.739212974498318},
    },
    "pcm": {
        "attacked": {
            "overall": 0.6481833333333333,
            "paraphrase": 0.62225,
            "shuffle": 0.64735,
            "deletion": 0.67495,
        },
        "clean_auc": 0.9787,
        "quality": {"distinct_2": 0.6970528023484878, "perplexity": 22.114841356043247},
        "calibration": {"mean_p_value": 0.036706318545605555, "brier_proxy": 0.00891608587122947},
        "efficiency": {"mean_latency": 83.50291026867926, "tokens_per_sec": 4.477363777841951},
    },
}

RESULT_600 = {
    "baseline": {
        "attacked": {
            "overall": 0.9562239583333333,
            "paraphrase": 0.968515625,
            "shuffle": 0.91390625,
            "deletion": 0.98625,
        },
        "clean_auc": 0.99,
    },
    "multibit": {
        "attacked": {
            "overall": 0.8253645833333333,
            "paraphrase": 0.825,
            "shuffle": 0.82328125,
            "deletion": 0.8278125,
        },
        "clean_auc": 0.9165625,
    },
    "pcm": {
        "attacked": {
            "overall": 0.7048958333333332,
            "paraphrase": 0.646640625,
            "shuffle": 0.751484375,
            "deletion": 0.7165625,
        },
        "clean_auc": 0.9909375,
    },
}


def _grouped_attack_plot(setting_data, methods, title, out_path):
    attacks = ["paraphrase", "shuffle", "deletion"]
    x = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    vals_a = [setting_data[methods[0]]["attacked"][a] for a in attacks]
    vals_b = [setting_data[methods[1]]["attacked"][a] for a in attacks]

    ax.bar(x - width / 2, vals_a, width, label=methods[0].capitalize(), color="#4e79a7")
    ax.bar(x + width / 2, vals_b, width, label=methods[1].upper() if methods[1] == "pcm" else methods[1].capitalize(), color="#f28e2b")

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in attacks])
    ax.set_ylabel("Attacked AUC")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_attacked_auc_vs_length(out_path):
    lengths = [400, 600]
    methods = ["baseline", "multibit", "pcm"]
    labels = {"baseline": "Baseline", "multibit": "Multibit", "pcm": "PCM"}
    colors = {"baseline": "#4e79a7", "multibit": "#59a14f", "pcm": "#e15759"}
    data_400 = RESULT_400
    data_600 = RESULT_600

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in methods:
        y = [data_400[m]["attacked"]["overall"], data_600[m]["attacked"]["overall"]]
        ax.plot(lengths, y, marker="o", linewidth=2.0, markersize=6, color=colors[m], label=labels[m])

    ax.set_xlabel("Max Tokens")
    ax.set_ylabel("Attacked Overall AUC")
    ax.set_title("Attacked AUC vs Generation Length")
    ax.set_xticks(lengths)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_quality_calib_eff_400(out_path):
    methods = ["baseline", "multibit", "pcm"]
    labels = ["Baseline", "Multibit", "PCM"]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    distinct2 = [RESULT_400[m]["quality"]["distinct_2"] for m in methods]
    ppl = [RESULT_400[m]["quality"]["perplexity"] for m in methods]
    ax = axes[0]
    ax2 = ax.twinx()
    ax.bar(x, distinct2, color="#4e79a7", alpha=0.85, width=0.55)
    ax2.plot(x, ppl, color="#e15759", marker="o", linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Quality (400 tokens)")
    ax.set_ylabel("Distinct-2", color="#4e79a7")
    ax2.set_ylabel("Perplexity", color="#e15759")
    ax.grid(axis="y", alpha=0.2)

    pvals = [RESULT_400[m]["calibration"]["mean_p_value"] for m in methods]
    brier = [RESULT_400[m]["calibration"]["brier_proxy"] for m in methods]
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(x, pvals, color="#59a14f", alpha=0.85, width=0.55)
    ax2.plot(x, brier, color="#f28e2b", marker="o", linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Calibration (400 tokens)")
    ax.set_ylabel("Mean p-value", color="#59a14f")
    ax2.set_ylabel("Brier proxy", color="#f28e2b")
    ax.grid(axis="y", alpha=0.2)

    latency = [RESULT_400[m]["efficiency"]["mean_latency"] for m in methods]
    tps = [RESULT_400[m]["efficiency"]["tokens_per_sec"] for m in methods]
    ax = axes[2]
    ax2 = ax.twinx()
    ax.bar(x, latency, color="#b07aa1", alpha=0.85, width=0.55)
    ax2.plot(x, tps, color="#76b7b2", marker="o", linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Efficiency (400 tokens)")
    ax.set_ylabel("Mean latency (s)", color="#b07aa1")
    ax2.set_ylabel("Tokens/s", color="#76b7b2")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_accuracy_and_perplexity_from_json(json_path, out_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_len = int(data.get("config", {}).get("max_tokens", 0))
    token_label = f"{token_len} tokens" if token_len > 0 else "matched setting"

    methods = ["baseline", "multibit", "pcm"]
    labels = ["Baseline", "Multibit", "PCM"]
    x = np.arange(len(methods))

    clean_acc = [data["summary"][m]["clean"]["accuracy"] for m in methods]
    attacked_acc = [data["summary"][m]["attacked"]["accuracy_mean"] for m in methods]

    ppl_labels = ["Plain", "Baseline", "Multibit", "PCM"]
    ppl_methods = ["plain", "baseline", "multibit", "pcm"]
    ppl_vals = [data["quality"][m]["mean_perplexity"] for m in ppl_methods]
    x2 = np.arange(len(ppl_labels))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    width = 0.35
    ax.bar(x - width / 2, clean_acc, width=width, color="#4e79a7", label="Clean accuracy")
    ax.bar(x + width / 2, attacked_acc, width=width, color="#f28e2b", label="Attacked mean accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy: Clean vs Attacked ({token_label})")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(x2, ppl_vals, color=["#9c755f", "#59a14f", "#e15759", "#76b7b2"], alpha=0.9)
    ax.set_xticks(x2)
    ax.set_xticklabels(ppl_labels, rotation=20)
    ax.set_ylabel("Mean perplexity")
    ax.set_title(f"Perplexity: Plain vs Watermarked ({token_label})")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_quality_calib_eff_from_json(json_path, out_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_len = int(data.get("config", {}).get("max_tokens", 0))
    token_label = f"{token_len} tokens" if token_len > 0 else "matched setting"

    methods = ["baseline", "multibit", "pcm"]
    labels = ["Baseline", "Multibit", "PCM"]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    distinct2 = [data["quality"][m]["distinct_2"] for m in methods]
    ppl = [data["quality"][m]["mean_perplexity"] for m in methods]
    ax = axes[0]
    ax2 = ax.twinx()
    ax.bar(x, distinct2, color="#4e79a7", alpha=0.85, width=0.55)
    ax2.plot(x, ppl, color="#e15759", marker="o", linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title(f"Quality ({token_label})")
    ax.set_ylabel("Distinct-2", color="#4e79a7")
    ax2.set_ylabel("Perplexity", color="#e15759")
    ax.grid(axis="y", alpha=0.2)

    pvals = [data["summary"][m]["calibration"]["mean_p_value"] for m in methods]
    brier = [data["summary"][m]["calibration"]["brier_proxy"] for m in methods]
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(x, pvals, color="#59a14f", alpha=0.85, width=0.55)
    ax2.plot(x, brier, color="#f28e2b", marker="o", linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title(f"Calibration ({token_label})")
    ax.set_ylabel("Mean p-value", color="#59a14f")
    ax2.set_ylabel("Brier proxy", color="#f28e2b")
    ax.grid(axis="y", alpha=0.2)

    latency = [data["summary"][m]["efficiency"]["mean_latency_sec"] for m in methods]
    tps = [data["summary"][m]["efficiency"]["mean_tokens_per_sec"] for m in methods]
    ax = axes[2]
    ax2 = ax.twinx()
    ax.bar(x, latency, color="#b07aa1", alpha=0.85, width=0.55)
    ax2.plot(x, tps, color="#76b7b2", marker="o", linewidth=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title(f"Efficiency ({token_label})")
    ax.set_ylabel("Mean latency (s)", color="#b07aa1")
    ax2.set_ylabel("Tokens/s", color="#76b7b2")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    out_dir = "research_final/results/figures"
    os.makedirs(out_dir, exist_ok=True)

    _grouped_attack_plot(
        RESULT_400,
        methods=["baseline", "pcm"],
        title="Attacked AUC by Attack: Baseline vs PCM (400 tokens)",
        out_path=os.path.join(out_dir, "auc_baseline_vs_pcm_400.png"),
    )
    _grouped_attack_plot(
        RESULT_600,
        methods=["baseline", "pcm"],
        title="Attacked AUC by Attack: Baseline vs PCM (600 tokens)",
        out_path=os.path.join(out_dir, "auc_baseline_vs_pcm_600.png"),
    )
    _grouped_attack_plot(
        RESULT_400,
        methods=["multibit", "pcm"],
        title="Attacked AUC by Attack: Multibit vs PCM (400 tokens)",
        out_path=os.path.join(out_dir, "auc_multibit_vs_pcm_400.png"),
    )
    _grouped_attack_plot(
        RESULT_600,
        methods=["multibit", "pcm"],
        title="Attacked AUC by Attack: Multibit vs PCM (600 tokens)",
        out_path=os.path.join(out_dir, "auc_multibit_vs_pcm_600.png"),
    )

    plot_attacked_auc_vs_length(os.path.join(out_dir, "attacked_auc_vs_length.png"))
    plot_quality_calib_eff_400(os.path.join(out_dir, "quality_calib_efficiency_400.png"))
    plot_accuracy_and_perplexity_from_json(
        "research_final/results/results_100p_400t.json",
        os.path.join(out_dir, "accuracy_and_perplexity_100p_400t.png"),
    )
    plot_accuracy_and_perplexity_from_json(
        "research_final/results/results_80p_600t.json",
        os.path.join(out_dir, "accuracy_and_perplexity_80p_600t.png"),
    )
    plot_quality_calib_eff_from_json(
        "research_final/results/results_80p_600t.json",
        os.path.join(out_dir, "quality_calib_efficiency_600.png"),
    )

    print("Saved figures to", out_dir)


if __name__ == "__main__":
    main()
