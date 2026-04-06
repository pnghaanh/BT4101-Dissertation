import os

import matplotlib.pyplot as plt
import numpy as np


RESULT_400 = {
    "baseline": {
        "attacked": {
            "overall": 0.52415,
            "paraphrase": 0.5053,
            "shuffle": 0.58065,
            "deletion": 0.4865,
        },
        "clean_auc": 0.9801,
        "quality": {"distinct_2": 0.6559795152914022, "perplexity": 11.708248185379993},
        "calibration": {"mean_p_value": 0.024209862818268507, "brier_proxy": 0.022634532238935096},
        "efficiency": {"mean_latency": 107.34576786447316, "tokens_per_sec": 3.6449935373105062},
    },
    "multibit": {
        "attacked": {
            "overall": 0.4705,
            "paraphrase": 0.409,
            "shuffle": 0.5153,
            "deletion": 0.4872,
        },
        "clean_auc": 0.9725,
        "quality": {"distinct_2": 0.5918187168922896, "perplexity": 11.448574173885657},
        "calibration": {"mean_p_value": 0.022673199857270024, "brier_proxy": 0.13109084683080732},
        "efficiency": {"mean_latency": 105.0828746946156, "tokens_per_sec": 3.5954959155164046},
    },
    "pcm": {
        "attacked": {
            "overall": 0.6583666666666667,
            "paraphrase": 0.5676,
            "shuffle": 0.70915,
            "deletion": 0.69835,
        },
        "clean_auc": 0.9994,
        "quality": {"distinct_2": 0.7255821043405358, "perplexity": 18.14595996566106},
        "calibration": {"mean_p_value": 0.004436466081262339, "brier_proxy": 0.0018808627828971096},
        "efficiency": {"mean_latency": 89.48625882219523, "tokens_per_sec": 4.168936234087911},
    },
}

RESULT_600 = {
    "baseline": {
        "attacked": {
            "overall": 0.48263703703703703,
            "paraphrase": 0.4528,
            "shuffle": 0.5037333333333334,
            "deletion": 0.4913777777777778,
        },
        "clean_auc": 0.9866666666666667,
    },
    "multibit": {
        "attacked": {
            "overall": 0.5500444444444444,
            "paraphrase": 0.5203555555555556,
            "shuffle": 0.5912888888888889,
            "deletion": 0.5384888888888889,
        },
        "clean_auc": 0.9719111111111111,
    },
    "pcm": {
        "attacked": {
            "overall": 0.7137481481481481,
            "paraphrase": 0.6088888888888889,
            "shuffle": 0.7708444444444444,
            "deletion": 0.7615111111111111,
        },
        "clean_auc": 0.9861333333333333,
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

    ax.set_ylim(0.35, 0.82)
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
    ax.set_ylim(0.43, 0.76)
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

    print("Saved figures to", out_dir)


if __name__ == "__main__":
    main()
