import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(summary: dict, quality: dict):
    rows = []
    for method, vals in summary.items():
        row = {
            "method": method,
            "clean_auc": vals["clean"]["auc"],
            "clean_accuracy": vals["clean"]["accuracy"],
            "attacked_auc_mean": vals["attacked"]["auc_mean"],
            "attacked_auc_ci_low": vals["attacked"]["auc_ci_low"],
            "attacked_auc_ci_high": vals["attacked"]["auc_ci_high"],
            "attacked_accuracy_mean": vals["attacked"]["accuracy_mean"],
            "attacked_accuracy_ci_low": vals["attacked"]["accuracy_ci_low"],
            "attacked_accuracy_ci_high": vals["attacked"]["accuracy_ci_high"],
            "mean_p_value": vals["calibration"]["mean_p_value"],
            "brier_proxy": vals["calibration"]["brier_proxy"],
            "mean_latency_sec": vals["efficiency"]["mean_latency_sec"],
            "p95_latency_sec": vals["efficiency"]["p95_latency_sec"],
            "mean_tokens_per_sec": vals["efficiency"]["mean_tokens_per_sec"],
        }
        if method in quality:
            row["distinct_1"] = quality[method]["distinct_1"]
            row["distinct_2"] = quality[method]["distinct_2"]
            row["mean_perplexity"] = quality[method]["mean_perplexity"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    return df


def _save(fig, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(df: pd.DataFrame, summary: dict, output_dir: str):
    if plt is None:
        return []

    methods = df["method"].tolist()
    x = np.arange(len(methods))
    out_files = []

    fig1, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    width = 0.35
    ax1.bar(x - width / 2, df["clean_auc"], width, label="Clean AUC", alpha=0.85)
    ax1.bar(x + width / 2, df["attacked_auc_mean"], width, label="Attacked AUC", alpha=0.85)
    ax1.set_title("AUC (Clean vs Attacked)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20)
    ax1.set_ylim(0.0, 1.05)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    out_auc = str(Path(output_dir) / "auc_bar.png")
    _save(fig1, out_auc)
    out_files.append(out_auc)

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))
    ax2.bar(x - width / 2, df["clean_accuracy"], width, label="Clean Acc", alpha=0.85)
    ax2.bar(x + width / 2, df["attacked_accuracy_mean"], width, label="Attacked Acc", alpha=0.85)
    ax2.set_title("Accuracy (Clean vs Attacked)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=20)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    out_acc = str(Path(output_dir) / "accuracy_bar.png")
    _save(fig2, out_acc)
    out_files.append(out_acc)

    fig3, ax3 = plt.subplots(1, 1, figsize=(9, 5))
    ax3.bar(x, df["mean_perplexity"], alpha=0.85, color="darkred")
    ax3.set_title("Text Quality (Mean Perplexity)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=20)
    ax3.grid(axis="y", alpha=0.3)
    out_ppl = str(Path(output_dir) / "perplexity_bar.png")
    _save(fig3, out_ppl)
    out_files.append(out_ppl)

    return out_files


def main():
    parser = argparse.ArgumentParser(description="Generate summary tables/plots from comparison results")
    parser.add_argument("--input_json", required=True, help="Path to results.json")
    parser.add_argument("--output_csv", default="research_final/results/summary_table.csv")
    parser.add_argument("--output_dir", default="research_final/results")
    args = parser.parse_args()

    payload = load_results(args.input_json)
    df = build_dataframe(payload["summary"], payload.get("quality", {}))

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    plot_paths = plot_summary(df, payload["summary"], args.output_dir)

    print(f"Saved table: {args.output_csv}")
    for path in plot_paths:
        print(f"Saved plot : {path}")
if __name__ == "__main__":
    main()
