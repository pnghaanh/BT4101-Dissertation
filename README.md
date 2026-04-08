# Phung Nguyen Hanh BT4101 Dissertation Code Release

Benchmark suite for comparing three language-model watermarking methods:
- Baseline watermark
- Multi-bit watermark
- PCM watermark

The pipeline evaluates clean detection, robustness under attacks, quality, calibration, and efficiency.

## Folder structure

- `scripts/run_comparison.py`: main benchmark runner
- `pipeline/generation.py`: generation helper with optional watermark processor
- `pipeline/data_helper.py`: WikiText prompt construction
- `schemes/baseline_watermark.py`: baseline generator/detector
- `schemes/multibit_watermark.py`: multi-bit generator/detector
- `schemes/pcm_watermark.py`: PCM generator/detector
- `comparison_components/`: attacks, metrics, quality, robustness, reporting
- `schemes/factory.py`: method registry and builder
- `scripts/plot_results.py`: summary plotting from results JSON
- `scripts/generate_report_figures.py`: report-specific figure generation
- `results/`: output JSON, CSV, and figures

## Requirements

Install dependencies from this folder:

```bash
pip install -r research_final/requirements.txt
```

If you are using a conda environment, activate it first.

## Quick start

Run from the project root (the directory that contains `research_final`):

```bash
python -m research_final.scripts.run_comparison \
  --model_name gpt2 \
  --device cpu \
  --n_prompts 100 \
  --max_tokens 400 \
  --seed 42 \
  --output_json research_final/results/results.json
```

This writes a JSON payload with:
- config
- summary (clean/attacked detection, calibration, efficiency)
- quality metrics
- attack strengths

## Plotting

General summary plots:

```bash
python research_final/scripts/plot_results.py \
  --results_json research_final/results/results.json \
  --out_dir research_final/results
```

Report figures (pre-filled comparison plots):

```bash
python research_final/scripts/generate_report_figures.py
```

Figures are saved to:
- `research_final/results/`
- `research_final/results/figures/`

## Notes

- Default model is GPT-2.
- The benchmark loads prompts from WikiText via the `datasets` package.
- For reproducibility, the script sets Python, NumPy, and PyTorch seeds from `--seed`.
