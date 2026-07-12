# Debate Diagnostic Metrics

This package computes process-level and outcome-level diagnostics for multi-agent debate traces.

## File

- `debate_diagnostic_metrics_with_judge.py`

## Input Format

Input should be an Excel workbook (`.xlsx`) with one debate per row.

Expected agent/round columns follow this pattern:

```text
R1 Agent1 Answer
R1 Agent1 Conf
R1 Agent1 Response
R2 Agent1 Answer
R2 Agent1 Conf
R2 Agent1 Response
...
```

Repeat the same pattern for each agent and each round.

Optional metadata columns:

```text
Question #
Question
Scenario
Prompt
Correct?
Rounds to Consensus
```

`Supervisor` and `Moderator` columns are ignored for process metrics.

## Subjective / Likert Debates

Use this mode when agents give ordinal stances, such as:

```text
1, 2, 3, 4, 5
```

or:

```text
-2, -1, 0, 1, 2
```

Run:

```bash
python debate_diagnostic_metrics_with_judge.py \
  --input YOUR_FILE.xlsx \
  --stance-mode likert \
  --q-source llm \
  --metric-version paper
```

## Objective / Multiple-Choice Debates

Use this mode when agents give categorical answers, such as:

```text
A, B, C, D
```

Run:

```bash
python debate_diagnostic_metrics_with_judge.py \
  --input YOUR_FILE.xlsx \
  --stance-mode categorical \
  --q-source llm \
  --metric-version paper
```

## Reasoning Quality Scores

By default, use:

```bash
--q-source llm
```

This uses the built-in LLM-as-judge / RecEval-style scorer to produce reasoning-quality scores `q_ta` in `[0, 1]`.

This requires:

```bash
export OPENAI_API_KEY=...
```

If the workbook already has confidence values and you want to use those as `q_ta`, run with:

```bash
--q-source confidence
```

This result will be significantly worse as receval values are expected to understand if reasoning change was supported. 

## Outputs

The script writes outputs to `diagnostic_metric_results/` by default, or to a custom folder with:

```bash
--out-dir YOUR_OUTPUT_DIR
```

When using `--q-source llm`, it writes:

```text
<input>.llm_judgments.csv
```

For all runs, it writes:

```text
<input>.<stance_mode>.<q_source>.<metric_version>.scores.csv
```

## Metrics

The score CSV includes:

- `engagement`
- `responsiveness`
- `influence_asymmetry`
- `balance`
- `stability`
- `group_welfare`

For objective multiple-choice debates, the metrics are categorical analogues of the Likert equations because answer labels do not have meaningful numeric distances.

## Recommended Setting

Use:

```bash
--metric-version paper
```

This uses the camera-ready equations, including the balance equation:

```text
Balance = (1 - C_max / (C_total + eps)) * (1 - V / (T - 2))
```

## GPU-Backed Qwen Runs

Fixture mode does not use a GPU because it does not load Qwen. For a real local
Qwen generation run with the `transformers` backend, require CUDA explicitly:

```bash
python3.9 qwen_methodology_code.py \
  --backend transformers \
  --model-id Qwen/Qwen2.5-14B-Instruct \
  --require-gpu \
  --device-map auto \
  --torch-dtype auto
```

Before running, verify that the scheduler or shell can see a GPU. The PyTorch
check is the decisive one for this script:

```bash
nvidia-smi
python3.9 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If CUDA is not visible, `--require-gpu` stops the run before model loading so it
does not accidentally fall back to CPU. On some managed clusters, `nvidia-smi`
may fail even when PyTorch can use CUDA, so trust the Python check and the
script's startup line (`CUDA GPU available for Qwen generation: ...`).

## MMLU-Pro Debate Generation

To generate debate traces from MMLU-Pro instead of the built-in toy objective
questions, install the Hugging Face datasets loader and select the dataset
source:

```bash
python3.9 -m pip install pyarrow==15.0.2 datasets==2.19.2

python3.9 qwen_methodology_code.py \
  --dataset-source mmlu-pro \
  --mmlu-pro-dataset TIGER-Lab/MMLU-Pro \
  --mmlu-pro-split test \
  --objective-limit 50 \
  --subjective-limit 0 \
  --backend transformers \
  --model-id Qwen/Qwen2.5-14B-Instruct \
  --require-gpu \
  --device-map auto \
  --torch-dtype auto \
  --out-dir qwen_mmlu_pro_results
```

MMLU-Pro categories can be filtered by repeating `--mmlu-pro-category`, for
example:

```bash
--mmlu-pro-category math --mmlu-pro-category physics
```

The generated workbook will include MMLU-Pro rows in the `Debate_Traces` sheet
and the normal diagnostic outputs in `Reasoning_Quality`, `Diagnostic_Scores`,
`Aggregate_Summary`, and `Objective_Correlations`.

## 200-Question Qwen MMLU-Pro Batch Run

Use the dedicated wrapper to generate a new 200-question workbook without
overwriting the existing 50-question result:

```bash
python3.9 generate_qwen_mmlu_pro_200.py
```

The output path is:

```text
data/qwen_mmlu_pro_200/qwen_mmlu_pro_debate_traces.xlsx
```

The wrapper requires CUDA to be visible to PyTorch before it loads
`Qwen/Qwen2.5-14B-Instruct`.

To submit it as a Slurm batch job:

```bash
sbatch run_qwen_mmlu_pro_200.sbatch
```

If your cluster uses a different GPU partition or memory policy, edit the
`#SBATCH` lines in `run_qwen_mmlu_pro_200.sbatch` before submitting. If a run is
interrupted during trace generation, rerun with:

```bash
python3.9 generate_qwen_mmlu_pro_200.py --resume
```
