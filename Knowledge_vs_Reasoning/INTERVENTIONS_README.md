# Double-intervention 2×2 — knowledge vs reasoning ground truth

`generate_interventions.py` re-tests each **correct-absent** failure (a knowledge-gap *candidate* from
the pilot) under four scaffolds and measures which one unlocks the correct answer. This converts the
pilot's proxy labels into causal ground truth.

## Design

| | −reasoning | +reasoning (step-by-step scaffold) |
|---|---|---|
| **−knowledge** | `control` (plain re-ask) | `reasoning` |
| **+knowledge** (gold-fact brief) | `knowledge` | `both` |

Each condition runs `--repeats` times with independent seeds; recovery rate = P(correct) over repeats.

**Label rule** (control `c`, +knowledge `k`, +reasoning `r`, +both `b`; margin `m`):
- `c ≥ stoch` → **stochastic-recoverable** (never a real failure)
- best condition `< hard` → **hard/unrecoverable** (deep knowledge gap *or* systematic reasoning error)
- `k−c ≥ m`, `r−c < m` → **knowledge-limited**
- `r−c ≥ m`, `k−c < m` → **reasoning-limited**
- both help → **both-sufficient**; only `b` helps → **interaction (both needed)**

Raw per-condition rates are saved, so you can re-threshold with `--label-only` without regenerating.

## Knowledge brief (the sensitive part)

The `+knowledge` brief is written by the **same model** in an oracle-informed "tutor" pass: it is shown
the correct answer but instructed to produce answer-blind domain facts. A leakage filter rejects/
regenerates briefs that name the correct option letter, quote the option verbatim, or say "the answer
is …". Briefs are cached to `interventions/knowledge_briefs.jsonl` **for manual audit** — read them.
Any brief still flagged `leaked` after retries is excluded from the label summary.

> **Caveat to state in the paper:** knowledge injection can inadvertently make the answer inferable,
> inflating the knowledge-limited count. Audit the briefs; consider a stricter (retrieval-based) brief
> as a robustness check.

## Run

```bash
cd Knowledge_vs_Reasoning

# preview subset only (no model)
python generate_interventions.py --select-only --datasets gpqa mmlu-pro

# offline plumbing test
python generate_interventions.py --backend mock --limit 5

# real run (GPU, matches baseline_v2 model)
python generate_interventions.py --backend local --model-id Qwen/Qwen2.5-14B-Instruct \
    --datasets gpqa mmlu-pro --repeats 5 --require-gpu

# or Ollama
python generate_interventions.py --backend ollama --model-id qwen2.5:7b-instruct \
    --datasets gpqa --repeats 5

# re-label from existing results with different thresholds
python generate_interventions.py --label-only --margin 0.3 --stoch 0.5 --hard 0.2
```

Runs are incremental and safe to interrupt/resume (keyed on question_no × condition × repeat).

## Subset (absent-mode=all)

`correct_absent_subset.csv`: **132 questions** correct-absent from the R1 pool in *all* seeds
(51 GPQA + 81 MMLU-Pro). Cost ≈ 132 × 4 conditions × R repeats solve calls + 132 brief calls.
Use `--absent-mode any/majority` to widen, `--limit` to pilot.

## Outputs (`interventions/`)

- `correct_absent_subset.csv` — selected questions + per-seed baseline stats
- `knowledge_briefs.jsonl` — generated briefs + leak flags (audit these)
- `solve_results.jsonl` — one row per solve call (pred, correct, raw text)
- `intervention_labels.csv` — per-question condition rates + assigned label
