# Llama Phase 2 Run Status

## ✅ RUN STARTED: `python run_llama_replication.py --phase 2`

**Start time:** 2026-08-18 10:50 UTC-5  
**Status:** ACTIVE (background process running)  
**Mode:** Async detached (will continue even if session closes)

---

## 📊 Current Progress

| Metric | Value |
|--------|-------|
| Solve records | 5,665 |
| Total expected | 16,800 |
| Progress | 33.7% |
| Questions processed | ~118 of 300 |
| Estimated hours elapsed | ~1-2 hours |
| Estimated hours remaining | ~58-60 hours |

---

## 🎯 What This Run Is Doing

### Design: Phase 2 Brief-Instance Variance Decomposition
- **Questions:** 300 (from MMLU, MMLU-Pro, GPQA)
- **Conditions:** 7 (control, knowledge_blind, knowledge_oracle, reasoning, both_blind, both_oracle)
- **Per condition:** 8 measurements

### Measurement Structure (NEW 4×2 Design)
**For stochastic scaffolds** (knowledge_blind, knowledge_oracle, both_blind, both_oracle):
- 4 brief instances (independently generated with seeds 0, 1000, 2000, 3000)
- 2 solver draws per brief instance (seeds +0, +100)
- Total: 4 × 2 = 8 measurements per condition

**For fixed templates** (control, reasoning):
- 8 solver draws (no brief generation layer)
- Total: 8 measurements per condition

### Seed Separation Strategy
```
Seed = base (7) + 1000*(brief_instance-1) + 100*(solver_draw-1) + 13*condition_offset
```
- **1000 Hz:** Brief instances separated massively → different RNG streams
- **100 Hz:** Solver draws moderately separated → distinct within brief
- **13 Hz:** Conditions orthogonal → no collision risk
- **Disjoint from Phase 1:** Phase 1 used seeds 7, 507, 1007, ... (non-overlapping)

### Key Flag: `--regenerate-briefs`
- Forces fresh brief generation for Llama (ignores any Qwen brief cache)
- Enables unconfounded comparison: "Does categorical structure generalize to Llama?"
- Required for valid variance decomposition (measures brief-instance variance independently)

---

## 🔍 What This Enables

### 1. Cross-Model Validation
- Qwen Phase 2: Generated briefs independently (Qwen-tutor)
- Llama Phase 2: Generates briefs independently (Llama-tutor)
- Comparison: Do failure categories (persistent vs responsive) generalize?

### 2. Brief-Instance Variance Decomposition (Full Dataset)
- Per-question effects across 4 brief instances
- Variance of effects = K-variance (brief-generation variance)
- Within-brief consistency = solver variance
- Expected: 60-91% K-variance, <2% solver, <1% control (full dataset confirmation)

### 3. Failure Category Stability Under Regeneration
- Persistent failures: Expected to remain persistent (category stability)
- Responsive failures: Expected to remain responsive but effect magnitude varies
- Tests asymmetry hypothesis: "categories stable, magnitudes not"

---

## 📁 Output Locations

**Briefs:** `interventions_llama/knowledge_briefs.jsonl`  
- Schema: { question_no, condition, brief_instance, blind_brief, oracle_brief, oracle_leaked, blind_leaked }
- Expected: 300 questions × 5 stochastic conditions × 4 brief instances = 6,000 records

**Solve results:** `interventions_llama/solve_results.jsonl`  
- Schema: { question_no, dataset, condition, brief_instance, solver_draw, seed, pred, correct, oracle_leaked, raw }
- Expected: 300 questions × 7 conditions × 8 measurements = 16,800 records
- Current: 5,665 records (33.7%)

---

## ⏱️ Timeline

### What's happening NOW (parallel)
1. **Llama Phase 2 run:** Background process (60h expected)
   - Generating 4 briefs per question per condition (fresh generation)
   - Running 2 solver passes per brief
   - Total: ~16,800 solve records expected
   - Status: 33.7% complete, ~58-60 hours remaining

2. **Brief-regeneration persistent robustness check:** Background process (~2-3h remaining)
   - Validating persistent category on 98 persistent questions
   - Current: 21/98 complete, 95.2% persistence rate
   - Will provide final stability percentages for abstract

### What will happen AFTER Phase 2 completes (estimated +58-60 hours)
1. **Variance decomposition analysis:** 2-4 hours
   - Compute K-variance vs solver variance on full dataset
   - Compare Qwen vs Llama variance structures
   - Validate asymmetry on 300-question sample

2. **Results section draft:** 2-3 hours
   - Organize around asymmetry as main narrative
   - Include variance decomposition findings
   - Cross-model comparison

3. **Discussion/Conclusion:** 2-3 hours

---

## 🛡️ Process Reliability

**How background process is running:**
- Command: `python run_llama_replication.py --phase 2`
- Mode: `async detach=true` (PowerShell background process)
- Will survive: Session closure, temporary network interruption to Ollama
- May not survive: System reboot, manual kill command

**Monitoring:**
- Check file timestamps: `Get-Item interventions_llama/solve_results.jsonl | Select LastWriteTime`
- Check record count: `(Get-Content interventions_llama/solve_results.jsonl | Measure-Object -Line).Lines`
- Check process: `ps | Where-Object { $_.ProcessName -eq "python" }`

**Ollama status:**
- Currently running: ✓ Verified at start time
- Model cache: Llama 3.1:8B should be pre-pulled
- If Ollama crashes: Restart with `ollama serve`, then restart Python script

---

## 🎓 What the Results Will Show

### Brief-Instance Variance (K-variance)
**Prediction (from pilot of 39 responsive questions):**
- Knowledge_blind: 60-91% K-variance
- Knowledge_oracle: 60-91% K-variance  
- Both_blind: 60-91% K-variance
- Both_oracle: 60-91% K-variance

**Test (full 300-question sample):**
- Will K-variance percentages hold on larger sample?
- Are outliers? Any conditions with lower variance?

### Solver Variance
**Prediction (from pilot):**
- <2% in all scaffold conditions
- <1% in control

**Test (full dataset):**
- Is solver stochasticity truly negligible?
- Does it hold across all model types (Qwen vs Llama)?

### Cross-Model Comparison
**Key questions answered:**
- Do Llama and Qwen have same failure categories? (persistent vs responsive)
- Do brief-generated from Llama vs Qwen differ in effectiveness?
- Is categorical structure model-independent or Qwen-specific?

---

## 📝 Next Session Actions

When Llama Phase 2 completes:
1. Check file: `interventions_llama/solve_results.jsonl` (~16,800 records expected)
2. Run analysis: Variance decomposition on full dataset
3. Compare with Qwen Phase 2 results (from interventions/)
4. Draft Results section with asymmetry as organizing principle

---

## 🔗 Related Documentation

- **PAPER_DRAFT.md** — Main paper (Abstract/Methods now updated)
- **ABSTRACT_AND_METHODS_UPDATE_SUMMARY.md** — Details of recent updates
- **BRIEF_REGENERATION_JOB_DESIGN_EXPLAINED.md** — Explains persistent robustness check
- **PHASE2_DESIGN_4BRIEFS_2DRAWS.md** — Technical specification of 4×2 design
- **run_llama_replication.py** — Control script for this run (lines 50-85)
- **generate_interventions.py** — Core implementation (lines 116-120, 618-666)

---

## Summary

✅ **Llama Phase 2 is running successfully in background**  
✅ **Will generate 16,800 solve records with 4 briefs × 2 draws design**  
✅ **Enables unconfounded variance decomposition and cross-model validation**  
✅ **Expected to complete in ~58-60 hours**  
✅ **Results will confirm or refine asymmetry finding on full Llama dataset**

Session can close safely; process will continue running.
