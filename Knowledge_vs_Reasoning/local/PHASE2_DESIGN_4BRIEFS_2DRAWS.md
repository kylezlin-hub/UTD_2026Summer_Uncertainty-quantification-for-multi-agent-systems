# Phase 2 Design: 4 Brief Instances × 2 Solver Draws

## Overview

Phase 2 now measures both **brief-generation stochasticity** and **solver stochasticity** separately, enabling rigorous variance decomposition.

**Total measurements per question:** 6 conditions × 8 draws = 48 measurements

**Measurement structure:**
- Stochastic scaffolds (S, C): 4 brief instances × 2 solver draws = 8 measurements per condition
- Fixed templates (R, control): 8 solver draws (no brief instances)

---

## Why This Design?

### The Problem with Fixed Briefs
Previously, briefs were generated **once per question** and reused across all 8 solver repeats:
```
Question 1:
  Generate brief_fixed (once)
  Solve with brief_fixed, seed 1
  Solve with brief_fixed, seed 2
  ...
  Solve with brief_fixed, seed 8
  
Problem: All 8 solves use IDENTICAL brief text
Result: Measures solver variance ONLY, not brief variance
```

### The New Design
Now, briefs are generated **fresh for each solver pair**:
```
Question 1, Condition S (knowledge_blind):
  Generate brief_instance_1 (seed=7)
    Solve with brief_1, seed_solve_1 (seed=20)
    Solve with brief_1, seed_solve_2 (seed=120)
  
  Generate brief_instance_2 (seed=107)
    Solve with brief_2, seed_solve_1 (seed=1020)
    Solve with brief_2, seed_solve_2 (seed=1120)
  
  Generate brief_instance_3 (seed=207)
    Solve with brief_3, seed_solve_1 (seed=2020)
    Solve with brief_3, seed_solve_2 (seed=2120)
  
  Generate brief_instance_4 (seed=307)
    Solve with brief_4, seed_solve_1 (seed=3020)
    Solve with brief_4, seed_solve_2 (seed=3120)

Result: 4 different briefs × 2 solver draws = 8 measurements total
```

### Key Advantage: Variance Decomposition
With two solver draws per brief instance, we can cleanly separate:

**K-variance (brief-generation variance):**
- Effects that vary across the 4 brief instances
- Estimate: variance of the 4 (brief_1_effect, brief_2_effect, brief_3_effect, brief_4_effect)

**Solver variance:**
- Within-brief consistency (how similar are the 2 draws on the same brief?)
- Estimate: mean of within-brief variances across the 4 instances

---

## Implementation Details

### Constants
```python
STOCHASTIC_SCAFFOLD_CONDS = ("knowledge_blind", "knowledge_oracle", "both_blind", "both_oracle")
FIXED_TEMPLATE_CONDS = ("control", "reasoning")
```

### Brief Generation Function
```python
def generate_brief_instances(llm, q, options, base_seed, num_instances=4):
    """Generate 4 independent brief instances.
    
    Each instance uses a different seed:
    - Instance 1: seed = base_seed + 0
    - Instance 2: seed = base_seed + 100
    - Instance 3: seed = base_seed + 200
    - Instance 4: seed = base_seed + 300
    
    For oracle briefs, add +500 to each seed.
    
    Returns: {1: {"blind": (text, leaked), "oracle": (text, leaked)}, ...}
    """
```

### Solve Loop Structure

**For stochastic scaffolds (S, C):**
```python
for brief_inst_num in range(1, 5):  # 4 brief instances
    brief_inst = brief_instances[brief_inst_num]
    
    for solver_draw in range(1, 3):  # 2 solver draws per instance
        # Seed = base + 1000*(instance-1) + 100*(draw-1) + 13*condition_offset
        seed = seed_calc(base, brief_inst_num, solver_draw, cond)
        solve_once(q, cond, brief_inst[brief_key], seed)
        
        write JSONL:
          brief_instance: 1-4
          solver_draw: 1-2
```

**For fixed templates (R, control):**
```python
for solver_draw in range(1, 9):  # Always exactly 8 draws
    # Seed = base + 1000*(draw-1) + 13*condition_offset
    seed = seed_calc(base, 0, solver_draw, cond)
    solve_once(q, cond, briefs={}, seed)
    
    write JSONL:
      brief_instance: 0  (no instance layer)
      solver_draw: 1-8
```

### JSONL Structure
```json
{"question_no": "q1", "condition": "knowledge_blind", "brief_instance": 1, "solver_draw": 1, "seed": 20, "correct": true, "oracle_leaked": false, ...}
{"question_no": "q1", "condition": "knowledge_blind", "brief_instance": 1, "solver_draw": 2, "seed": 120, "correct": false, "oracle_leaked": false, ...}
{"question_no": "q1", "condition": "knowledge_blind", "brief_instance": 2, "solver_draw": 1, "seed": 1020, "correct": true, "oracle_leaked": false, ...}
{"question_no": "q1", "condition": "knowledge_blind", "brief_instance": 2, "solver_draw": 2, "seed": 1120, "correct": true, "oracle_leaked": false, ...}
...
{"question_no": "q1", "condition": "control", "brief_instance": 0, "solver_draw": 1, "seed": 7, "correct": false, "oracle_leaked": false, ...}
{"question_no": "q1", "condition": "control", "brief_instance": 0, "solver_draw": 2, "seed": 1007, "correct": false, "oracle_leaked": false, ...}
...
{"question_no": "q1", "condition": "control", "brief_instance": 0, "solver_draw": 8, "seed": 7007, "correct": false, "oracle_leaked": false, ...}
```

### Seed Calculation

```
For stochastic scaffolds (brief_inst_num = 1..4, solver_draw = 1..2):
  seed = base_seed + 1000*(brief_inst_num - 1) + 100*(solver_draw - 1) + 13*condition_offset
  
  Examples for condition knowledge_blind (offset=1) with base_seed=7:
    brief_inst 1, draw 1: 7 + 0 + 0 + 13 = 20
    brief_inst 1, draw 2: 7 + 0 + 100 + 13 = 120
    brief_inst 2, draw 1: 7 + 1000 + 0 + 13 = 1020
    brief_inst 2, draw 2: 7 + 1000 + 100 + 13 = 1120
    ...
    brief_inst 4, draw 2: 7 + 3000 + 100 + 13 = 3120

For fixed templates (brief_instance = 0, solver_draw = 1..8):
  seed = base_seed + 1000*(solver_draw - 1) + 13*condition_offset
  
  Examples for condition control (offset=0) with base_seed=7:
    draw 1: 7 + 0 + 0 = 7
    draw 2: 7 + 1000 + 0 = 1007
    draw 3: 7 + 2000 + 0 = 2007
    ...
    draw 8: 7 + 7000 + 0 = 7007
```

---

## Variance Decomposition Analysis

### Why Within-Brief Pairs?
The two solver draws on the **same brief instance** enable estimation of:
```
Within-brief variance = mean of (correct_draw1 - correct_draw2)² across pairs

This captures solver stochasticity holding brief text fixed.
```

The **across-brief differences** estimate:
```
Effect_brief_1 = mean(correct|brief_1) - mean(correct|control)
Effect_brief_2 = mean(correct|brief_2) - mean(correct|control)
Effect_brief_3 = mean(correct|brief_3) - mean(correct|control)
Effect_brief_4 = mean(correct|brief_4) - mean(correct|control)

K-variance = var(Effect_brief_1, ..., Effect_brief_4)

This captures brief-generation stochasticity.
```

### Example Variance Decomposition
On 39 previously-responsive questions (from regeneration experiment):

| Component | Blind | Oracle | Control |
|-----------|-------|--------|---------|
| K-variance (brief instances) | 60.4% | 91.1% | ~0% |
| Solver variance | 39.6% | 8.9% | ~100% |

**Interpretation:**
- **Blind briefs:** 60% of variance is from brief regeneration, 40% from solver noise
- **Oracle briefs:** 91% is brief variance, 9% solver noise (oracle briefs are more deterministic)
- **Control:** 100% solver variance (no scaffold to generate)

---

## Backward Compatibility & Resume

### `load_done()` Function
Tracks completed measurements as:
```python
done.add((question_no, condition, brief_instance, solver_draw))
```

For legacy data with only `"repeat"` field:
```python
brief_inst = d.get("brief_instance", 0)  # Defaults to 0
solver_draw = d.get("solver_draw", d.get("repeat", 0))  # Falls back to repeat
```

This allows resuming even if mixing old and new JSONL formats.

### Resume Behavior
- If result already exists for (q, cond, brief_inst, solver_draw), skip it
- If JSONL is interrupted, restart safely from where it left off
- Old results with only `"repeat"` field won't conflict (mapped to brief_inst=0, solver_draw=repeat)

---

## Comparison to Qwen Phase 2 (Old Design)

| Aspect | Old Design | New Design |
|--------|-----------|-----------|
| **Brief generation** | Once per question | 4 times (fresh per pair) |
| **Measurements per cond** | 8 solver draws on 1 brief | 4 briefs × 2 draws on each |
| **Total per question** | 6 × 8 = 48 | 6 × 8 = 48 (same!) |
| **Can measure K-variance?** | ✗ (no brief variance) | ✓ (60–91% of variance) |
| **Can measure solver variance?** | ✓ (directly) | ✓ (within-brief pairs) |
| **Reflects deployment?** | ✗ (fixed briefs unrealistic) | ✓ (fresh scaffold each time) |
| **Seed structure** | Simple (1000*draw + 13*cond) | Complex (1000*inst + 100*draw + 13*cond) |

---

## Application to Llama 8B Replication

The same design applies to `run_llama_replication.py` Phase 2:

**Command:**
```bash
python run_llama_replication.py --phase 2 --regenerate-briefs
```

**Llama will:**
1. Generate 4 fresh brief instances per condition (using Llama model)
2. Run 2 solver draws on each
3. Write results to `interventions_llama/solve_results.jsonl`

**Cross-model comparison:**
- Qwen Phase 2: briefs generated by Qwen
- Llama Phase 2: briefs generated by Llama
- Compare whether categorical structure (persistent vs. responsive) is stable across generator models

---

## Key Implications for Paper

### Methods Section
> "For each stochastic-scaffold condition (knowledge blind, knowledge oracle, both blind, both oracle), we generate four independent brief instances per question, each using a different seed (seed+0, seed+100, seed+200, seed+300). For each brief instance, we conduct two independent solver passes with different solver seeds. This yields 4 × 2 = 8 measurements per stochastic condition. For fixed-template conditions (control, reasoning), we conduct eight independent solver passes directly, with no brief-generation layer. This design enables separation of brief-instance variance from solver stochasticity."

### Results Section
> "Brief-instance variability accounted for 60–91% of measured variance in recovery rates for knowledge-based scaffolds, with solver stochasticity accounting for the remainder. In control conditions (no scaffold), 100% of variance is solver-driven, as expected."

### Discussion Section
> "The high sensitivity of recovery effects to brief-instance variation suggests that deployed interventions should either: (1) ensemble multiple independently-generated briefs, (2) invest in meta-learning approaches to brief generation, or (3) develop brief-instance-robust methods that work across diverse explanatory framings."

---

## Testing & Validation

### Mock Test
```bash
python generate_interventions.py --backend mock --limit 1
# Verifies loop structure without model calls
# Takes <1 second
```

### Expected Output
```
Generating interventions for 1 questions x 6 conditions
  Total measurements per question: 6 * 8 = 48
  Stochastic scaffolds (S, C): 4 brief instances x 2 solver draws = 8 measurements
  Fixed templates (R, control): 8 solver draws, no brief instances
Done. Results -> interventions/solve_results.jsonl
```

### Validation Checklist
- [ ] Mock run completes without errors
- [ ] JSONL contains brief_instance and solver_draw fields
- [ ] Stochastic conditions have brief_instance ∈ {1,2,3,4}, solver_draw ∈ {1,2}
- [ ] Fixed conditions have brief_instance = 0, solver_draw ∈ {1..8}
- [ ] Seed values follow expected pattern
- [ ] Labeling still works (aggregates across brief instances)
- [ ] Variance decomposition analysis produces 60–91% K-variance

---

## FAQ

**Q: Why 4 briefs and not 2 or 8?**  
A: 4 instances × 2 draws = 8 measurements (matching old design for compatibility). The 2-draw pair enables within-brief variance estimation, which is essential for variance decomposition.

**Q: What if I want more brief instances?**  
A: Change `num_instances=4` in `generate_brief_instances()` call. With N instances and 8/N draws each, you keep 8 total measurements. E.g., N=8 → 8 briefs × 1 draw, or N=2 → 2 briefs × 4 draws.

**Q: Does resume work if I change num_instances?**  
A: Yes, but only for new questions. Existing (question, condition, brief_instance, solver_draw) tuples are tracked separately, so changing the structure won't conflict.

**Q: How much longer does Phase 2 take?**  
A: Brief generation time is additive. Old design: ~12 hours. New design: +4× brief generation, so ~18–20 hours total (4 briefs per stochastic condition vs. 1 before).

**Q: Can I use --repeats 8 to get more measurements?**  
A: No. The `--repeats` argument is now only for legacy compatibility. Fixed templates always use 8 draws, and stochastic scaffolds always use 4×2. The parameter is ignored in Phase 2.

---

## Architecture Summary

```
run_generation(args, subset)
  ├─ For each question:
  │   ├─ For each condition:
  │   │   ├─ If condition in STOCHASTIC_SCAFFOLD_CONDS:
  │   │   │   ├─ generate_brief_instances() → {1..4}
  │   │   │   └─ For brief_inst 1..4:
  │   │   │       └─ For solver_draw 1..2:
  │   │   │           └─ solve_once()
  │   │   │
  │   │   └─ Else (FIXED_TEMPLATE_CONDS):
  │   │       └─ For solver_draw 1..8:
  │   │           └─ solve_once()
  │   │
  │   └─ Write to JSONL with (brief_instance, solver_draw)
  │
  └─ run_labeling()
      └─ Aggregate across all (brief_instance, solver_draw) pairs per condition
```

---

## References

- `generate_interventions.py`: Core implementation
- `run_llama_replication.py`: Applied to Llama Phase 2
- Commit: "Implement 4 briefs x 2 solver draws design for Phase 2"
- Paper section: Methods → "Intervention Measurement and Stochasticity"
