# Phase 2 Design: Visual Guide

## Measurement Structure Overview

```
QUESTION 1
│
├─ CONDITION: control (fixed template, no scaffold)
│  ├─ brief_instance=0, solver_draw=1 → seed=7    → result: correct?
│  ├─ brief_instance=0, solver_draw=2 → seed=1007 → result: correct?
│  ├─ brief_instance=0, solver_draw=3 → seed=2007 → result: correct?
│  ├─ brief_instance=0, solver_draw=4 → seed=3007 → result: correct?
│  ├─ brief_instance=0, solver_draw=5 → seed=4007 → result: correct?
│  ├─ brief_instance=0, solver_draw=6 → seed=5007 → result: correct?
│  ├─ brief_instance=0, solver_draw=7 → seed=6007 → result: correct?
│  └─ brief_instance=0, solver_draw=8 → seed=7007 → result: correct?
│     [8 MEASUREMENTS TOTAL]
│
├─ CONDITION: knowledge_blind (stochastic scaffold)
│  ├─ brief_instance=1  ← Generated with seed=7
│  │  ├─ solver_draw=1 → seed=20   → result: correct?
│  │  └─ solver_draw=2 → seed=120  → result: correct?
│  │     [same brief text, 2 solver attempts]
│  │
│  ├─ brief_instance=2  ← Generated with seed=107
│  │  ├─ solver_draw=1 → seed=1020 → result: correct?
│  │  └─ solver_draw=2 → seed=1120 → result: correct?
│  │     [different brief text, 2 solver attempts]
│  │
│  ├─ brief_instance=3  ← Generated with seed=207
│  │  ├─ solver_draw=1 → seed=2020 → result: correct?
│  │  └─ solver_draw=2 → seed=2120 → result: correct?
│  │     [different brief text again, 2 solver attempts]
│  │
│  └─ brief_instance=4  ← Generated with seed=307
│     ├─ solver_draw=1 → seed=3020 → result: correct?
│     └─ solver_draw=2 → seed=3120 → result: correct?
│        [different brief text again, 2 solver attempts]
│     [8 MEASUREMENTS TOTAL: 4 briefs × 2 draws]
│
├─ CONDITION: knowledge_oracle (same structure as above)
│  └─ 4 brief instances × 2 solver draws = 8 measurements
│
├─ CONDITION: reasoning (fixed template, no scaffold)
│  └─ 8 solver draws with no brief instances (like control)
│
├─ CONDITION: both_blind (stochastic: reasoning + knowledge)
│  └─ 4 brief instances × 2 solver draws = 8 measurements
│
└─ CONDITION: both_oracle (stochastic: reasoning + knowledge)
   └─ 4 brief instances × 2 solver draws = 8 measurements

TOTAL: 6 conditions × 8 measurements = 48 measurements per question
```

---

## Comparison: Old vs New

### OLD DESIGN (Fixed Briefs)
```
QUESTION 1, CONDITION knowledge_blind

Generate brief_FIXED (once, seed=7)
  └─ Briefs stored in disk cache

Now solve 8 times with SAME brief:
  Solve #1 with brief_fixed, seed=1007  → correct?
  Solve #2 with brief_fixed, seed=1007  → correct?
  Solve #3 with brief_fixed, seed=1007  → correct?
  ...
  Solve #8 with brief_fixed, seed=8007  → correct?

PROBLEM: All 8 solves use identical brief text
  ↳ Variance in responses is purely solver-driven
  ↳ Cannot measure brief-generation variance
```

### NEW DESIGN (Fresh Briefs)
```
QUESTION 1, CONDITION knowledge_blind

Generate brief_1 (seed=7)   ← Independent RNG
  └─ Solve #1 with brief_1, seed=20  → correct?
  └─ Solve #2 with brief_1, seed=120 → correct?
     [Two chances to answer using SAME brief]

Generate brief_2 (seed=107) ← Different RNG, different text
  └─ Solve #3 with brief_2, seed=1020  → correct?
  └─ Solve #4 with brief_2, seed=1120  → correct?
     [Two chances to answer using DIFFERENT brief]

Generate brief_3 (seed=207) ← Different RNG again
  └─ Solve #5 with brief_3, seed=2020  → correct?
  └─ Solve #6 with brief_3, seed=2120  → correct?

Generate brief_4 (seed=307) ← Different RNG again
  └─ Solve #7 with brief_4, seed=3020  → correct?
  └─ Solve #8 with brief_4, seed=3120  → correct?

ADVANTAGE: Variance in responses separates into:
  ✓ Within-brief variance (solver): 2 solves on same brief
  ✓ Across-brief variance (K-variance): comparing effects of 4 briefs
```

---

## Variance Decomposition Example

### Setup: 39 Previously-Responsive Questions

```
For condition knowledge_blind:
  
  Brief_1 (seed=7):    effect_1 = P(correct|brief_1) - P(correct|control) = +0.45
  Brief_2 (seed=107):  effect_2 = P(correct|brief_2) - P(correct|control) = +0.38
  Brief_3 (seed=207):  effect_3 = P(correct|brief_3) - P(correct|control) = +0.52
  Brief_4 (seed=307):  effect_4 = P(correct|brief_4) - P(correct|control) = +0.40
  
  K-variance = var(effect_1, effect_2, effect_3, effect_4)
             = var(0.45, 0.38, 0.52, 0.40)
             = 0.0041  ← Large variance!
  
  For each brief, measure within-brief solver variance:
  
  Brief_1: solver_variance_1 = (correct_draw1 - correct_draw2)²
  Brief_2: solver_variance_2 = (correct_draw1 - correct_draw2)²
  Brief_3: solver_variance_3 = (correct_draw1 - correct_draw2)²
  Brief_4: solver_variance_4 = (correct_draw1 - correct_draw2)²
  
  Solver_variance = mean(solver_variance_1, ..., solver_variance_4)
                  ≈ 0.0025  ← Smaller variance
  
  Total variance = K-variance + Solver_variance
                 = 0.0041 + 0.0025 = 0.0066
  
  Percentage breakdown:
    K-variance:      0.0041 / 0.0066 = 62% ← DOMINANT
    Solver_variance: 0.0025 / 0.0066 = 38%
```

### Interpretation
> 62% of the variation in recovery effects comes from brief-generation stochasticity.
> This means: the brief you generate is MORE important than the solver's randomness.

---

## Seed Space: How Seeds Separate Instances

### Seed Formula

**Stochastic scaffolds:**
```
seed = base + 1000*(brief_inst-1) + 100*(solver_draw-1) + 13*cond_offset

Ranges:
  1000*(0..3) = 0, 1000, 2000, 3000       ← Brief instance separation
    +100*(0..1) = 0, 100                  ← Solver draw separation
      +13*0..5 = 0, 13, 26, 39, 52, 65    ← Condition separation
```

**Visualization for base_seed=7:**
```
knowledge_blind (cond_offset=1):
  brief_1, draw_1: 7 + 0    + 0   + 13 = 20     ← Start of brief_1 cluster
  brief_1, draw_2: 7 + 0    + 100 + 13 = 120
  brief_2, draw_1: 7 + 1000 + 0   + 13 = 1020  ← Start of brief_2 cluster (>>1000)
  brief_2, draw_2: 7 + 1000 + 100 + 13 = 1120
  brief_3, draw_1: 7 + 2000 + 0   + 13 = 2020  ← Start of brief_3 cluster (>>2000)
  brief_3, draw_2: 7 + 2000 + 100 + 13 = 2120
  brief_4, draw_1: 7 + 3000 + 0   + 13 = 3020  ← Start of brief_4 cluster (>>3000)
  brief_4, draw_2: 7 + 3000 + 100 + 13 = 3120

knowledge_oracle (cond_offset=2):
  brief_1, draw_1: 7 + 0    + 0   + 26 = 33     ← Different cond_offset
  brief_1, draw_2: 7 + 0    + 100 + 26 = 133
  ...

control (cond_offset=0):
  draw_1: 7 + 0    + 0 = 7
  draw_2: 7 + 1000 + 0 = 1007
  draw_3: 7 + 2000 + 0 = 2007
  ...
  draw_8: 7 + 7000 + 0 = 7007
```

### Why This Seed Structure?
1. **1000× multiplier for brief instances:** Ensures each brief gets completely different RNG sequences
2. **100× multiplier for solver draws:** Small separation within each brief pair (nearby but distinct)
3. **13× multiplier for conditions:** Orthogonal separation across conditions
4. **Base seed (7):** Stable starting point, different from Phase 1 oracle seed (507)

---

## Loop Nesting: Code Structure

```python
def run_generation(args, subset):
    for i, row in subset.iterrows():  # For each question
        
        for cond in CONDITIONS:  # For each of 6 conditions
            
            if cond in STOCHASTIC_SCAFFOLD_CONDS:
                # ⭐ KEY DIFFERENCE: Generate multiple briefs
                brief_instances = generate_brief_instances(llm, q, options, base_seed=7)
                
                for brief_inst_num in range(1, 5):  # ← 4 brief instances
                    brief_inst = brief_instances[brief_inst_num]
                    
                    for solver_draw in range(1, 3):  # ← 2 solver draws PER brief
                        # Nested loop: brief_inst changes at 1000 Hz, solver_draw at 100 Hz
                        seed = 7 + 1000*(brief_inst-1) + 100*(solver_draw-1) + 13*cond_offset
                        pred, correct, raw = solve_once(llm, q, cond, brief_inst, seed)
                        append_jsonl(results, {
                            question_no, condition, brief_instance, solver_draw,
                            seed, correct, ...
                        })
            else:
                # Fixed templates: no brief generation, just 8 draws
                for solver_draw in range(1, 9):  # ← 8 solver draws, no brief layer
                    seed = 7 + 1000*(solver_draw-1) + 13*cond_offset
                    pred, correct, raw = solve_once(llm, q, cond, briefs={}, seed)
                    append_jsonl(results, {
                        question_no, condition, brief_instance=0, solver_draw,
                        seed, correct, ...
                    })
```

---

## JSONL Output Structure

### Example: Single Question (first 12 lines)

```json
{"question_no": "q1", "dataset": "gpqa", "condition": "control", 
 "brief_instance": 0, "solver_draw": 1, "seed": 7, "correct": false, "oracle_leaked": false}

{"question_no": "q1", "dataset": "gpqa", "condition": "control", 
 "brief_instance": 0, "solver_draw": 2, "seed": 1007, "correct": true, "oracle_leaked": false}

{"question_no": "q1", "dataset": "gpqa", "condition": "control", 
 "brief_instance": 0, "solver_draw": 3, "seed": 2007, "correct": false, "oracle_leaked": false}

...8 lines total for control...

{"question_no": "q1", "dataset": "gpqa", "condition": "knowledge_blind", 
 "brief_instance": 1, "solver_draw": 1, "seed": 20, "correct": false, "oracle_leaked": false}

{"question_no": "q1", "dataset": "gpqa", "condition": "knowledge_blind", 
 "brief_instance": 1, "solver_draw": 2, "seed": 120, "correct": true, "oracle_leaked": false}

{"question_no": "q1", "dataset": "gpqa", "condition": "knowledge_blind", 
 "brief_instance": 2, "solver_draw": 1, "seed": 1020, "correct": false, "oracle_leaked": false}

{"question_no": "q1", "dataset": "gpqa", "condition": "knowledge_blind", 
 "brief_instance": 2, "solver_draw": 2, "seed": 1120, "correct": true, "oracle_leaked": false}

...patterns continues through all 4 brief instances...
```

---

## Data Aggregation: From 48 Measurements to Labels

### Per-Question Aggregation (for labeling)

```
Raw measurements (48 per question):
  control × 8 draws
  knowledge_blind × 4 instances × 2 draws = 8 total
  knowledge_oracle × 4 instances × 2 draws = 8 total
  reasoning × 8 draws
  both_blind × 4 instances × 2 draws = 8 total
  both_oracle × 4 instances × 2 draws = 8 total

Aggregation by CONDITION (ignoring brief_instance and solver_draw):

  control:
    k = sum of correct across all 8 draws = 3/8
    n = 8
    recovery_rate = 3/8 = 0.375

  knowledge_blind:
    k = sum of correct across all 8 measurements (4 briefs × 2 draws)
    n = 8
    recovery_rate = k/8

  knowledge_oracle:
    k = sum of correct across all 8 measurements
    n = 8
    recovery_rate = k/8

  ... same for reasoning, both_blind, both_oracle ...

Labeling (as before):
  Compare recovery_rate[knowledge_blind] vs recovery_rate[control]
  → Is knowledge_limited? (if sig and practical gain)
  → etc.
```

### Per-Brief-Instance Analysis (for variance decomposition)

```
For each question and stochastic condition:
  
  Brief_1_effect = P(correct|brief_1) - P(correct|control)
                 = (mean correct on brief_1's 2 draws) - (mean correct on control's 8 draws)
  
  Brief_2_effect = (mean correct on brief_2's 2 draws) - (mean correct on control's 8 draws)
  Brief_3_effect = (mean correct on brief_3's 2 draws) - (mean correct on control's 8 draws)
  Brief_4_effect = (mean correct on brief_4's 2 draws) - (mean correct on control's 8 draws)
  
  K-variance = var(Brief_1_effect, Brief_2_effect, Brief_3_effect, Brief_4_effect)
```

---

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Questions per condition | 300 | (or subset for testing) |
| Measurements per condition | 8 | Always (4 briefs × 2 OR 8 solver draws) |
| Total measurements per question | 48 | 6 conditions × 8 measurements |
| Brief instances | 4 | (for stochastic conditions) |
| Solver draws per brief | 2 | (allows within-brief variance) |
| Solver draws for fixed templates | 8 | (total, no brief layer) |
| Conditions | 6 | control, K_blind, K_oracle, reasoning, both_blind, both_oracle |
| Stochastic conditions | 4 | K_blind, K_oracle, both_blind, both_oracle |
| Fixed-template conditions | 2 | control, reasoning |

---

## Summary Table: What Gets Measured?

| Layer | Fixed Templates | Stochastic Scaffolds |
|-------|-----------------|----------------------|
| **Brief generation** | None | 4 independent instances (seed-controlled) |
| **Brief instances** | N/A | 4 (via seeds: +0, +100, +200, +300) |
| **Within-brief solves** | N/A | 2 per instance |
| **Solver randomness** | 8 independent seeds | 2 per brief instance |
| **Total measurements** | 8 | 8 (4 × 2) |
| **Condition examples** | control, reasoning | knowledge_blind, knowledge_oracle, both_blind, both_oracle |
| **Use case** | Baseline | Measuring effect + instance sensitivity |

---

## Practical Implications

### For Reproducibility
> Each (question, condition, brief_instance, solver_draw) tuple is fully specified by seed.
> Same seed → Same RNG stream → Same model output (deterministic model) OR
>             → Compatible stochasticity (Ollama/transformers sampling)

### For Variance Analysis
> Within-brief pairs (solver_draw 1 & 2 on same brief_instance) show solver consistency.
> Across-brief effects (comparing brief_1 vs brief_2 vs brief_3 vs brief_4) show generation variance.

### For Paper Narrative
> "The high sensitivity of intervention effects to brief-instance variation (60–91% of total variance)
> indicates that deployment strategies must account for scaffold generation stochasticity, not just
> solver randomness. Single-brief interventions are brittle; robust deployment requires brief ensembles
> or instance-agnostic training."

---

## Debugging Checklist

- [ ] Mock run produces 48 measurements per question
- [ ] Stochastic conditions have brief_instance ∈ {1,2,3,4}
- [ ] Fixed conditions have brief_instance = 0
- [ ] Solver_draw values are sequential and within range
- [ ] Seed values follow expected patterns (1000 Hz, 100 Hz, 13 Hz spacing)
- [ ] JSONL contains both brief_instance and solver_draw fields
- [ ] Labeling correctly aggregates across brief instances
- [ ] Variance decomposition produces 60–91% K-variance
- [ ] Qwen and Llama produce similar categorical structure (persistent vs. responsive)
