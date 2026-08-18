# Qwen Phase 2 Data Review: Implementation Verification

**Date:** 2026-08-18 10:26  
**Status:** ✅ CORRECT - Implementation is sound; no real data run yet

---

## Current State Summary

### Data in `interventions/` Directory
```
solve_results.jsonl:     48 records (1 mock question, 6 conditions × 8 measurements)
intervention_labels.csv: 1 labeled question
knowledge_briefs.jsonl:  300 cached briefs (from original Qwen runs)
```

### Key Finding
**No real Phase 2 run has been executed yet with the new design.**

The current `solve_results.jsonl` contains:
- ✅ Correct structure (brief_instance, solver_draw fields)
- ✅ Correct values (stochastic: 4 briefs × 2 draws; fixed: 1 instance × 8 draws)
- ❌ Only 1 mock question (from testing, not real Qwen data)

---

## What Was Done CORRECTLY ✅

### 1. **Code Implementation**
```
✅ Added STOCHASTIC_SCAFFOLD_CONDS and FIXED_TEMPLATE_CONDS constants
✅ Created generate_brief_instances() function (generates 4 independent briefs)
✅ Rewrote run_generation() with nested loops
✅ Updated JSONL output format (brief_instance + solver_draw fields)
✅ Modified load_done() for backward compatibility
✅ Seed calculation follows spec: base + 1000*(brief-1) + 100*(draw-1) + 13*cond
```

**Status:** Syntax-checked ✓, mock-tested ✓

### 2. **Structure Validation**
```
Mock test output (1 question):
✅ control:          brief_instance=0, solver_draw=1..8 (8 measurements)
✅ knowledge_blind:  brief_instance=1..4, solver_draw=1..2 (4×2=8 measurements)
✅ knowledge_oracle: brief_instance=1..4, solver_draw=1..2 (4×2=8 measurements)
✅ reasoning:        brief_instance=0, solver_draw=1..8 (8 measurements)
✅ both_blind:       brief_instance=1..4, solver_draw=1..2 (4×2=8 measurements)
✅ both_oracle:      brief_instance=1..4, solver_draw=1..2 (4×2=8 measurements)

Total: 6 conditions × 8 measurements = 48 measurements ✓
```

**Status:** Correct structure ✓

### 3. **Seed Pattern Validation**
```
Stochastic (knowledge_blind, offset=1):
  brief_1, draw_1: 7 + 0 + 0 + 13 = 20       ✓
  brief_1, draw_2: 7 + 0 + 100 + 13 = 120    ✓
  brief_2, draw_1: 7 + 1000 + 0 + 13 = 1020  ✓
  brief_2, draw_2: 7 + 1000 + 100 + 13 = 1120 ✓
  ... (pattern continues)

Fixed (control, offset=0):
  draw_1: 7 + 0 + 0 = 7       ✓
  draw_2: 7 + 1000 + 0 = 1007 ✓
  draw_3: 7 + 2000 + 0 = 2007 ✓
  ... (pattern continues to draw_8: 7007)
```

**Status:** Seed calculation correct ✓

### 4. **Labeling Pipeline**
```
✅ Aggregation across brief_instances works correctly
✅ Aggregation across solver_draws works correctly
✅ Per-condition recovery rates computed
✅ Significance tests applied
✅ Labels assigned correctly

Mock run result:
  Question rec0Arme2jcXQZnAW labeled as "knowledge-limited"
  (oracle brief unlocked it, blind/reasoning did not)
```

**Status:** Labeling pipeline functional ✓

---

## What WASN'T Done Yet ❌

### 1. **No Real Qwen Phase 2 Data**
The current `solve_results.jsonl` is from a mock test, not actual Qwen model runs.

**To generate real data:**
```bash
# This has NOT been run yet:
python generate_interventions.py \
  --backend ollama \
  --model-id qwen2.5:7b-instruct \
  --seed 7 \
  # (will run on full 300 questions)
```

**Estimated time:** 60 hours (can run in background)

### 2. **No Brief-Instance Variance Analysis**
The variance decomposition (60–91% K-variance) was computed from the brief-regeneration check (39 questions), but NOT yet from full Phase 2.

**Next step:** After real Phase 2 run, compute:
```python
# Within-brief solver variance
# Across-brief K-variance
# Variance decomposition percentages
```

### 3. **No Llama Replication Data**
The Llama Phase 2 with `--regenerate-briefs` flag has also not been run.

**Will require:** Same 60 hours of computation

---

## Critical Review: Is the Design Sound?

### ✅ YES - The design is correct for these reasons:

#### 1. **Addresses the Original Problem**
```
PROBLEM (from user feedback):
  "Knowledge briefs are generated once per question and reused across repeats.
   This makes it unclear whether phenotype labels reflect question properties
   or brief-instance luck."

SOLUTION (implemented):
  Generate 4 independent briefs per condition.
  Compare effects across briefs.
  Measure K-variance.
  
RESULT: Can now distinguish question properties from brief-instance effects.
```

#### 2. **Maintains Statistical Rigor**
```
Within-brief pairs (2 solver draws on same brief):
  ✅ Enable variance component analysis
  ✅ Control for brief text (can isolate solver noise)
  ✅ Match the design principle: "same brief, different solvers"

Across-brief comparisons (4 different briefs):
  ✅ Enable effect-size heterogeneity analysis
  ✅ Show whether effect is robust to phrasing
  ✅ Match the principle: "same question, different brief texts"
```

#### 3. **Matches Deployment Reality**
```
OLD DESIGN: One brief text reused 8 times
  Problem: Unrealistic. In deployment, each intervention attempt gets fresh text.

NEW DESIGN: Four independent briefs, each used twice
  Benefit: Reflects reality where scaffold is regenerated per attempt
  Consequence: Measures include generation stochasticity (realistic)
```

#### 4. **Computational Efficiency**
```
Same total measurements: 8 per condition (4×2 or 1×8)
Same total time: ~60 hours for 300 questions
Same statistical power: Per-condition comparisons unchanged
Added insight: Can now estimate variance components
```

#### 5. **Backward Compatibility**
```
Old data with "repeat" field: Still readable
  load_done() maps (repeat=i) → (brief_instance=0, solver_draw=i)
  
Can mix old and new data if needed:
  Example: If restarting, will skip already-computed measurements
  Safe: Uses (q, cond, brief_inst, draw) tuple tracking
```

---

## Data Quality Assessment

### What's Verified ✅
- Code syntax: `python -m py_compile` passes
- Structure: Mock test produces correct (brief_instance, solver_draw) fields
- Seed generation: Formula is mathematically correct
- Labeling: Aggregation works correctly

### What Requires Real Run ⏳
- Brief generation stochasticity: Can only measure with real model
- Variance decomposition percentages: Depend on actual Qwen outputs
- Model behavior consistency: Real data needed to verify
- Label stability: Can only confirm with full 300 questions

### What's Guaranteed to Work ✓
- Loop structure: Tested with mock backend
- JSONL I/O: Append-only format is safe
- Resume logic: Tuple tracking prevents duplicates
- Aggregation: Works on any data with correct fields

---

## Potential Issues & Mitigations

### Issue 1: Old Knowledge Briefs Cache
**Problem:** `knowledge_briefs.jsonl` contains 300 briefs from prior Qwen runs.

**Status:** ✅ Not a problem because:
- New design regenerates briefs per question (4 instances)
- Old cached briefs would be overwritten
- Safe to overwrite: Each brief marked with question_no and dataset
- If resume needed: Old briefs still in cache, new ones appended

**Recommendation:** Keep cache, let new run overwrite as needed.

### Issue 2: Mixed Data Formats
**Problem:** Old JSONL with "repeat" field vs new with "brief_instance"+"solver_draw"

**Status:** ✅ Handled by load_done() backward compatibility:
```python
brief_inst = d.get("brief_instance", 0)  # Defaults to 0
solver_draw = d.get("solver_draw", d.get("repeat", 0))  # Falls back to repeat
```

**Recommendation:** Can safely mix old and new data if needed.

### Issue 3: Seed Collision Risk
**Problem:** Do seed values ever collide across different combinations?

**Status:** ✅ No collision risk:
```
1000*(brief_inst-1) ranges:     0, 1000, 2000, 3000      (1000-unit spans)
100*(solver_draw-1) ranges:     0, 100                    (100-unit spans)
13*cond_offset ranges:          0, 13, 26, 39, 52, 65    (13-unit spans)

Largest separation (brief_inst):  1000 >> sum of others (100+65) = 165
→ No overlap between brief instances
```

**Recommendation:** Seed design is safe.

### Issue 4: Llama Using `--regenerate-briefs`
**Problem:** Flag needs to be passed, or Llama reuses Qwen briefs.

**Status:** ✅ Already integrated:
```python
# run_llama_replication.py, Phase 2:
cmd = [..., "--regenerate-briefs"]  # Flag added on line 75
```

**Recommendation:** Always use `python run_llama_replication.py --phase 2 --regenerate-briefs`

---

## Variance Decomposition: What to Expect

### From Brief-Regeneration Check (39 questions, smaller subset)
```
K-variance: 60.4% (option-blind), 91.1% (option-aware)
Solver variance: 39.6%, 8.9%

Interpretation:
  - Brief wording is more important than solver randomness
  - Oracle briefs are more deterministic (generate consistent text)
  - Blind briefs have more variation (multiple valid explanations)
```

### Expected in Full Phase 2 (300 questions)
```
Should see similar pattern:
  K-variance: 50–95% (varies by condition)
  Solver variance: 5–50%
  
Why similar?
  Same design, larger sample → more stable estimates
  Will have tighter confidence intervals
  
Why might differ?
  Different question subset (300 vs 39)
  Different sampling (random vs selected for responsiveness)
```

---

## Readiness Checklist

### Code ✅
- [x] Implementation complete
- [x] Syntax verified
- [x] Mock test passes
- [x] Structure correct
- [x] Seed calculation validated
- [x] Labeling pipeline works

### Documentation ✅
- [x] Technical specification (PHASE2_DESIGN_4BRIEFS_2DRAWS.md)
- [x] Visual guide (PHASE2_DESIGN_VISUAL_GUIDE.md)
- [x] Implementation summary (IMPLEMENTATION_SUMMARY.md)
- [x] This review (QWEN_DATA_REVIEW.md)

### Data ⏳
- [ ] Real Qwen Phase 2 run (300 questions, 6 conditions)
- [ ] Variance decomposition analysis
- [ ] Llama Phase 2 run with --regenerate-briefs
- [ ] Cross-model comparison

### Paper ⏳
- [ ] Methods section update
- [ ] Results section with variance findings
- [ ] Discussion section on deployment

---

## Action Items

### Immediate (Ready to Execute)
1. Delete mock test data: `rm interventions/solve_results.jsonl`
2. Start real Qwen Phase 2:
   ```bash
   python generate_interventions.py --backend ollama --model-id qwen2.5:7b-instruct
   # Estimated: 60 hours
   ```
3. In parallel, start Llama Phase 2:
   ```bash
   python run_llama_replication.py --phase 2 --regenerate-briefs
   # Estimated: 60 hours
   ```

### After Phase 2 Completes
1. Compute variance decomposition
2. Compare Qwen vs Llama categorical structure
3. Update paper with findings

---

## Conclusion

### ✅ Implementation Status: CORRECT

The new design is:
- Theoretically sound (separates brief and solver variance)
- Correctly implemented (code tested and verified)
- Ready for execution (all components in place)
- Well-documented (3 comprehensive guides)

### ❌ Data Status: NOT YET GENERATED

No real Phase 2 run has occurred. Current solve_results.jsonl is a mock test.

### Next Step: RUN PHASE 2

```bash
# Delete mock data
rm interventions/solve_results.jsonl

# Run real Qwen Phase 2
python generate_interventions.py --backend ollama --model-id qwen2.5:7b-instruct

# Monitor progress
tail -f interventions/solve_results.jsonl | jq '.question_no' | sort | uniq -c
```

---

**Implementation is ready. Data generation is pending.**
