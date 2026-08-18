# Brief-Regeneration Robustness Check: Design Explanation

## Current Running Job: Persistent Questions Test

**Status:** Running in background (45/98 complete)  
**Model:** Qwen 2.5 7B  
**Purpose:** Test whether "persistent" (non-responsive) failures stay persistent when briefs are regenerated

---

## Design: Do We Use New Briefs for Every Refresh?

### ❌ NO - NOT Every Refresh Gets a Fresh Brief

The current design is:
```
For each question:
  Generate control sample: 4 fresh solves (no brief)
  
  Generate STEM-ONLY BRIEF once
    └─ Use this SAME brief for 4 solver attempts
  
  Generate CHOICE-AWARE BRIEF once
    └─ Use this SAME brief for 4 solver attempts
  
  (Optional) Generate reasoning sample: 4 fresh solves (static template)

Total: 1 stem-only brief × 4 attempts + 1 choice-aware brief × 4 attempts = 12 solves
```

### Structure Per Question
```
CONTROL:                4 solves (no brief)
STEM-ONLY (S) FRESH:    1 brief → 4 solves (SAME brief for all 4)
CHOICE-AWARE (C) FRESH: 1 brief → 4 solves (SAME brief for all 4)
REASONING FRESH:        4 solves (static scaffold)

Total: 12 solve calls + 2 brief calls per question
```

---

## Key Difference: Old vs New vs This Job

| Design | Briefs Generated | Solves Per Brief | Total Measurements | What It Measures |
|--------|------------------|------------------|-------------------|-----------------|
| **Phase 1 Original** | 1 per condition | 8 | 8 per condition | Solver variance (fixed brief) |
| **Phase 2 New Design** | 4 per condition | 2 | 8 per condition (4×2) | Solver + brief variance |
| **This Job (Brief Regen Check)** | 1 per question | 4 | 4 per condition | Solver variance (fresh brief, but fixed across 4 attempts) |

---

## Why This Design (Not 4 Briefs × 2 Draws)?

This job was designed to test a **narrower question** than Phase 2:

### Job's Question
> "Do the persistent/responsive categories REMAIN the same if we use a fresh brief 
> (instead of the cached Phase 1 brief) and fresh solver attempts (instead of Phase 1 solves)?"

This requires:
- Fresh brief (to avoid Phase 1 brief luck)
- 4 solver attempts (to measure recovery rate)
- But the SAME fresh brief across those 4 attempts (no need for variance decomposition here)

### Phase 2's Question
> "How much of the intervention effect variance comes from brief-generation stochasticity 
> vs solver stochasticity?"

This requires:
- 4 independent briefs (to measure generation variance)
- 2 solves per brief (to measure solver variance within each brief)
- 4×2 = 8 total measurements (same as before, but now separable)

---

## This Job's Measurement Logic

### For Persistent Questions (Currently Running)

```
Design per question:

1. CONTROL (fresh):
   4 independent solver attempts (no scaffold)
   All with different seeds from FRESH_SEED_BASE
   Recovery rate: P(correct | no scaffold)

2. STEM-ONLY (fresh):
   Generate 1 new stem-only brief (seed=FRESH_SEED_BASE + 1)
   Use THIS BRIEF for all 4 solver attempts
   All with different seeds (FRESH_SEED_BASE + 100 + 13*rep)
   Recovery rate: P(correct | fresh stem-only brief, SAME brief, varied solver)
   Effect: Recovery(S_fresh) - Recovery(control_fresh)

3. CHOICE-AWARE (fresh):
   Generate 1 new choice-aware brief (seed=FRESH_SEED_BASE + 2)
   Use THIS BRIEF for all 4 solver attempts
   All with different seeds (FRESH_SEED_BASE + 200 + 13*rep)
   Recovery rate: P(correct | fresh choice-aware brief, SAME brief, varied solver)
   Effect: Recovery(C_fresh) - Recovery(control_fresh)

Decision rule:
  - Persistent if G_S < 2 AND G_C < 2 (neither brief recovers the question)
  - Responsive if G_S >= 2 OR G_C >= 2 (at least one brief recovers)
  
  (Same 2-out-of-4 rule as Phase 1)
```

### Seed Structure
```
FRESH_SEED_BASE = 900000 (completely disjoint from Phase 1 seeds)

Control:       seed = 900000 + 13*rep     (900000, 900013, 900026, 900039)
Stem-only:     seed = 900100 + 13*rep     (900100, 900113, 900126, 900139)
Choice-aware:  seed = 900200 + 13*rep     (900200, 900213, 900226, 900239)
Reasoning:     seed = 900300 + 13*rep     (900300, 900313, 900326, 900339)

→ Completely independent from Phase 1 (which used seeds 7, 507, 1007, 2007, etc.)
```

---

## Why NOT Use Phase 2 Design Here?

The Phase 2 design (4 briefs × 2 draws) would be **overkill** for this job because:

### This Job Only Tests PERSISTENCE
```
Question: "Does the failure category change?"
Answer needed: Binary - persistent or responsive (or changed to responsive)?
Granularity needed: Single effect estimate per condition

4 briefs × 2 draws would give:
  - 4 separate effect estimates (one per brief)
  - Within-brief variance estimates
  - Variance decomposition
  
But for persistence check, we only care:
  - Does brief S help? (Yes/No)
  - Does brief C help? (Yes/No)
  → Single decision per condition is enough
```

### Computational Cost
```
This job (current design):
  2 briefs × 4 solves = 8 solves per question
  98 questions × 8 = 784 solves
  Time: ~2-3 hours

If we used Phase 2 design (4 briefs × 2 draws):
  4 briefs × 2 solves = 8 solves per question (same!)
  BUT: 4 briefs instead of 2
  98 questions × (4×2 briefs + 4×2 solves) = 98 × 16 brief+solve calls
  Time: ~4-6 hours (2x longer)
  
Result: More data, but more expensive, for a simpler question
```

---

## What This Job IS Good For

### ✅ What It Tests
1. **Persistence Asymmetry**
   - Are persistent questions truly "persistent" as a property of the question?
   - Or are they "persistent" due to luck with Phase 1 briefs?
   
2. **Category Stability**
   - Do the 81% of persistent questions that remained persistent in pilot generalize?
   - Can we trust the "persistent" label?

3. **Cross-Brief Consistency**
   - If a question is persistent with fresh stem-only brief, is it also persistent with fresh choice-aware brief?
   - Or does brief type matter even for persistent failures?

### ❌ What It Does NOT Test
- Variance decomposition (60-91% split)
- Brief-instance sensitivity
- Within-brief solver consistency
- Per-brief effect heterogeneity

→ **Phase 2 will test those things** with the 4×2 design

---

## Relationship to Phase 2

### This Job (Persistent Check)
```
BEFORE Phase 2 starts:
  Test: Do persistent labels from Phase 1 hold up under fresh briefs?
  Purpose: Validate that "persistent" is a real question property
  Data: 98 questions, 2 brief types, 4 solves each = small pilot
  Output: Binary classification (still persistent? or now responsive?)
```

### Phase 2 (Full Measurement)
```
AFTER this job completes:
  Test: Full variance decomposition on 300 questions
  Purpose: Measure brief-generation variance (60-91% expected)
  Data: 300 questions, 4 brief types, 2 solves each = complete dataset
  Output: Variance percentages, effect heterogeneity, cross-model comparison
```

---

## Current Job Progress

### What's Happening Right Now
```
Question 45 / 98:
  ✓ Generated fresh control sample
  ✓ Generated fresh stem-only brief
  ✓ Solved 4 times against that brief
  ✓ Generated fresh choice-aware brief
  ✓ Solved 4 times against that brief
  
  → Computing: Is this question still persistent?
```

### Expected Pattern
```
If designed correctly, we expect:
  ~81 questions remain persistent (81%)
  ~17 questions become responsive (19%)
  
Early data (21 questions): 17 persistent, 4 responsive = 81% ✓
Current progress (45 questions): Likely similar pattern
```

### When This Completes
- Full 98-question sample
- Can compute confidence intervals on persistence rate
- Will confirm asymmetry holds

### Then Phase 2 Can Start
- Knowing that persistent ≈ stable property
- Ready to measure brief-instance variance on full dataset
- Can interpret results with confidence

---

## Summary

### Current Design
```
Per question:
  1 control brief × 4 solves
  1 stem-only brief × 4 solves
  1 choice-aware brief × 4 solves
  = 12 solves total

Per 98 questions:
  ~1,176 solves
  Time: ~2-3 hours
  
This answers: "Do persistent stay persistent?"
```

### vs Phase 2 Design
```
Per question:
  4 control × 1 solve each = 4 solves
  4 stem-only briefs × 2 solves each = 8 solves
  4 choice-aware briefs × 2 solves each = 8 solves
  = 20 solves total

Per 300 questions:
  ~6,000 solves
  Time: ~60 hours
  
This answers: "What fraction of variance is brief-generation?"
```

### Key Insight
- **This job** uses 1 fresh brief × 4 solves per condition
- **Phase 2** uses 4 fresh briefs × 2 solves per condition
- Same total solves (8) but different structure
- Different questions, so different designs make sense

---

## Answer to Your Question

**"Does it use a new generated brief on every refresh for all 8 refreshes?"**

### ❌ NO

**What it actually does:**
1. Generates 1 fresh stem-only brief
2. Uses that SAME brief for 4 solver attempts
3. Generates 1 fresh choice-aware brief
4. Uses that SAME brief for 4 solver attempts

**Why not 4 briefs × 2 draws?**
- This job only needs to know: persistent or responsive?
- Doesn't need variance decomposition
- 4 briefs × 2 draws overkill (2x computational cost)
- Single brief per condition sufficient for binary decision

**Phase 2 will be different:**
- Will use 4 briefs × 2 draws for stochastic conditions
- Will enable variance decomposition
- Will measure brief-instance variance (60-91% expected)

---

## Timeline

```
NOW:            Brief-regeneration check running (persistent, 45/98)
→ ~2 hours:     Job completes, results analyzed
→ Confirm:      81% persistence rate holds in full sample

THEN:           Delete mock Phase 2 data
→ Phase 2:      Start Qwen and Llama Phase 2 (60 hours each)
→ 3-4 days:     Phase 2 completes
→ Analysis:     Compute variance decomposition
→ Paper:        Update with findings
```

Does this clarify the design? Any other questions about how the persistent job works vs Phase 2?
