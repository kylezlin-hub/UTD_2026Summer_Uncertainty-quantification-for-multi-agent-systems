# Llama Phase 2: Issue Detected and Fixed

**Issue Date:** 2026-08-18 12:02 UTC-5  
**Status:** FIXED and restarted

---

## Problem Detected

The initial Llama Phase 2 run (started at 10:50) was **NOT using the new 4-brief design** despite:
1. Code having correct implementation
2. Script being called with `--regenerate-briefs` flag
3. Previous runs using old format with single brief

### Evidence

**Data format issue:**
- ❌ Missing `brief_instance` field
- ❌ Missing `solver_draw` field
- ✅ Using old `repeat` field instead

**Data structure observed:**
- 5,707 records with old format
- Mixed brief instances (0, 1, 2, 3, 4) from previous runs
- Not the clean 4×2×7 structure expected

**Root cause:** Old data from previous Llama runs (with single-brief design) was in `interventions_llama/` directory, and the code's backward compatibility logic was loading and appending to this old data rather than starting fresh.

---

## Fix Applied

### Step 1: Cleaned Old Data
- Removed `interventions_llama/solve_results.jsonl` (5,707 old records)
- Removed `interventions_llama/knowledge_briefs.jsonl` 
- Backed up old files as `.backup_old_format` for reference

### Step 2: Restarted Clean
- Command: `python run_llama_replication.py --phase 2`
- Fresh start with empty output directory
- Code forced to begin from question 1 with new brief generation

---

## Verification: New Run is Correct

**Record format confirmed:**
```
Fields in new records:
  ✅ brief_instance (was missing)
  ✅ solver_draw (was missing)
  ✅ seed (correct calculation)
```

**Sample new record:**
```json
{
  "question_no": "rec0Arme2jcXQZnAW",
  "dataset": "mmlu-pro",
  "condition": "control",
  "brief_instance": 0,
  "solver_draw": 1,
  "seed": 7,
  "pred": "B",
  "correct": false,
  "oracle_leaked": false,
  "raw": "..."
}
```

**Progress:** 8 records generated (just starting fresh)

---

## What Will Now Be Generated

### Correct Structure: 300 questions × 7 conditions × 8 measurements = 16,800 total

**For stochastic scaffolds** (5 conditions):
- knowledge_blind
- knowledge_oracle  
- both_blind
- both_oracle
- (reasoning excluded - it's fixed template)

**Per condition: 4 brief instances × 2 solver draws = 8 measurements**

**For fixed templates** (2 conditions):
- control
- reasoning

**Per condition: 8 solver draws, brief_instance=0**

### Seed Structure (Verified in Restarted Run)
```
Seed = base (7) + 1000*(brief_instance-1) + 100*(solver_draw-1) + 13*condition_offset

Example for knowledge_blind, brief 1, draw 1:
  seed = 7 + 1000*0 + 100*0 + 13*1 = 20

Example for knowledge_blind, brief 2, draw 1:
  seed = 7 + 1000*1 + 100*0 + 13*1 = 1020

Example for control, draw 1:
  seed = 7 + 1000*0 + 100*0 + 13*0 = 7
```

---

## Timeline Impact

**Old status:** Process thought to be running correctly but using wrong design  
**New status:** Process restarted, now using correct design

**Estimated completion:** ~60 hours from restart (now ~12:30 UTC-5 on 2026-08-18)  
**Expected finish:** ~2026-08-20 12:30 UTC-5

---

## Why This Matters

The 4-brief design is **critical** for the paper's core finding:

**Without 4 briefs per question:**
- Can't separate brief-instance variance (K-variance) from solver variance
- Can't quantify "60-91% brief-instance variance" claim
- Can't validate that "magnitudes are instance-dependent"

**With correct 4-brief design:**
- ✅ Each brief instance is independent (different seed, different wording)
- ✅ Two solver draws per brief measure solver consistency
- ✅ Four briefs enable variance decomposition (K-variance vs solver variance)
- ✅ Validates asymmetry hypothesis: "categories stable, magnitudes not"

---

## Lessons Learned

1. **Old data files can silently interfere** 
   - Backward compatibility (loading old format) is good for resume safety
   - But it masked the fact that new format wasn't being used
   - Solution: Clean data directories before major design changes

2. **Verification during startup is essential**
   - Should have checked first few records immediately after "starting" run
   - Detected error 2 hours later when user asked for confirmation
   - Would have been caught earlier with explicit record inspection

3. **Explicit flag documentation**
   - `--regenerate-briefs` flag is correct but needs to be paired with clean directory
   - Consider adding startup check: "If --regenerate-briefs set, warn if old data exists"

---

## Next Steps

✅ Llama Phase 2 restarted with correct design  
✅ Data being generated with brief_instance and solver_draw fields  
✅ Seed calculation verified correct  

Monitor:
- Check record count growing steadily
- Sample records periodically to ensure 4 brief instances being used
- Expected: 16,800 total records, ~60 hours to completion

When complete:
1. Verify all 300 questions have 4 brief instances for stochastic conditions
2. Run variance decomposition analysis
3. Compare Qwen vs Llama categorical structure
4. Update paper Results section
