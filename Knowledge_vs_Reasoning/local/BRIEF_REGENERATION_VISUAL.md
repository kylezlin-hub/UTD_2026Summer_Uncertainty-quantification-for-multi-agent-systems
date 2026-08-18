# Brief Regeneration Flow: Visual Guide

## 1. HIGH LEVEL: WHAT'S DIFFERENT

### Old Design (BROKEN)
```
Phase 2 (Qwen):
  generate_interventions.py (default, uses cache)
    ├─ Generate: blind_brief, oracle_brief
    ├─ Write to: interventions/knowledge_briefs.jsonl
    └─ Solve: Qwen reads these briefs
    
Phase 2 (Llama):
  generate_interventions.py (NO flag, uses same cache)
    ├─ Load: SAME briefs from interventions/
    └─ Solve: Llama reads QWEN's briefs ❌ WRONG
```

**Problem:** Can't tell if Llama behaves differently because:
- Llama is different? OR
- These particular briefs just work differently for Llama?

### New Design (FIXED)
```
Phase 2 (Qwen):
  generate_interventions.py (default)
    ├─ Generate: blind_brief, oracle_brief (Qwen-model)
    ├─ Write to: interventions/knowledge_briefs.jsonl
    └─ Solve: 8 reps × 6 conditions → interventions/solve_results.jsonl
    
Phase 2 (Llama):
  generate_interventions.py --regenerate-briefs
    ├─ Generate: blind_brief, oracle_brief (Llama-model, FRESH)
    ├─ Write to: interventions_llama/knowledge_briefs.jsonl (different dir!)
    └─ Solve: 8 reps × 6 conditions → interventions_llama/solve_results.jsonl
```

**Solution:** Both models get independent briefs from same generator → Fair comparison

---

## 2. CODE FLOW: HOW --regenerate-briefs WORKS

### Decision Tree (Lines 564–570)

```
START: run_generation(args)
  |
  ├─ briefs_path = OUT_DIR / "knowledge_briefs.jsonl"
  |
  ├─ briefs = {}  (empty dict)
  |
  └─ if briefs_path.exists() AND NOT args.regenerate_briefs:
     │
     ├─ YES ──→ Load from disk (CACHE)
     │           for line in briefs_path:
     │             briefs[q_no] = {brief data}
     │           print("[Briefs] Loaded N cached briefs")
     │
     └─ NO ──→ elif args.regenerate_briefs:
                 print("[Briefs] --regenerate-briefs set; forcing fresh generation")
                 (briefs dict stays empty)
```

### Brief Generation (Lines 585–594)

```
FOR each question:
  |
  └─ if q.question_no NOT IN briefs  OR  args.regenerate_briefs:
     │
     ├─ YES ──→ GENERATE FRESH
     │           1. generate_brief(llm, ..., oracle=False)
     │              → seed = args.seed (e.g., 7)
     │
     │           2. generate_brief(llm, ..., oracle=True)
     │              → seed = args.seed + 500 (e.g., 507)
     │
     │           3. Store in rec = {question_no, brief_blind, brief_oracle, ...}
     │
     │           4. append_jsonl(briefs_path, rec)
     │              → Write to interventions/ OR interventions_llama/
     │
     │           5. briefs[question_no] = rec
     │              → Cache in memory for this run
     │
     └─ NO ──→ Use existing cached brief
                 brief_texts = {
                   "blind": briefs[q.question_no]["brief_blind"],
                   "oracle": briefs[q.question_no]["brief_oracle"]
                 }
```

### Key: Two Levels of Caching

```
LEVEL 1: Disk Cache (persistent)
  briefs_path = interventions/knowledge_briefs.jsonl  (Qwen)
  briefs_path = interventions_llama/knowledge_briefs.jsonl  (Llama)
  
  Controlled by: --regenerate-briefs flag
  ├─ False (default) → Read from disk if exists
  └─ True (new) → Ignore disk, generate fresh

LEVEL 2: In-Memory Cache (session)
  briefs = {}  (dict in Python)
  
  Purpose: Avoid regenerating same question multiple times in one run
  ├─ Question A processed → briefs["A"] loaded into memory
  ├─ Question A needed again → Use briefs["A"] from memory (instant)
  └─ On resume after crash → Reload from disk (level 1)
```

---

## 3. DIRECTORY STRUCTURE

### After Qwen Phase 2 completes:
```
Knowledge_vs_Reasoning/local/
├── interventions/
│   ├── knowledge_briefs.jsonl          (Qwen-generated briefs)
│   │   └─ Lines: {"question_no": "10790", "brief_blind": "...", ...}
│   │
│   ├── solve_results.jsonl             (Qwen solve outputs)
│   │   └─ Lines: {"question_no": "10790", "condition": "knowledge_blind", "correct": true, ...}
│   │
│   └── intervention_labels.csv         (Taxonomy)
│       └─ label,label_confidence,p_knowledge_blind,...
```

### After Llama Phase 2 completes (with --regenerate-briefs):
```
Knowledge_vs_Reasoning/local/
├── interventions_llama/
│   ├── knowledge_briefs.jsonl          (Llama-generated briefs, DIFFERENT content)
│   │   └─ Lines: {"question_no": "10790", "brief_blind": "...", ...}
│   │
│   ├── solve_results.jsonl             (Llama solve outputs)
│   │   └─ Lines: {"question_no": "10790", "condition": "knowledge_blind", "correct": ?, ...}
│   │
│   └── intervention_labels.csv         (Taxonomy from Llama)
│       └─ May differ from Qwen!
```

**Critical:** Separate directories mean:
- Qwen briefs never contaminate Llama
- Easy to compare: `diff interventions/ interventions_llama/`
- Safe to run in parallel (different processes)

---

## 4. EXECUTION SCENARIOS

### Scenario A: Fresh Llama Run

```
$ python run_llama_replication.py --phase 2

Invokes:
  python generate_interventions.py \
    --out-dir interventions_llama/ \
    --regenerate-briefs \
    ... (other flags)

run_generation():
  briefs_path = interventions_llama/knowledge_briefs.jsonl
  
  [CHECK] Does it exist? 
    └─ NO (first time) → Skip the if-exists-load-cache block
  
  [CHECK] --regenerate-briefs set?
    └─ YES → Print "[Briefs] --regenerate-briefs is set; forcing fresh generation"
  
  [FOR EACH QUESTION]
    [CHECK] q.question_no NOT in briefs OR regenerate_briefs?
      └─ YES (always on first run) → GENERATE fresh briefs using Llama model
      
  [FOR EACH CONDITION × REPEAT]
    [CHECK] (q_no, cond, rep) already in solve_results.jsonl?
      └─ NO (first time) → SOLVE and write result

RESULT:
  interventions_llama/knowledge_briefs.jsonl (fresh Llama briefs)
  interventions_llama/solve_results.jsonl (Llama solves with fresh briefs)
```

### Scenario B: Resume After Interruption (Same Run)

```
$ python run_llama_replication.py --phase 2   (AGAIN)

run_generation():
  briefs_path = interventions_llama/knowledge_briefs.jsonl
  
  [CHECK] Does it exist?
    └─ YES (from before) → Try to load
  
  [CHECK] --regenerate-briefs set?
    └─ YES (still set!) → SKIP the load, force regeneration
    → Print "[Briefs] --regenerate-briefs is set; forcing fresh generation"
  
  [FOR EACH QUESTION]
    [CHECK] q.question_no NOT in briefs (empty dict) OR regenerate_briefs?
      └─ YES (briefs dict empty) → Process for generation
      
      [CHECK] Does this question's brief already exist in file?
        LLAMA MODEL generates fresh brief for this question
        append_jsonl() appends to the same file
        
        ⚠️ IMPORTANT: JSONL is append-only, so you get:
        {Line 1} question_no: 10790, brief_blind: "...", ...
        {Line 2} question_no: 10790, brief_blind: "...", ...  (DUPLICATE?)
        
        SOLUTION: The next step loads into briefs dict and uses in-memory cache
  
  [FOR EACH CONDITION × REPEAT]
    [CHECK] (q_no, cond, rep) already in solve_results.jsonl?
      └─ YES (from before) → SKIP (no duplicates)
      └─ NO (new) → SOLVE and write result

RESULT:
  - Solves resume from where they left off
  - No duplicate solve results (done() tracking prevents)
  - Brief file may have duplicates but in-memory dict is de-duped
```

**Note:** If you want to truly resume without regenerating briefs:
```bash
# Omit --regenerate-briefs on resume
python generate_interventions.py \
  --out-dir interventions_llama/ \
  ... (no --regenerate-briefs flag)

# Now it will:
# 1. Load existing briefs from file
# 2. Resume solves only
# 3. Faster!
```

---

## 5. DETAILED COMPARISON: First vs. Later Runs

### First Run (Qwen)
```
generate_interventions.py
  --out-dir interventions/
  (NO --regenerate-briefs)

Logic:
  briefs_path = interventions/knowledge_briefs.jsonl
  if exists(briefs_path) and not regenerate_briefs:
    → False (file doesn't exist yet)
  elif regenerate_briefs:
    → False (flag not set)
  
  So briefs dict stays empty → All questions get generated
  
  For each question in first run:
    ✓ Generate blind and oracle briefs (Qwen model)
    ✓ Write to interventions/knowledge_briefs.jsonl
    ✓ Solve 6 conditions × 8 repeats
    ✓ Write to interventions/solve_results.jsonl
```

### First Run (Llama)
```
generate_interventions.py
  --out-dir interventions_llama/
  --regenerate-briefs

Logic:
  briefs_path = interventions_llama/knowledge_briefs.jsonl
  if exists(briefs_path) and not regenerate_briefs:
    → False (flag is True, skips the whole condition)
  elif regenerate_briefs:
    → True → Print "[Briefs] --regenerate-briefs is set..."
  
  So briefs dict stays empty → All questions get generated (FRESH!)
  
  For each question in first run:
    ✓ Generate blind and oracle briefs (Llama model)
    ✓ Write to interventions_llama/knowledge_briefs.jsonl
    ✓ Solve 6 conditions × 8 repeats
    ✓ Write to interventions_llama/solve_results.jsonl
```

**Result:** Qwen and Llama have different briefs in different directories!

---

## 6. THE CRITICAL SECTION (Why This Matters)

Line 585 is the KEY:
```python
if q.question_no not in briefs or args.regenerate_briefs:
    # Generate fresh briefs
```

This condition is TRUE when:
1. Question has never been seen in this run → Generate
2. `--regenerate-briefs` is set → Regenerate (even if cached)

The `OR args.regenerate_briefs` is what makes the difference:
- **Qwen (no flag):** Only case 1 is true → Generate only missing briefs
- **Llama (--regenerate-briefs):** Both cases true → All briefs regenerated

---

## 7. EXAMPLE: Question 10790

### Qwen Run:
```
Question: "In a circuit with resistors..."
Correct answer: "C"

Brief generation:
  Seed 7 → Qwen generates: 
    "Ohm's law relates voltage, current, and resistance. 
     Series resistances add linearly. R_total = R1 + R2 + ..."
  
  Seed 507 → Qwen generates (oracle, sees answer C):
    "The circuit behavior depends on series/parallel configuration.
     When resistors are in series, voltage drops across each...
     This results in answer C."

Write to: interventions/knowledge_briefs.jsonl
```

### Llama Run:
```
Question: "In a circuit with resistors..."  (SAME question)
Correct answer: "C"

Brief generation (--regenerate-briefs set):
  Seed 7 → Llama generates:
    "Electric potential relationships. Consider how current flows
     through a linear arrangement of components. Each component
     affects the overall resistance proportionally."
  
  Seed 507 → Llama generates (oracle, sees answer C):
    "Circuit analysis requires understanding of series configuration.
     When passive elements connect sequentially, their individual
     impedances combine. The correct answer reflects..."

Write to: interventions_llama/knowledge_briefs.jsonl
```

### Key Insight:
- Same seeds (7, 507)
- Same question
- **Different models → Different briefs**
- Both briefs stored independently

When we compare results:
```
Qwen with Qwen-brief: Recovery = 2/8 for "knowledge_blind"
Llama with Llama-brief: Recovery = ? for "knowledge_blind"

Difference = Model effect + Brief effect + Interaction

But we can aggregate across 300 questions to estimate model effect,
because the brief effects are independently distributed.
```

---

## Summary Table

| Aspect | Qwen | Llama |
|--------|------|-------|
| **Flag** | (none, default) | `--regenerate-briefs` |
| **Output dir** | `interventions/` | `interventions_llama/` |
| **Brief generation** | On first run | Always fresh (flag overrides cache) |
| **Brief file** | `interventions/knowledge_briefs.jsonl` | `interventions_llama/knowledge_briefs.jsonl` |
| **Brief instances** | One per question (cached) | One per question per run (independent) |
| **Model** | Qwen2.5-7B | Llama 3.1 8B |
| **Research value** | Baseline failure structure | Cross-model robustness of categorical structure |

The `--regenerate-briefs` flag is the control knob that makes Llama independent from Qwen while using the same methodological framework.
