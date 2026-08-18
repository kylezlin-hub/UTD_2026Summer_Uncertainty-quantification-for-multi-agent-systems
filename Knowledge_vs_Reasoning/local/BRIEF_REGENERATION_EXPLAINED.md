# Llama 8B Phase 2 Brief Regeneration: Complete Explanation

## TL;DR

The new Llama Phase 2 design uses `--regenerate-briefs` flag to generate **independent briefs for each model**. This solves the fundamental problem: *old design reused Qwen's briefs for Llama, making cross-model comparison impossible*.

```
Old: Qwen briefs → Qwen solves + Llama solves (❌ Can't distinguish)
New: Qwen briefs → Qwen solves
     Llama briefs → Llama solves           (✓ Fair comparison)
```

---

## How It's Designed

### The `--regenerate-briefs` Flag

**Purpose:** Override brief caching so each model gets fresh briefs

**Location in code:**
- **Defined:** `generate_interventions.py` line ~806 (argparse)
- **Used:** `generate_interventions.py` line 564 (cache check) + line 585 (generation check)
- **Called:** `run_llama_replication.py` line 75 (Phase 2 command)

**Behavior:**
```python
ap.add_argument("--regenerate-briefs", action="store_true",
                help="ignore cached briefs and generate fresh ones")

# Then in run_generation():
if briefs_path.exists() and not args.regenerate_briefs:
    # Load from cache (fast)
    for line in briefs_path.read_text():
        briefs[q_no] = json.loads(line)

elif args.regenerate_briefs:
    # Skip cache entirely, force generation
    print("[Briefs] --regenerate-briefs is set; forcing fresh generation")

# Later, for each question:
if q.question_no not in briefs or args.regenerate_briefs:
    # Generate (happens ALWAYS for Llama because flag is set)
    blind_brief = generate_brief(llm, q, ..., oracle=False)
    oracle_brief = generate_brief(llm, q, ..., oracle=True)
    # Write to LLAMA-SPECIFIC directory
    append_jsonl(briefs_path, rec)  # interventions_llama/
```

### The Critical Line (585)

```python
if q.question_no not in briefs or args.regenerate_briefs:
```

This condition is TRUE when:
1. **Question not cached:** First time seeing this question in this run
2. **Regenerate flag set:** Force new brief regardless of cache

**For Qwen (no flag):** Condition 1 only → Generate only missing briefs (fast)
**For Llama (--flag):** Conditions 1 AND 2 → All briefs regenerated (thorough)

---

## How It's Called

### In `run_llama_replication.py` Phase 2

```python
def run_phase2():
    """Generate interventions on 300 questions with Llama."""
    
    llama_runs_dir = LLAMA_OUT_DIR / "runs"
    llama_interv_dir = HERE / "interventions_llama"  # ← Different directory!
    
    cmd = [
        PYTHON, str(HERE / "generate_interventions.py"),
        "--backend", "ollama",
        "--model-id", "llama3.1:8b",           # ← Different model
        "--datasets", "mmlu", "mmlu-pro", "gpqa",
        "--selection", "stratified",
        "--repeats", "8",
        "--seed", "7",
        "--runs-dir", str(llama_runs_dir),
        "--out-dir", str(llama_interv_dir),   # ← Different output dir
        "--regenerate-briefs",                  # ← KEY: Force fresh briefs
    ]
    result = subprocess.run(cmd, cwd=str(HERE))
```

### Execution Flow

```
$ python run_llama_replication.py --phase 2

↓ Calls subprocess:

$ python generate_interventions.py \
    --model-id llama3.1:8b \
    --out-dir interventions_llama/ \
    --regenerate-briefs

↓ In generate_interventions.py:

run_generation(args):
    OUT_DIR = interventions_llama/
    briefs_path = interventions_llama/knowledge_briefs.jsonl
    
    if briefs_path.exists() and not args.regenerate_briefs:
        → Condition 1: briefs_path.exists() = True (maybe)
        → Condition 2: NOT args.regenerate_briefs = False (flag is set!)
        → Overall: False AND False = False → Skip load
    
    elif args.regenerate_briefs:
        → Condition: args.regenerate_briefs = True ✓
        → Execute: Print "[Briefs] --regenerate-briefs set..."
        → Effect: briefs dict stays empty
    
    FOR each question:
        if q.question_no NOT in briefs OR args.regenerate_briefs:
            → First condition: briefs dict empty → True ✓
            → OR second condition: flag is set → True ✓
            → Overall: True OR True = True ✓
            → Execute: generate_brief() using Llama model
            → Write to: interventions_llama/knowledge_briefs.jsonl
    
    FOR each condition × repeat:
        Solve using fresh Llama briefs
        Write to: interventions_llama/solve_results.jsonl
```

---

## What's Different from Qwen

### Directory Separation

**Qwen results:**
```
interventions/
├── knowledge_briefs.jsonl      (Qwen-generated, cached for reuse)
├── solve_results.jsonl         (Qwen solves with Qwen briefs)
└── intervention_labels.csv     (Taxonomy from Qwen solves)
```

**Llama results:**
```
interventions_llama/
├── knowledge_briefs.jsonl      (Llama-generated, FRESH, independent)
├── solve_results.jsonl         (Llama solves with Llama briefs)
└── intervention_labels.csv     (Taxonomy from Llama solves)
```

### Flag Difference

| Model | Flag | Cache Behavior | Brief Generation |
|-------|------|---|---|
| **Qwen** | (none) | Use cache if exists | Only missing briefs |
| **Llama** | `--regenerate-briefs` | Ignore cache | All briefs fresh |

### Seed Handling (SAME for both)

Both use `--seed 7`:
```python
seed_blind = args.seed + 0        # = 7
seed_oracle = args.seed + 500     # = 507
seed_solve_rep_i_cond_j = args.seed + 1000*i + 13*j
```

**Important:** Same seeds, DIFFERENT models → Different briefs
- Ensures fair RNG initialization
- Not about reproducibility (model makes text different)
- About fair sampling of brief quality

---

## The Two-Level Caching Strategy

### Level 1: Disk Cache (Persistent)

**Purpose:** Persist briefs across runs/interruptions

```python
briefs_path = OUT_DIR / "knowledge_briefs.jsonl"

if briefs_path.exists() and not args.regenerate_briefs:
    # Load: JSON lines (one dict per line)
    for line in briefs_path.read_text(encoding="utf-8").splitlines():
        briefs[d["question_no"]] = d
```

**Files:**
- Qwen: `interventions/knowledge_briefs.jsonl`
- Llama: `interventions_llama/knowledge_briefs.jsonl`

**Control:** `--regenerate-briefs` flag
- False (default) → Read disk cache
- True → Ignore disk cache, generate fresh

### Level 2: In-Memory Cache (Session)

**Purpose:** Avoid regenerating same question multiple times in one run

```python
briefs = {}  # Empty dict at start

for each question:
    if q.question_no not in briefs or args.regenerate_briefs:
        # Generate fresh
        # ...
        briefs[q.question_no] = rec  # ← Cache in memory
    
    # Now use briefs[q.question_no]
    # If same question needed again → Use memory cache (instant!)
```

**Efficiency:**
- First mention of question → Generate (slow)
- Second mention of question (same run) → Use memory cache (instant)

**Safety:**
- If run crashes/interrupts → Restart from beginning
- New run reloads from disk (Level 1)
- No duplicate generation

---

## Resume Safety

### How JSONL Format Enables Safe Resumption

**append_jsonl()** writes one JSON object per line:

```python
def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
```

**Safety mechanism:**
```
First run (interrupted at question 15):
  knowledge_briefs.jsonl has lines for: q1, q2, ..., q15

Resume run (same process):
  ✓ Load existing briefs.jsonl (q1-q15 loaded)
  ✓ Continue from q16 onward
  ✓ New lines appended to file
  ✓ No duplicates (JSONL format allows append-only)

solve_results.jsonl has similar protection:
  done = load_done(results_path)
  for each (q_no, condition, repeat) pair:
      if (q_no, cond, rep) in done:
          skip  # Already solved
      else:
          solve  # New
```

---

## What Happens When You Run Llama Phase 2

### Command
```bash
python run_llama_replication.py --phase 2
```

### Subprocess Call (Inside run_phase2())
```bash
python generate_interventions.py \
    --backend ollama \
    --model-id llama3.1:8b \
    --datasets mmlu mmlu-pro gpqa \
    --selection stratified \
    --repeats 8 \
    --seed 7 \
    --runs-dir data/debate_llama_8b/runs \
    --out-dir interventions_llama/ \
    --regenerate-briefs
```

### What `--regenerate-briefs` Does

1. **Skip disk cache** (even if `interventions_llama/knowledge_briefs.jsonl` exists)
2. **Force fresh generation** for all questions
3. **Use Llama model** to generate briefs (different from Qwen)
4. **Write to different directory** (`interventions_llama/`)
5. **Enable fair comparison** (Qwen briefs ≠ Llama briefs)

### Timeline
```
Time 0:00   Start Llama Phase 2
Time 0:01   Load 300 questions (MMLU 100, MMLU-Pro 100, GPQA 100)
Time 0:05   Start generating briefs (seed 7, 507)
Time 1:00   Finish generating 300 briefs
Time 1:05   Start solving conditions (6 conditions × 8 reps each)
Time 11:00  Finish all solves
Time 11:05  Generate intervention_labels.csv (taxonomy)

Total: ~11 hours for full 300-question run
```

---

## Key Questions Answered

### Q: Won't Llama Phase 2 be slower because of fresh briefs?

**A:** Yes, brief generation adds ~1-2 hours overhead, but:
- **Necessary for validity:** Must control brief instance
- **Acceptable cost:** 1-2 hours vs 11-hour total is small
- **Alternative:** `--limit 50` for quick validation

### Q: Can I run Qwen and Llama in parallel?

**A:** Yes! They use different directories:
- Qwen → `interventions/`
- Llama → `interventions_llama/`

No conflicts. Can run simultaneously on same machine (if enough CPU/GPU).

### Q: What if I want to skip brief regeneration on Llama?

**A:** Don't pass `--regenerate-briefs`:
```python
# In run_llama_replication.py, comment out the flag:
cmd = [
    # ... other args ...
    # "--regenerate-briefs",  ← Comment this out
]
```

Then Llama will:
1. Check if `interventions_llama/knowledge_briefs.jsonl` exists
2. If yes → Load and reuse (fast, but not fair comparison!)
3. If no → Generate fresh (because condition: `not in briefs` is true)

### Q: How do I validate the briefs are truly different?

**A:** Check the files:
```python
import json

# Load Qwen briefs
qwen = {}
for line in open('interventions/knowledge_briefs.jsonl'):
    d = json.loads(line)
    qwen[d['question_no']] = d['brief_blind']

# Load Llama briefs
llama = {}
for line in open('interventions_llama/knowledge_briefs.jsonl'):
    d = json.loads(line)
    llama[d['question_no']] = d['brief_blind']

# Compare
q = list(qwen.keys())[0]
print("Qwen:", qwen[q][:150])
print("Llama:", llama[q][:150])
# Should see clearly different text
```

---

## Summary

| Aspect | Explanation |
|--------|-------------|
| **What** | `--regenerate-briefs` flag forces independent brief generation |
| **Why** | Isolate model effects from brief effects for fair cross-model comparison |
| **Where** | `generate_interventions.py` (lines 564, 585) and `run_llama_replication.py` (line 75) |
| **How** | When flag is set, ignore cached briefs and always generate fresh using current model |
| **Output** | Qwen briefs in `interventions/`, Llama briefs in `interventions_llama/` |
| **Impact** | Enables honest cross-model validation of categorical structure (persistent vs. responsive) |

The design is elegant: same procedural generator, different models → Independent brief instances → Fair comparison.
