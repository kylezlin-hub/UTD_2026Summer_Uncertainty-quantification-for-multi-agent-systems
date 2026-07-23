
My Research Question:
Can adaptive influence balancing improve debate quality?
Can minority viewpoints be preserved long enough to improve final decisions?
Can process interventions improve reliability and calibration?
## Here are my recommendationed experiments:
### Experimental Setting 1: Baseline
Original paper setup
Fixed debate protocol
No intervention
Reference point for comparison
### Experimental Setting 2:
Penalty-to-Loser
Agents losing influence become more assertive
Require stronger defense of their position
Encourage persistence of alternative viewpoints
### Experimental Setting 3: Minority Protection
Minority agents receive additional reasoning budget
Extra evidence generation
Additional speaking opportunities
Protect potentially valuable dissenting opinions
### Experimental Setting 4: Devil's Advocate
Dominant agents must challenge their own position
Generate counterarguments
Reduce overconfidence and groupthink
Encourage exploration of alternatives


### Experiment Configuration
- **Model**: Qwen2.5:7b-instruct (via Ollama)
- **Hardware**: NVIDIA RTX A6000 
- **Dataset**: 50 questions × 4 agents × 5 rounds 
- 
### Key Findings

✓ **Process metrics successfully detect interaction failures**:
- Dogmatic agent (Q1) → Highest influence asymmetry (0.598)
- Low engagement (Q5) → Zero engagement (0.000)
- Dominant agent (Q3) → Highest engagement variation (0.175)

✓ **Metrics match paper's qualitative patterns**:
- Each metric responds to its corresponding pathology
- Direction of effects matches Table 3 expectations
- Magnitudes smaller (7B vs GPT-4o/72B) but patterns clear

✓ **Qwen2.5-7B works as LLM judge**:
- 160/160 successful structured JSON responses
- Zero parsing failures
- Coherent and justified quality scores

### Metric Ranges (Likert Questions)

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Engagement | 0.000 | 0.175 | 0.045 |
| Responsiveness | 0.000 | 0.106 | 0.044 |
| Influence Asymmetry | 0.000 | 0.598 | 0.150 |
| Balance | 0.000 | 1.000 | 0.500 |

---

## Comparison to Paper

### Table 3 Analog: Construct Validity

| Pathology | Paper Δ | Our Δ | Match? |
|-----------|---------|-------|--------|
| Dogmatism → Engagement | -0.81 | -0.059 | ✓ Direction |
| Dogmatism → Influence | +0.83 | +0.552 | ✓✓ Strong |
| Low Engage → Engagement | Large drop | -0.175 to 0.000 | ✓✓ Strong |

**Conclusion**: Qualitative patterns match; quantitative magnitudes smaller due to model size and synthetic data.

---

## File Structure

```
C:\Proj1\
├── working_pipeline.py              # Main pipeline (fixed)
├── synthetic_debates.xlsx           # Test dataset
├── analyze_results.py              # Analysis script
├── visualize_results.py            # Visualization
├── FINAL_RESULTS.md                # Complete results doc
├── PAPER_COMPARISON.md             # Tables 1-3 comparison
├── READ.md                         # This file
├── CLAUDE.md                       # Technical documentation
└── diagnostic_metric_results/      # All outputs
    ├── synthetic_debates.llm_judgments.csv     (160 rows)
    ├── synthetic_debates.likert.llm.paper.scores.csv  (8 rows)
    ├── diagnostic_metrics_visualization.png
    └── [160 .judge.json cache files]
```

---

## Dependencies

```bash
# Core
pandas
numpy
scipy

# Visualization
matplotlib

# LLM Infrastructure
ollama (with qwen2.5:7b-instruct model)
```
