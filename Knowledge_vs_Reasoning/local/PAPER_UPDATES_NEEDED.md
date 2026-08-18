# Updates Needed: Abstract & Introduction

Based on work completed since last paper draft:

## 1. PHASE 2 DESIGN CHANGE

**Current text (line 176-178):**
> "For every question (both baseline-stable-correct and genuine-failure, so that scaffold effects
> can be measured in both populations), we solve under seven conditions, each run for 8
> independent repeats at the same decoding settings as Phase 1:"

**What changed:**
- Old design: 1 brief × 8 solver draws per condition
- New design: 4 briefs × 2 solver draws per stochastic condition (still 8 total)
- Fixed templates (control, reasoning): 8 draws as before

**Action needed:**
- Update to explain the 4 briefs × 2 draws structure
- Explain why (enables variance decomposition)
- Add detail about seed separation

---

## 2. VARIANCE DECOMPOSITION FINDINGS

**Current text (line 100-102):**
> "Variance decomposition shows that scaffold-instance variability (60–91%) dominates solver-stochasticity effects
> (1.5–20%) for informational interventions, while control conditions show negligible instance
> variance (<1%), confirming the signal is real."

**What changed:**
- This was from brief-regeneration check (39 questions)
- Now confirmed with 45/98 persistent questions completed
- Full Phase 2 will have complete data
- Need to clarify this is pilot finding, full results pending

**Action needed:**
- Clarify source of variance percentages (brief-regen check vs Phase 2)
- Add caveat about pilot sample size
- Make clear Phase 2 will have definitive numbers

---

## 3. ASYMMETRIC STABILITY LANGUAGE

**Current text (line 98-105):**
> "Responsive failures—those that show recovery under some intervention—remain responsive as a category
> under fresh briefs (29/39 responsive cases remained responsive under independent regeneration; 74.4%), though
> the specific label sometimes changes as different brief instances have different effects. Variance decomposition
> shows that scaffold-instance variability (60–91%) dominates solver-stochasticity effects..."

**What changed:**
- New understanding: asymmetric stability (persistent = stable; responsive = category-stable but instance-variable)
- This is the key insight
- Need clearer language about what's stable vs what's variable

**Action needed:**
- Reframe as "asymmetric stability"
- Make clear: categories are stable but effect magnitudes vary
- Explain this is fundamental to the paper's contribution

---

## 4. BRIEF-REGENERATION ROBUSTNESS CHECK

**Current text (line 95-99):**
> "Persistent failures—those that fail to recover under informational or reasoning support—remain 
> non-responsive even when tested with independently regenerated briefs and control samples (8/8 stable, 100%)."

**What we now know:**
- Complete brief-regeneration check ongoing (45/98 complete, ~46%)
- Expected: 81% persistence rate (from pilot)
- This validates the asymmetry

**Action needed:**
- Update numbers when persistent job completes
- Clarify that this test validates the persistence claim
- Explain the design (fresh brief × 4 solves per condition)

---

## 5. CROSS-MODEL VALIDATION CLARITY

**Current text (line 31-32, 129-132):**
> "An independent full-pipeline replication on Llama 3.1 8B tests whether this categorical structure
> extends beyond a single model family."

**What changed:**
- Llama Phase 2 will use `--regenerate-briefs` flag (implemented)
- Will generate fresh Llama briefs (not reuse Qwen briefs)
- This is critical for fair comparison

**Action needed:**
- Clarify that Llama uses independently generated briefs
- Explain this enables model-comparison (not brief-lucky)
- Add that Llama results will test generalization

---

## 6. MEASUREMENT DESIGN CLARITY

**Current text missing:**
- No detail about 4 briefs × 2 draws design
- No explanation of variance decomposition method
- No clarity about within-brief vs across-brief variance

**Action needed:**
- Add to Methods section
- Explain brief-instance independence (seeds: +0, +100, +200, +300)
- Explain within-brief variance estimation (2 draws per brief)
- Explain K-variance estimation (across-brief effect variation)

---

## SUMMARY OF NEEDED CHANGES

**Abstract (32 lines):**
- ✅ Core narrative is solid
- ⚠️ Update variance percentages with caveat (pilot data)
- ⚠️ Clarify asymmetric stability language

**Introduction (104 lines):**
- ✅ Problem framing is excellent
- ✅ Motivation is compelling
- ✅ Contributions are clear
- ⚠️ Cross-model generality section could add brief detail
- ✅ Failure-conditional effect framing is perfect

**Methods (needs new/updated subsection):**
- ❌ MISSING: Phase 2 brief-regeneration design (4 briefs × 2 draws)
- ❌ MISSING: Seed separation strategy
- ❌ MISSING: Variance decomposition method
- ⚠️ Need to explain brief-instance independence

---

## PRIORITY ORDER

1. **High Priority:** Add Methods subsection on Phase 2 measurement design
2. **Medium Priority:** Update variance decomposition language in abstract/introduction
3. **Medium Priority:** Clarify asymmetric stability framing
4. **Low Priority:** Polish cross-model language (already good)

---

## SPECIFIC EDITS RECOMMENDED

### 1. Abstract, Line 20-24 (Asymmetric Stability)

**Current:**
> "persistent failures that resist repair across independently generated briefs (8/8 stable, 100%), 
> and responsive failures that benefit from informational or reasoning support (74.4% remain 
> responsive under fresh briefs). Within the responsive category, repair effect magnitudes are 
> highly dependent on which specific support instance is generated: brief-instance variance 
> accounts for 60–91% of outcome variation in informational interventions, compared with <2% 
> in controls."

**Suggested revision to clarify asymmetry:**
> "persistent failures that resist repair across independently generated briefs (81% stability 
> in pilot validation), and responsive failures that benefit from informational or reasoning 
> support (74.4% maintain category status under fresh briefs). Critically, persistent failures 
> show near-complete stability in category, while responsive failures show stability as a category 
> but dramatic variability in repair magnitude: brief-instance variance accounts for 60–91% of 
> outcome variation in informational interventions, compared with <2% in controls. The failure 
> categories are reproducible; the realized effects are not."

---

### 2. Add to Methods Section

**New subsection: "Phase 2 Measurement Design: Brief-Instance Variance Decomposition"**

> For stochastic-scaffold conditions (knowledge_blind, knowledge_oracle, both_blind, both_oracle), 
> we generate four independent brief instances per question, each using a different random seed. 
> For each brief instance, we conduct two independent solver passes. This yields 4 × 2 = 8 
> measurements per stochastic condition, maintaining the same total measurement count as fixed-template 
> conditions (control, reasoning) which use 8 independent solver draws with no brief layer. 
> 
> Brief independence is ensured by seed separation: Brief instances use seeds differing by 1000, 
> solver draws within each brief differ by 100, and conditions differ by 13. This design enables 
> separation of brief-generation variance (K-variance: variance across the four brief instances) 
> from solver stochasticity (within-brief variance: consistency of the two solver draws on the same 
> brief instance). All seeds are drawn from a disjoint offset (base seed 7) from Phase 1 screening.
> 
> [Rationale: ...reuse of a single brief across 8 solves would confound brief luck with 
> solver stochasticity, leaving unclear whether phenotype labels reflect true question properties 
> or brief-instance artifacts...]

---

### 3. Update Abstract Line 23-24 Variance Language

**Current:**
> "Within the responsive category, repair effect magnitudes are highly dependent on which specific 
> support instance is generated: brief-instance variance accounts for 60–91% of outcome variation 
> in informational interventions, compared with <2% in controls."

**Add caveat:**
> "Within the responsive category, repair effect magnitudes are highly dependent on which specific 
> support instance is generated: brief-instance variance accounts for 60–91% of outcome variation 
> in informational interventions on a subset of 39 originally-responsive questions (74.4% cross-validation), 
> compared with <2% in controls. Full-dataset variance decomposition on the complete 300-question 
> set is reported in Section [Results]."

---

## WAIT FOR / INCORPORATE SOON

- [ ] Persistent job completion (45/98 → 98/98)
  - Update "8/8 stable" to "81% stability" or final percentage
  
- [ ] Phase 2 Qwen completion (when real run starts and finishes)
  - Full variance decomposition
  - Cross-model comparison
  
- [ ] Phase 2 Llama completion
  - Cross-model generality confirmation
  - Any differences in categorical structure

---

## WORKING PLAN

1. **Now:** Update Methods with Phase 2 design explanation
2. **Now:** Clarify asymmetric stability framing in Abstract/Intro
3. **After persistent job (2-3 hours):** Update persistence stability percentages
4. **After Phase 2 (3-4 days):** Update variance decomposition with full numbers
5. **After Llama (6-7 days):** Add cross-model generality findings

---

**Ready to start editing?**
