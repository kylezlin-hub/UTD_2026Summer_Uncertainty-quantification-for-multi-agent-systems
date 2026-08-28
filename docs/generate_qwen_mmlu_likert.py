"""BLUEPRINT (provisional): MMLU debates with a numeric 5-point Likert stance.

STATUS: pending author confirmation of the MMLU stance representation.
Do NOT treat this as validated. It encodes ONE defensible reading of paper
§A.3.5 ("agents report their stance on a five-point Likert scale ... mapped to
x in {-2,-1,0,1,2}") applied to multiple-choice questions. Finalize the framing
once the authors reply to the clarification email.

WHY THIS EXISTS
---------------
We established that our metric code is byte-identical to the authors' reference
(authors_code.py). The divergence from the paper is UPSTREAM: our MMLU debates
emit answer LETTERS, which forces the categorical metric branch (binary
engagement). The paper's §A.3.5 describes a numeric Likert stance, which routes
to paper_likert_metrics (magnitude engagement, Eq. 4). To reproduce the paper we
must GENERATE MMLU debates that emit a numeric stance, then score in likert mode.

THE DESIGN DECISION (the part that needs author confirmation)
-------------------------------------------------------------
A 5-point *agreement* Likert is about a single proposition, but MMLU is N-way.
The cleanest reconciliation that (a) yields an ordinal stance and (b) still
measures correctness is a VERIFY-THE-ANSWER framing:

  For each question, pair it with ONE candidate answer and ask agents to rate
  agreement (1-5) that the candidate is correct. Debate updates the agreement.

    stance x = agreement mapped {1..5} -> {-2..-1..0..1..2}   (magnitude-valid)
    |x_t - x_{t-1}| is now meaningful  -> Eq. 4 engagement applies directly

  To keep accuracy non-trivial (not "always agree"), balance the candidates:
    - half the questions: candidate = the GOLD answer   (correct => agree)
    - half the questions: candidate = a sampled DISTRACTOR (correct => disagree)
  Debate "decision": accept candidate iff final mean stance > 0.
  Correct iff (accepted == candidate_is_gold).

  NOTE: this converts SELECTION into VERIFICATION. It is a faithful Likert
  representation but a different task shape than letter-MCQ. If the authors say
  MMLU used letters+categorical instead, discard this file and keep the existing
  categorical pipeline (which already matches their reference).

WIRING (once framing is confirmed)
----------------------------------
The existing pipeline already has a Likert path used for "subjective" debates:
  - qwen_initial_messages / qwen_update_messages (subjective branch) elicit
    "Answer: <1/2/3/4/5>" on its own line -- exactly §A.3.5's format.
  - build_matrices(stance_mode="likert") -> extract_likert_stance -> x in {-2..2}
  - paper_likert_metrics -> magnitude engagement.
So the cleanest implementation reuses that path: emit reframed MMLU items as
Likert-stance debates and force stance_mode="likert", rather than writing new
metric code. This module only needs to (1) build candidate-paired questions and
(2) supply agreement-style prompts; scoring is unchanged authors' code.

This file gives the loader + prompt builders as a starting point. The exact
score-time wiring (routing reframed items through the likert branch and defining
the accept/correct rule) is left as TODOs to finalize post-confirmation.
"""

from __future__ import annotations

import random

import qwen_methodology_code as qmc

# Fixed sample/candidate seed so the question+candidate set is identical across
# debate seeds (paper: question sets held fixed across runs).
_PAIR_SEED = 20240101


def build_likert_mmlu_items(rows: list[dict], limit: int, pair_seed: int = _PAIR_SEED):
    """Pair each MMLU row with a candidate answer for the verify-the-answer framing.

    Returns a list of dicts with the question text, the proposed candidate, and
    whether that candidate is the gold answer. Balanced ~50/50 gold vs distractor
    so 'agree' is not trivially always-correct.

    PROVISIONAL: confirm with authors before relying on this framing.
    """
    rng = random.Random(pair_seed)
    rng.shuffle(rows)
    items = []
    for idx, row in enumerate(rows):
        if len(items) >= limit:
            break
        question_text = qmc.clean_text(row.get("question"))
        choices = [qmc.clean_text(c) for c in (row.get("choices") or []) if qmc.clean_text(c)]
        if not question_text or len(choices) < 2:
            continue
        try:
            gold_idx = int(row.get("answer"))
            gold = choices[gold_idx]
        except (TypeError, ValueError, IndexError):
            continue
        # Balance: even index -> propose gold; odd -> propose a distractor.
        propose_gold = (idx % 2 == 0)
        if propose_gold:
            candidate = gold
        else:
            distractors = [c for i, c in enumerate(choices) if i != gold_idx]
            candidate = rng.choice(distractors) if distractors else gold
        items.append({
            "question_no": f"mmlu-likert-{idx + 1:05d}",
            "question": question_text,
            "candidate": candidate,
            "candidate_is_gold": propose_gold,
            "gold": gold,
            "subject": qmc.clean_text(row.get("subject")),
        })
    return items


# --- Agreement-style prompts (emit a 1-5 stance on its own line, per A.3.5) ---

_LIKERT_SCALE = (
    "1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree"
)


def likert_initial_user(item: dict) -> str:
    """Round-1 independent agreement rating for the proposed answer."""
    return (
        f"Question:\n{item['question']}\n\n"
        f"Proposed answer: {item['candidate']}\n\n"
        "Independently judge whether the proposed answer is correct. Report your "
        f"agreement on a five-point scale ({_LIKERT_SCALE}).\n"
        "Use exactly this format:\n"
        "Answer: <1/2/3/4/5>\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <brief justification>"
    )


def likert_update_user(item: dict, own_prev: dict, other_lines: list[str]) -> str:
    """Later-round update after seeing peers' agreement ratings + reasons."""
    return (
        f"Question:\n{item['question']}\n\n"
        f"Proposed answer: {item['candidate']}\n\n"
        f"Your previous agreement was {own_prev['answer']} with explanation: "
        f"{own_prev['response']}\n\n"
        "Other agents' previous-round ratings:\n" + "\n".join(other_lines) + "\n\n"
        "Consider their arguments and update your agreement only if warranted. "
        f"Rate agreement on the five-point scale ({_LIKERT_SCALE}).\n"
        "Use exactly this format:\n"
        "Answer: <1/2/3/4/5>\n"
        "Confidence: <number from 0 to 1>\n"
        "Explanation: <brief justification grounded in the debate>"
    )


# --- Accuracy rule for the verify-the-answer framing --------------------------

def debate_is_correct(final_mean_stance: float, candidate_is_gold: bool) -> bool:
    """Accept the candidate iff final mean stance > 0 (agree); correct iff the
    accept/reject decision matches whether the candidate was actually gold."""
    accepted = final_mean_stance > 0.0
    return accepted == candidate_is_gold


# --- TODO (finalize AFTER author confirmation) --------------------------------
# 1. Confirm the MMLU->Likert framing with the authors (verify-the-answer vs
#    a different mapping vs letters+categorical). If they used letters+categorical,
#    delete this file -- the existing pipeline already matches their reference.
# 2. Wire these items/prompts into the generation loop so agents emit the 1-5
#    stance, and force scoring with stance_mode="likert" (NOT the objective->
#    categorical auto-selection at qwen_methodology_code.py:1623). Simplest route:
#    tag these debates so score_mixed_debates treats them as likert, reusing
#    extract_likert_stance + paper_likert_metrics unchanged.
# 3. Record candidate / candidate_is_gold per row and compute correctness via
#    debate_is_correct(mean_final_stance_mu_T, candidate_is_gold).
# 4. Sanity check: round-1 stance distribution should span 1-5 (not all 3s), and
#    accuracy should be well above chance if the framing is sound.


if __name__ == "__main__":
    print(__doc__)
    print("This is a provisional blueprint; finalize wiring after author confirmation.")
