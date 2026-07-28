"""
Pilot: can same-model debate dynamics separate knowledge- vs reasoning-limited failures?

Cheap go/no-go on data already in hand (New/baseline_v2_*, 3 seeds each).
No new generation, no interventions yet. Proxy split:

  * UNANIMOUS-WRONG at R1  -> shared blindspot  -> KNOWLEDGE-limitation signature
  * DISAGREEMENT at R1     -> stochastic divergence -> REASONING-limitation signature

Thesis: debate repairs stochastic reasoning errors but not shared knowledge gaps.
Prediction: among initially-WRONG questions, recovery (wrong->correct by final round)
is much higher for the disagreement group than the unanimous-wrong group; and the
debate accuracy gain over single-shot / self-consistency concentrates on disagreement.
"""
import re
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import mannwhitneyu, fisher_exact

AGENTS = ['Agent1', 'Agent2', 'Agent3']
DATASETS = {
    'MMLU-Pro': ['baseline_v2_mmlu-pro_s7.xlsx', 'baseline_v2_mmlu-pro_s17.xlsx', 'baseline_v2_mmlu-pro_s42.xlsx'],
    'GPQA':     ['baseline_v2_gpqa_s7.xlsx', 'baseline_v2_gpqa_s17.xlsx', 'baseline_v2_gpqa_s42.xlsx'],
}


def norm(x):
    return None if pd.isna(x) else str(x).strip().upper()


def plurality(ans):
    present = [a for a in ans if a]
    if not present:
        return None
    return Counter(present).most_common(1)[0][0]


def per_question_features(path):
    df = pd.read_excel(path, sheet_name='Debate_Traces')
    rounds = sorted({int(m.group(1)) for c in df.columns if (m := re.match(r'R(\d+) Agent1 Answer', c))})
    rows = []
    for _, r in df.iterrows():
        correct = norm(r.get('Correct Answer'))
        # per-round answers/confidence, forward-filled
        ans, conf = [], []
        last_a = {a: None for a in AGENTS}
        last_c = {a: None for a in AGENTS}
        for rd in rounds:
            arow, crow = [], []
            for ag in AGENTS:
                a = r.get(f'R{rd} {ag} Answer')
                a = last_a[ag] if pd.isna(a) else norm(a)
                last_a[ag] = a
                c = r.get(f'R{rd} {ag} Conf')
                c = last_c[ag] if pd.isna(c) else float(c)
                last_c[ag] = c
                arow.append(a)
                crow.append(c)
            ans.append(arow)
            conf.append(crow)

        init = ans[0]
        final = ans[-1]
        n_distinct_init = len({a for a in init if a})
        unanimous_init = (n_distinct_init == 1)
        correct_in_init = (correct in {a for a in init if a}) if correct else np.nan

        init_pred = plurality(init)
        final_pred = plurality(final)
        init_correct = (init_pred == correct) if correct else np.nan
        final_correct = (final_pred == correct) if correct else np.nan
        # true single-shot: a single agent's R1 answer (average correctness over the 3)
        single_shot_acc = np.mean([1.0 if (a == correct) else 0.0 for a in init if a]) if correct else np.nan

        # trajectory features
        switches = 0
        osc = 0  # A->B->A within an agent
        for k in range(3):
            traj = [ans[t][k] for t in range(len(ans))]
            for t in range(1, len(traj)):
                if traj[t] != traj[t-1]:
                    switches += 1
            for t in range(2, len(traj)):
                if traj[t] == traj[t-2] and traj[t] != traj[t-1]:
                    osc += 1
        # rounds to unanimous consensus (T+1 if never)
        tau = len(rounds) + 1
        for t in range(len(ans)):
            present = [a for a in ans[t] if a]
            if len(present) == 3 and len(set(present)) == 1:
                tau = t + 1  # 1-indexed round
                break
        mean_init_conf = np.nanmean([c for c in conf[0] if c is not None])
        mean_final_conf = np.nanmean([c for c in conf[-1] if c is not None])

        rows.append(dict(
            correct=correct, init_pred=init_pred, final_pred=final_pred,
            init_correct=init_correct, final_correct=final_correct, single_shot_acc=single_shot_acc,
            n_distinct_init=n_distinct_init, unanimous_init=unanimous_init,
            correct_in_init=correct_in_init,
            switches=switches, oscillation=osc, tau=tau,
            mean_init_conf=mean_init_conf, mean_final_conf=mean_final_conf,
            conf_change=mean_final_conf - mean_init_conf,
        ))
    return pd.DataFrame(rows)


def analyze(name, paths):
    df = pd.concat([per_question_features(p) for p in paths], ignore_index=True)
    df = df.dropna(subset=['init_correct', 'final_correct'])
    n = len(df)

    print(f'\n{"="*78}\n{name}  (pooled 3 seeds, n={n})\n{"="*78}')

    # ---- group definitions ----
    df['group'] = np.where(df['unanimous_init'], 'unanimous', 'disagreement')

    # overall accuracies
    ss = df['single_shot_acc'].mean()
    sc = df['init_correct'].mean()      # self-consistency (R1 plurality)
    fin = df['final_correct'].mean()    # debate final
    print(f'accuracy:  single-shot(1 agent)={ss:.3f}   self-consistency(R1 vote)={sc:.3f}   debate-final={fin:.3f}')
    print(f'debate gain vs single-shot = {fin-ss:+.3f}   vs self-consistency = {fin-sc:+.3f}')

    # ---- MONEY RESULT: recovery among initially-WRONG, by group ----
    wrong = df[df['init_correct'] == False]
    print(f'\ninitially-wrong questions: {len(wrong)}/{n} ({len(wrong)/n:.1%})')
    print(f'{"group":>14} {"n":>5} {"recover%":>9} {"(wrong->correct by final round)":>0}')
    rec = {}
    for g in ['unanimous', 'disagreement']:
        sub = wrong[wrong['group'] == g]
        r = sub['final_correct'].mean() if len(sub) else np.nan
        rec[g] = (len(sub), r, sub['final_correct'].sum())
        print(f'{g:>14} {len(sub):>5} {r:>9.3f}')
    # Fisher exact on recovery (disagreement vs unanimous)
    if all(rec[g][0] > 0 for g in rec):
        a = int(rec['disagreement'][2]); b = rec['disagreement'][0]-a
        c = int(rec['unanimous'][2]);    d = rec['unanimous'][0]-c
        try:
            orr, pval = fisher_exact([[a, b], [c, d]])
            print(f'   Fisher exact (disagreement vs unanimous recovery): OR={orr:.2f}, p={pval:.4g}')
        except Exception as e:
            print('   Fisher exact failed:', e)

    # ---- REFINEMENT: recovery by whether the correct answer is latent in the initial pool ----
    # The binding variable is not disagreement per se, but whether SOME clone already
    # reached the correct answer. Correct-absent = knowledge gap (debate can't create it);
    # correct-present-minority = reasoning/aggregation failure (debate can promote it).
    print('\nrecovery among initially-wrong, refined by "is correct answer in the R1 pool?":')
    refine = [
        ('unanimous-wrong (correct ABSENT)',        wrong[wrong['unanimous_init']]),
        ('disagree, correct ABSENT from pool',      wrong[(~wrong['unanimous_init']) & (~wrong['correct_in_init'])]),
        ('disagree, correct PRESENT (minority)',    wrong[(~wrong['unanimous_init']) & (wrong['correct_in_init'])]),
    ]
    for lbl, g in refine:
        if len(g):
            print(f'{lbl:>40}: n={len(g):>4}  recover%={g["final_correct"].mean():.3f}')

    # ---- corruption among initially-CORRECT, by group (bandwagon check) ----
    corr0 = df[df['init_correct'] == True]
    print(f'\ninitially-correct questions: {len(corr0)}/{n}   (corruption = correct->wrong by final)')
    for g in ['unanimous', 'disagreement']:
        sub = corr0[corr0['group'] == g]
        if len(sub):
            print(f'{g:>14} {len(sub):>5}  corrupt%={1-sub["final_correct"].mean():.3f}')

    # ---- signature separation: unanimous-WRONG vs disagreement-WRONG ----
    uw = wrong[wrong['group'] == 'unanimous']
    dw = wrong[wrong['group'] == 'disagreement']
    print('\nsignature of initially-wrong groups (knowledge=unanimous-wrong, reasoning=disagreement-wrong):')
    print(f'{"feature":>16} {"unanim-wrong":>14} {"disagr-wrong":>14} {"MWU p":>10}')
    for feat in ['mean_init_conf', 'switches', 'oscillation', 'tau', 'conf_change']:
        u, dd = uw[feat].dropna(), dw[feat].dropna()
        p = mannwhitneyu(u, dd).pvalue if len(u) > 3 and len(dd) > 3 else np.nan
        print(f'{feat:>16} {u.mean():>14.3f} {dd.mean():>14.3f} {p:>10.4g}')
    return df


if __name__ == '__main__':
    import os
    os.chdir(os.path.join(os.path.dirname(__file__) or '.', '..', 'New'))
    for name, paths in DATASETS.items():
        analyze(name, paths)
