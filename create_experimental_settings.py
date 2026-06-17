"""
Generate debate data for 4 experimental settings:
1. Baseline: Original method
2. Penalty-to-Loser: Losing agents become more assertive
3. Minority Protection: Minority agents receive extra reasoning
4. Devil's Advocate: Dominant agent must challenge own position
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

np.random.seed(42)

# Shared questions for all settings
QUESTIONS = [
    {
        "type": "categorical",
        "question": "Which renewable energy source should receive priority funding? A) Solar B) Nuclear C) Wind D) Hydroelectric",
        "correct": "B",  # Nuclear for base load
    },
    {
        "type": "categorical",
        "question": "What is the most effective way to reduce urban traffic? A) Public transit B) Road expansion C) Remote work D) Congestion pricing",
        "correct": "A",
    },
    {
        "type": "categorical",
        "question": "Which education reform improves outcomes most? A) Smaller classes B) Teacher training C) Technology D) Personalized learning",
        "correct": "D",
    },
    {
        "type": "categorical",
        "question": "Best approach to reduce crime? A) More police B) Education C) Economic opportunity D) Rehabilitation",
        "correct": "C",
    },
    {
        "type": "categorical",
        "question": "Most effective healthcare cost reduction? A) Preventive care B) Drug negotiation C) Single payer D) Competition",
        "correct": "A",
    },
]

AGENTS = ["Agent_Alpha", "Agent_Beta", "Agent_Gamma", "Agent_Delta"]
NUM_ROUNDS = 5


def generate_baseline_debate(question, q_idx):
    """Baseline: Standard debate with no interventions."""
    # Initial stances: distributed
    initial_stances = {
        "Agent_Alpha": "A",
        "Agent_Beta": "B",
        "Agent_Gamma": "C",
        "Agent_Delta": "B",  # Beta and Delta agree on B
    }

    correct_answer = question["correct"]

    debate_data = {}
    current_stances = initial_stances.copy()

    for round_num in range(1, NUM_ROUNDS + 1):
        for agent in AGENTS:
            if round_num == 1:
                # Initial positions
                answer = initial_stances[agent]
                conf = 60 + np.random.randint(0, 30)
                response = f"I believe {answer} is the best option because [initial reasoning from {agent}]."
            else:
                # Natural convergence toward majority
                votes = Counter(current_stances.values())
                majority = votes.most_common(1)[0][0]

                # 30% chance to move toward majority
                if np.random.rand() < 0.3 and current_stances[agent] != majority:
                    answer = majority
                    current_stances[agent] = answer
                    conf = 65 + np.random.randint(0, 25)
                    response = f"After considering others' arguments, I'm shifting to {answer}."
                else:
                    answer = current_stances[agent]
                    conf = 60 + np.random.randint(0, 30)
                    response = f"I maintain {answer} based on [reasoning]."

            debate_data[f"R{round_num} {agent} Answer"] = answer
            debate_data[f"R{round_num} {agent} Conf"] = f"{conf}%"
            debate_data[f"R{round_num} {agent} Response"] = response

    return debate_data


def generate_penalty_loser_debate(question, q_idx):
    """Penalty-to-Loser: Losing agents (minority) become more assertive."""
    initial_stances = {
        "Agent_Alpha": "A",
        "Agent_Beta": "B",
        "Agent_Gamma": "C",
        "Agent_Delta": "B",
    }

    correct_answer = question["correct"]
    debate_data = {}
    current_stances = initial_stances.copy()

    for round_num in range(1, NUM_ROUNDS + 1):
        # Identify minority agents
        votes = Counter(current_stances.values())
        majority = votes.most_common(1)[0][0]
        minority_agents = [a for a in AGENTS if current_stances[a] != majority]

        for agent in AGENTS:
            if round_num == 1:
                answer = initial_stances[agent]
                conf = 60 + np.random.randint(0, 30)
                response = f"I believe {answer} is the best option because [initial reasoning]."
            else:
                # Minority agents become MORE assertive (penalty for losing)
                if agent in minority_agents:
                    # Stay with position, increase confidence
                    answer = current_stances[agent]
                    conf = 75 + np.random.randint(0, 20)  # Higher confidence
                    response = f"**ASSERTIVE**: Despite being in minority, I strongly maintain {answer} because [detailed counter-arguments]. The majority may be wrong."
                else:
                    # Majority agents stay confident
                    answer = current_stances[agent]
                    conf = 70 + np.random.randint(0, 20)
                    response = f"I continue with {answer}, which has majority support."

            debate_data[f"R{round_num} {agent} Answer"] = answer
            debate_data[f"R{round_num} {agent} Conf"] = f"{conf}%"
            debate_data[f"R{round_num} {agent} Response"] = response

    return debate_data


def generate_minority_protection_debate(question, q_idx):
    """Minority Protection: Minority agents get extra reasoning rounds."""
    initial_stances = {
        "Agent_Alpha": "A",
        "Agent_Beta": "B",
        "Agent_Gamma": "C",
        "Agent_Delta": "B",
    }

    correct_answer = question["correct"]
    debate_data = {}
    current_stances = initial_stances.copy()

    for round_num in range(1, NUM_ROUNDS + 1):
        votes = Counter(current_stances.values())
        majority = votes.most_common(1)[0][0]
        minority_agents = [a for a in AGENTS if current_stances[a] != majority]

        for agent in AGENTS:
            if round_num == 1:
                answer = initial_stances[agent]
                conf = 60 + np.random.randint(0, 30)
                response = f"I believe {answer} is the best option."
            else:
                # Minority agents get EXTRA REASONING space
                if agent in minority_agents:
                    answer = current_stances[agent]
                    conf = 65 + np.random.randint(0, 25)
                    # Extra detailed reasoning for minority
                    response = (
                        f"**MINORITY PROTECTION ACTIVE**: As a minority voice, I provide extended reasoning: "
                        f"I maintain {answer} for these key reasons: "
                        f"(1) [evidence point 1], (2) [evidence point 2], (3) [counter to majority]. "
                        f"This perspective deserves equal consideration."
                    )
                else:
                    # Majority agents more willing to listen
                    if round_num >= 3 and np.random.rand() < 0.2:
                        # Some majority agents reconsider
                        minority_view = minority_agents[0] if minority_agents else agent
                        answer = current_stances[minority_view] if minority_agents else current_stances[agent]
                        current_stances[agent] = answer
                        conf = 60 + np.random.randint(0, 20)
                        response = f"After hearing minority arguments, I'm reconsidering. Moving to {answer}."
                    else:
                        answer = current_stances[agent]
                        conf = 65 + np.random.randint(0, 25)
                        response = f"I acknowledge minority views. Still supporting {answer}."

            debate_data[f"R{round_num} {agent} Answer"] = answer
            debate_data[f"R{round_num} {agent} Conf"] = f"{conf}%"
            debate_data[f"R{round_num} {agent} Response"] = response

    return debate_data


def generate_devils_advocate_debate(question, q_idx):
    """Devil's Advocate: Dominant agent must challenge their own position."""
    initial_stances = {
        "Agent_Alpha": "A",
        "Agent_Beta": "B",
        "Agent_Gamma": "C",
        "Agent_Delta": "B",
    }

    correct_answer = question["correct"]
    debate_data = {}
    current_stances = initial_stances.copy()

    for round_num in range(1, NUM_ROUNDS + 1):
        votes = Counter(current_stances.values())
        if len(votes) > 0:
            dominant_position = votes.most_common(1)[0][0]
            dominant_agents = [a for a in AGENTS if current_stances[a] == dominant_position]
        else:
            dominant_agents = []

        for agent in AGENTS:
            if round_num == 1:
                answer = initial_stances[agent]
                conf = 60 + np.random.randint(0, 30)
                response = f"I believe {answer} is the best option."
            else:
                # Dominant agents must play devil's advocate
                if agent in dominant_agents and len(dominant_agents) >= 2:
                    answer = current_stances[agent]
                    conf = 55 + np.random.randint(0, 20)  # Slightly lower confidence
                    response = (
                        f"**DEVIL'S ADVOCATE MODE**: While I support {answer}, "
                        f"I must challenge my own position: What if [counter-argument]? "
                        f"Alternative {chr(65 + (ord(answer) - 65 + 1) % 4)} might have merit because [challenge]. "
                        f"However, I still lean toward {answer}."
                    )
                else:
                    # Non-dominant agents argue normally
                    answer = current_stances[agent]
                    conf = 65 + np.random.randint(0, 25)
                    response = f"I continue to support {answer}. [reasoning]"

            debate_data[f"R{round_num} {agent} Answer"] = answer
            debate_data[f"R{round_num} {agent} Conf"] = f"{conf}%"
            debate_data[f"R{round_num} {agent} Response"] = response

    return debate_data


def create_workbook(setting_name, generator_func):
    """Create a full workbook for a given setting."""
    rows = []

    for idx, q in enumerate(QUESTIONS, start=1):
        row = {
            "Question #": idx,
            "Question": q["question"],
            "Correct?": q["correct"],
            "Rounds to Consensus": None,  # Will be computed
        }

        # Generate debate data
        debate_data = generator_func(q, idx)
        row.update(debate_data)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save to Excel
    output_path = Path(f"debates_{setting_name}.xlsx")
    df.to_excel(output_path, index=False, sheet_name="Debates")
    print(f"Created: {output_path}")

    return output_path


def main():
    print("="*70)
    print("CREATING 4 EXPERIMENTAL SETTINGS")
    print("="*70)
    print()

    settings = {
        "baseline": ("Baseline (Original)", generate_baseline_debate),
        "penalty_loser": ("Penalty-to-Loser", generate_penalty_loser_debate),
        "minority_protection": ("Minority Protection", generate_minority_protection_debate),
        "devils_advocate": ("Devil's Advocate", generate_devils_advocate_debate),
    }

    created_files = []

    for setting_key, (setting_name, generator) in settings.items():
        print(f"\nCreating: {setting_name}")
        print(f"  Mechanism: {generator.__doc__.split('.')[0]}")
        filepath = create_workbook(setting_key, generator)
        created_files.append(filepath)

    print()
    print("="*70)
    print("CREATED FILES:")
    print("="*70)
    for f in created_files:
        print(f"  - {f}")

    print()
    print("Next steps:")
    print("  1. Run: python run_all_experiments.py")
    print("  2. Compare: python compare_settings.py")


if __name__ == "__main__":
    main()
