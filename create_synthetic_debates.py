"""Generate synthetic multi-agent debate data for testing the diagnostic pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# Define debate questions
questions = [
    {
        "type": "likert",
        "question": "Climate change policies should prioritize economic growth over environmental protection.",
        "correct": None,
    },
    {
        "type": "categorical",
        "question": "Which energy source should receive the most government investment? A) Solar B) Nuclear C) Wind D) Natural Gas",
        "correct": "B",
    },
    {
        "type": "likert",
        "question": "Universal basic income would improve social welfare more than targeted assistance programs.",
        "correct": None,
    },
    {
        "type": "categorical",
        "question": "What is the primary cause of urban traffic congestion? A) Insufficient public transit B) Poor road design C) Excessive private vehicle use D) Inadequate traffic management",
        "correct": "C",
    },
    {
        "type": "likert",
        "question": "Remote work should become the default for knowledge workers.",
        "correct": None,
    },
    {
        "type": "categorical",
        "question": "Which education reform would most improve student outcomes? A) Smaller class sizes B) Higher teacher pay C) More technology D) Personalized learning",
        "correct": "D",
    },
    {
        "type": "likert",
        "question": "Social media platforms should be legally liable for user-generated content.",
        "correct": None,
    },
    {
        "type": "categorical",
        "question": "What is the most effective approach to reduce urban crime? A) More police B) Better education C) Economic opportunity D) Rehabilitation programs",
        "correct": "C",
    },
]

agents = ["Agent_Alpha", "Agent_Beta", "Agent_Gamma", "Agent_Delta"]
num_rounds = 5

def generate_likert_debate(question_data, agent_behaviors):
    """Generate a Likert-style debate with specified agent behaviors."""
    initial_stances = {
        "Agent_Alpha": -2,
        "Agent_Beta": 2,
        "Agent_Gamma": 0,
        "Agent_Delta": 1,
    }

    debate_data = {}

    for round_num in range(1, num_rounds + 1):
        for agent in agents:
            behavior = agent_behaviors.get(agent, "normal")

            # Determine stance
            if round_num == 1:
                stance = initial_stances[agent]
            else:
                if behavior == "dogmatic":
                    # Stay fixed
                    stance = initial_stances[agent]
                elif behavior == "sycophant":
                    # Quickly move to dominant agent's view
                    stance = initial_stances["Agent_Beta"] if round_num >= 2 else initial_stances[agent]
                elif behavior == "responsive":
                    # Gradually move toward consensus
                    prev_stance = initial_stances[agent]
                    mean_others = np.mean([initial_stances[a] for a in agents if a != agent])
                    stance = int(np.clip(prev_stance + 0.3 * (mean_others - prev_stance), -2, 2))
                else:  # normal
                    # Some movement with noise
                    prev_stance = initial_stances[agent]
                    drift = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                    stance = int(np.clip(prev_stance + drift, -2, 2))
                    initial_stances[agent] = stance

            # Map to text
            stance_map = {-2: "Strongly Disagree", -1: "Disagree", 0: "Neutral", 1: "Agree", 2: "Strongly Agree"}
            debate_data[f"R{round_num} {agent} Answer"] = stance_map[stance]

            # Confidence
            if behavior == "dogmatic":
                conf = 95 + np.random.randint(0, 5)
            elif behavior == "sycophant":
                conf = 50 + np.random.randint(0, 30)
            else:
                conf = 60 + np.random.randint(0, 30)
            debate_data[f"R{round_num} {agent} Conf"] = f"{conf}%"

            # Response (explanation)
            if round_num == 1:
                response = f"Initial position: I believe this because [reasoning from {agent}'s perspective]."
            else:
                if behavior == "dogmatic":
                    response = f"I maintain my original position. My reasoning remains unchanged from round 1."
                elif behavior == "sycophant":
                    response = f"After hearing others, I agree with Agent_Beta's point. Changing my view to align."
                elif behavior == "responsive":
                    response = f"Considering the arguments presented, I see merit in moving toward consensus. Adjusting my stance."
                else:
                    response = f"Reflecting on round {round_num-1} discussion, I {['maintain', 'slightly adjust', 'reconsider'][np.random.randint(0, 3)]} my position."

            debate_data[f"R{round_num} {agent} Response"] = response

    return debate_data

def generate_categorical_debate(question_data, agent_behaviors):
    """Generate a categorical-answer debate."""
    initial_answers = {
        "Agent_Alpha": "A",
        "Agent_Beta": "B",
        "Agent_Gamma": "C",
        "Agent_Delta": "B",
    }

    debate_data = {}

    for round_num in range(1, num_rounds + 1):
        for agent in agents:
            behavior = agent_behaviors.get(agent, "normal")

            if round_num == 1:
                answer = initial_answers[agent]
            else:
                if behavior == "dogmatic":
                    answer = initial_answers[agent]
                elif behavior == "sycophant":
                    # Copy majority or dominant agent
                    answer = "B" if round_num >= 2 else initial_answers[agent]
                elif behavior == "responsive":
                    # Move toward most common answer
                    if round_num >= 3:
                        answer = "B"  # Converge
                    else:
                        answer = initial_answers[agent]
                else:
                    # Some chance of changing
                    if np.random.rand() < 0.3 and round_num >= 2:
                        answer = np.random.choice(["A", "B", "C", "D"])
                    else:
                        answer = initial_answers[agent]
                    initial_answers[agent] = answer

            debate_data[f"R{round_num} {agent} Answer"] = answer

            # Confidence
            conf = 60 + np.random.randint(0, 35)
            debate_data[f"R{round_num} {agent} Conf"] = f"{conf}%"

            # Response
            if round_num == 1:
                response = f"I choose {answer} because it addresses the root cause effectively."
            else:
                if behavior == "dogmatic":
                    response = f"I still believe {answer} is the correct answer. No change in my reasoning."
                elif behavior == "sycophant":
                    response = f"Others make good points. I'm switching to {answer} to align with the group."
                else:
                    response = f"After round {round_num-1}, I {['maintain', 'reconsider'][int(answer != initial_answers.get(agent, answer))]} my answer of {answer}."

            debate_data[f"R{round_num} {agent} Response"] = response

    return debate_data

# Generate dataset
rows = []

for idx, q in enumerate(questions, start=1):
    # Vary agent behaviors across questions
    if idx == 1:
        behaviors = {"Agent_Alpha": "dogmatic"}  # Dogmatism test
    elif idx == 2:
        behaviors = {"Agent_Gamma": "sycophant"}  # Sycophancy test
    elif idx == 3:
        behaviors = {"Agent_Beta": "dominant"}  # Domination test
    elif idx == 4:
        behaviors = {a: "responsive" for a in agents}  # Healthy debate
    elif idx == 5:
        behaviors = {"Agent_Alpha": "dogmatic", "Agent_Delta": "dogmatic"}  # Low engagement
    else:
        behaviors = {a: "normal" for a in agents}  # Normal variety

    row = {
        "Question #": idx,
        "Question": q["question"],
        "Correct?": q["correct"] if q["correct"] else "",
        "Rounds to Consensus": np.random.choice([None, 3, 4, 5]),
    }

    if q["type"] == "likert":
        row.update(generate_likert_debate(q, behaviors))
    else:
        row.update(generate_categorical_debate(q, behaviors))

    rows.append(row)

df = pd.DataFrame(rows)

# Save to Excel
output_path = Path("synthetic_debates.xlsx")
df.to_excel(output_path, index=False, sheet_name="Debates")
print(f"Created synthetic debate workbook: {output_path}")
print(f"Questions: {len(df)}")
print(f"Agents: {len(agents)}")
print(f"Rounds: {num_rounds}")
print(f"\nColumns: {len(df.columns)}")
print(f"Sample question types: {[q['type'] for q in questions[:3]]}")
