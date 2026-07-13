"""Evidence quality scoring module.

This module implements evidence-aware scoring by analyzing the reasoning
text and computing quality metrics beyond just the LLM judge scores.
"""

from typing import Dict, List, Tuple
import re
import numpy as np
import pandas as pd


class EvidenceScorer:
    """Scores evidence quality from agent reasoning text."""

    def __init__(self):
        """Initialize evidence scorer with pattern matchers."""
        # Patterns for specific reasoning indicators
        self.specific_patterns = [
            r'\b(?:because|since|given that|considering)\b',
            r'\b(?:data|evidence|study|research|paper)\b',
            r'\b(?:calculation|compute|derive|prove)\b',
            r'\b(?:example|instance|case)\b',
            r'\b\d+%\b',  # Percentages
            r'\b\d{4}\b',  # Years
            r'\b(?:equation|formula|theorem)\b',
        ]

        # Patterns for hedging/uncertainty
        self.hedging_patterns = [
            r'\b(?:maybe|perhaps|possibly|might|could)\b',
            r'\b(?:I think|I believe|I guess)\b',
            r'\b(?:not sure|unclear|uncertain)\b',
        ]

        # Patterns for copying/agreement
        self.copying_patterns = [
            r'\bI agree with Agent\d+\b',
            r'\bsame as Agent\d+\b',
            r'\b(?:Agent\d+ is correct|Agent\d+ said)\b',
        ]

        # Patterns for counterevidence consideration
        self.counterevidence_patterns = [
            r'\b(?:however|but|although|despite)\b',
            r'\b(?:on the other hand|alternatively)\b',
            r'\b(?:counterargument|counter-example)\b',
            r'\b(?:while|whereas)\b',
        ]

    def score_reasoning_text(self, text: str) -> Dict[str, float]:
        """Score a piece of reasoning text.

        Args:
            text: Agent's reasoning explanation

        Returns:
            Dictionary of evidence quality scores
        """
        if not text or pd.isna(text):
            return {
                'specificity': 0.0,
                'hedging': 0.0,
                'copying': 0.0,
                'counterevidence': 0.0,
                'length_normalized': 0.0,
            }

        text_lower = text.lower()
        word_count = len(text.split())

        # Specificity: specific claims and evidence
        specificity_matches = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.specific_patterns
        )
        specificity = min(specificity_matches / max(word_count / 20, 1), 1.0)

        # Hedging: uncertainty markers (lower is better)
        hedging_matches = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.hedging_patterns
        )
        hedging = min(hedging_matches / max(word_count / 30, 1), 1.0)

        # Copying: agreement without reasoning (lower is better)
        copying_matches = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.copying_patterns
        )
        copying = min(copying_matches / max(word_count / 20, 1), 1.0)

        # Counterevidence: addresses alternative views (higher is better)
        counter_matches = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.counterevidence_patterns
        )
        counterevidence = min(counter_matches / max(word_count / 20, 1), 1.0)

        # Length normalization (penalize very short responses)
        length_normalized = min(word_count / 50, 1.0)

        return {
            'specificity': specificity,
            'hedging': hedging,
            'copying': copying,
            'counterevidence': counterevidence,
            'length_normalized': length_normalized,
        }

    def compute_composite_quality(
        self,
        llm_quality: float,
        text_scores: Dict[str, float],
        confidence: float,
    ) -> float:
        """Combine multiple quality signals into composite score.

        Args:
            llm_quality: LLM judge explanation quality (0-1)
            text_scores: Dictionary of text-based scores
            confidence: Agent's confidence (0-1)

        Returns:
            Composite quality score (0-1)
        """
        # Weighted combination
        composite = (
            0.40 * llm_quality +  # LLM judge is primary signal
            0.15 * text_scores['specificity'] +
            0.10 * (1 - text_scores['hedging']) +  # Less hedging is better
            0.10 * (1 - text_scores['copying']) +  # Less copying is better
            0.10 * text_scores['counterevidence'] +
            0.10 * text_scores['length_normalized'] +
            0.05 * confidence  # Slight weight on confidence
        )

        return np.clip(composite, 0, 1)


def augment_features_with_evidence_scores(
    features_df: pd.DataFrame,
    debates_df: pd.DataFrame,
    agents: List[str],
) -> pd.DataFrame:
    """Augment feature dataframe with evidence-aware scores.

    Args:
        features_df: DataFrame with minority features
        debates_df: Debate_Traces DataFrame with reasoning text
        agents: List of agent names

    Returns:
        Augmented features DataFrame with evidence scores
    """
    scorer = EvidenceScorer()

    # Add evidence score columns
    evidence_cols = [
        'evidence_specificity',
        'evidence_hedging',
        'evidence_copying',
        'evidence_counterevidence',
        'evidence_length',
        'evidence_composite',
    ]

    for col in evidence_cols:
        features_df[col] = 0.0

    print("Computing evidence-aware scores...")

    for idx, row in features_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(features_df)} minority situations")

        question_idx = row['question_idx']
        round_num = row['round']
        minority_answer = row['minority_answer']

        # Find agents supporting this minority
        debate_row = debates_df.iloc[question_idx]

        minority_text_scores = []
        minority_composites = []

        for agent in agents:
            # Check if this agent has the minority answer
            answer_col = f'R{round_num} {agent} Answer'
            agent_answer = debate_row.get(answer_col, None)

            if agent_answer != minority_answer:
                continue

            # Get reasoning text
            response_col = f'R{round_num} {agent} Response'
            reasoning_text = debate_row.get(response_col, '')

            # Score reasoning text
            text_scores = scorer.score_reasoning_text(reasoning_text)
            minority_text_scores.append(text_scores)

            # Compute composite with LLM quality and confidence
            llm_quality = row['minority_quality_mean']  # Use average from features
            confidence = row['minority_conf_mean']

            composite = scorer.compute_composite_quality(
                llm_quality=llm_quality,
                text_scores=text_scores,
                confidence=confidence,
            )
            minority_composites.append(composite)

        # Average across minority agents
        if minority_text_scores:
            features_df.at[idx, 'evidence_specificity'] = np.mean(
                [s['specificity'] for s in minority_text_scores]
            )
            features_df.at[idx, 'evidence_hedging'] = np.mean(
                [s['hedging'] for s in minority_text_scores]
            )
            features_df.at[idx, 'evidence_copying'] = np.mean(
                [s['copying'] for s in minority_text_scores]
            )
            features_df.at[idx, 'evidence_counterevidence'] = np.mean(
                [s['counterevidence'] for s in minority_text_scores]
            )
            features_df.at[idx, 'evidence_length'] = np.mean(
                [s['length_normalized'] for s in minority_text_scores]
            )
            features_df.at[idx, 'evidence_composite'] = np.mean(minority_composites)

    print(f"  Completed all {len(features_df)} minority situations")

    return features_df


if __name__ == '__main__':
    # Test the scorer
    scorer = EvidenceScorer()

    test_texts = [
        "I strongly agree because the data from 2020 shows a 15% increase.",
        "I think maybe it could be correct, not sure though.",
        "I agree with Agent1, they are correct.",
        "While Agent2 makes a good point, however the equation shows otherwise.",
    ]

    print("Testing Evidence Scorer:\n")
    for i, text in enumerate(test_texts, 1):
        scores = scorer.score_reasoning_text(text)
        print(f"Text {i}: {text[:60]}...")
        for metric, value in scores.items():
            print(f"  {metric:20s}: {value:.3f}")
        print()
