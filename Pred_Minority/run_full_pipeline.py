"""Run the complete minority predictor pipeline.

This script orchestrates:
1. Feature extraction from baseline workbook
2. Evidence-aware scoring (optional)
3. Model training
4. Model evaluation
"""

from pathlib import Path
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_features import extract_minority_features
from evidence_scorer import augment_features_with_evidence_scores
from utils import load_workbook_sheets, get_agent_names


def main():
    parser = argparse.ArgumentParser(
        description="Run complete minority predictor pipeline"
    )
    parser.add_argument(
        '--baseline-workbook',
        type=Path,
        default=Path('docs/qwen_mmlu_exp1.xlsx'),
        help='Path to baseline debate workbook'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('Pred_Minority'),
        help='Output directory for all results'
    )
    parser.add_argument(
        '--skip-evidence',
        action='store_true',
        help='Skip evidence-aware scoring (faster, slightly lower accuracy)'
    )
    parser.add_argument(
        '--min-round',
        type=int,
        default=2,
        help='Minimum round to extract'
    )
    parser.add_argument(
        '--max-round',
        type=int,
        default=4,
        help='Maximum round to extract'
    )

    args = parser.parse_args()

    # Setup paths
    features_path = args.output_dir / 'features_baseline.csv'
    models_dir = args.output_dir / 'models'
    eval_dir = args.output_dir / 'evaluation'

    print("="*70)
    print("MINORITY PREDICTOR PIPELINE")
    print("="*70)
    print(f"Baseline workbook: {args.baseline_workbook}")
    print(f"Output directory: {args.output_dir}")
    print(f"Evidence scoring: {'Disabled' if args.skip_evidence else 'Enabled'}")
    print("="*70)

    # Step 1: Extract features
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING FEATURES")
    print("="*70)

    features_df = extract_minority_features(
        workbook_path=args.baseline_workbook,
        min_round=args.min_round,
        max_round=args.max_round,
    )

    # Step 2: Add evidence-aware scores (optional)
    if not args.skip_evidence:
        print("\n" + "="*70)
        print("STEP 2: COMPUTING EVIDENCE-AWARE SCORES")
        print("="*70)

        sheets = load_workbook_sheets(args.baseline_workbook)
        debates_df = sheets['Debate_Traces']
        agents = get_agent_names(debates_df)

        features_df = augment_features_with_evidence_scores(
            features_df=features_df,
            debates_df=debates_df,
            agents=agents,
        )

    # Save features
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"\n✓ Saved features to: {features_path}")

    # Step 3: Train models
    print("\n" + "="*70)
    print("STEP 3: TRAINING MODELS")
    print("="*70)

    import train_predictor
    sys.argv = [
        'train_predictor.py',
        '--features', str(features_path),
        '--output-dir', str(models_dir),
    ]
    train_predictor.main()

    # Step 4: Evaluate models
    print("\n" + "="*70)
    print("STEP 4: EVALUATING MODELS")
    print("="*70)

    import evaluate_predictor
    sys.argv = [
        'evaluate_predictor.py',
        '--features', str(features_path),
        '--model', str(models_dir / 'ensemble_model.pkl'),
        '--output-dir', str(eval_dir),
    ]
    evaluate_predictor.main()

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  Features:    {features_path}")
    print(f"  Models:      {models_dir}/")
    print(f"  Evaluation:  {eval_dir}/")
    print(f"\nKey outputs:")
    print(f"  - Trained predictor: {models_dir}/ensemble_model.pkl")
    print(f"  - Predictions: {eval_dir}/predictions.csv")
    print(f"  - ROC curve: {eval_dir}/roc_curve.png")
    print(f"  - PR curve: {eval_dir}/pr_curve.png")
    print(f"  - Calibration: {eval_dir}/calibration.png")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review evaluation metrics in:")
    print(f"   {eval_dir}/")
    print("\n2. Use predictor in debates:")
    print("   from Pred_Minority.predictor import MinorityPredictor")
    print(f"   predictor = MinorityPredictor('{models_dir}/ensemble_model.pkl')")
    print("\n3. Run Experiment 4 with learned predictor:")
    print("   python docs/generate_qwen_mmlu_exp4.py \\")
    print(f"       --predictor {models_dir}/ensemble_model.pkl")


if __name__ == '__main__':
    main()
