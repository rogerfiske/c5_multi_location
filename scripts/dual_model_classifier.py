"""
Dual-Model with Confidence Classifier

Approach:
1. Set A: Top-20 pool (adjacency model)
2. Set B: Remaining 19 parts (inverse)
3. Classifier predicts: Use A, Use B, or Abstain

Goal: Push predictions to extremes (0-1 wrong or 4-5 wrong)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_data():
    """Load CA5 matrix data"""
    df = pd.read_csv(DATA_DIR / "CA5_matrix.csv")
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_pool_prediction(df, current_idx, window=30):
    """
    Generate top-20 pool using adjacency model (simplified version)
    Returns: set of top 20 predicted parts
    """
    if current_idx < 1:
        return set(range(1, 21))  # Default

    prev_row = df.iloc[current_idx - 1]
    prev_parts = [prev_row['L_1'], prev_row['L_2'], prev_row['L_3'],
                  prev_row['L_4'], prev_row['L_5']]

    # Score parts by adjacency to previous day + position frequency
    scores = defaultdict(float)

    # Adjacency boost (within +/- 3)
    for p in prev_parts:
        for offset in range(-3, 4):
            candidate = p + offset
            if 1 <= candidate <= 39:
                boost = 3.0 if offset == 0 else (2.0 if abs(offset) <= 1 else 1.0)
                scores[candidate] += boost

    # Position frequency from rolling window
    start_idx = max(0, current_idx - window)
    window_df = df.iloc[start_idx:current_idx]

    for pos in range(1, 6):
        col = f'L_{pos}'
        counts = window_df[col].value_counts()
        for part, count in counts.items():
            scores[part] += count / len(window_df)

    # Rank and return top 20
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_20 = set([p for p, s in ranked[:20]])

    return top_20


def evaluate_pool(pool, actual_parts):
    """
    Evaluate how many actual parts are in the pool
    Returns: wrong count (0-5)
    """
    hits = len(actual_parts & pool)
    wrong = 5 - hits
    return wrong


def extract_features(df, idx):
    """
    Extract features for the classifier to predict regime
    """
    if idx < 7:
        return None

    features = {}

    # Previous day's parts
    prev = df.iloc[idx - 1]
    prev_parts = set([prev['L_1'], prev['L_2'], prev['L_3'], prev['L_4'], prev['L_5']])

    # Day of week
    features['day_of_week'] = df.iloc[idx]['date'].dayofweek

    # Previous days' adjacency pattern
    for lookback in [1, 2, 3]:
        if idx >= lookback + 1:
            day_prev = df.iloc[idx - lookback]
            day_prev2 = df.iloc[idx - lookback - 1]

            parts_prev = set([day_prev['L_1'], day_prev['L_2'], day_prev['L_3'],
                            day_prev['L_4'], day_prev['L_5']])
            parts_prev2 = set([day_prev2['L_1'], day_prev2['L_2'], day_prev2['L_3'],
                             day_prev2['L_4'], day_prev2['L_5']])

            # How many parts repeated
            repeats = len(parts_prev & parts_prev2)
            features[f'repeats_lag{lookback}'] = repeats

            # Adjacency count
            adjacent = 0
            for p1 in parts_prev:
                for p2 in parts_prev2:
                    if abs(p1 - p2) <= 3:
                        adjacent += 1
            features[f'adjacent_lag{lookback}'] = adjacent

    # Spread of previous day's parts
    prev_list = [prev['L_1'], prev['L_2'], prev['L_3'], prev['L_4'], prev['L_5']]
    features['prev_spread'] = max(prev_list) - min(prev_list)
    features['prev_mean'] = np.mean(prev_list)
    features['prev_std'] = np.std(prev_list)

    # Gap patterns
    gaps = [prev_list[i+1] - prev_list[i] for i in range(4)]
    features['max_gap'] = max(gaps)
    features['min_gap'] = min(gaps)
    features['mean_gap'] = np.mean(gaps)

    # Rolling volatility (how much do parts change day-over-day)
    volatility = 0
    for lookback in range(1, min(8, idx)):
        day1 = df.iloc[idx - lookback]
        day2 = df.iloc[idx - lookback - 1]
        parts1 = set([day1['L_1'], day1['L_2'], day1['L_3'], day1['L_4'], day1['L_5']])
        parts2 = set([day2['L_1'], day2['L_2'], day2['L_3'], day2['L_4'], day2['L_5']])
        volatility += 5 - len(parts1 & parts2)
    features['volatility_7d'] = volatility

    # Position-specific features
    for pos in range(1, 6):
        col = f'L_{pos}'
        features[f'prev_L{pos}'] = prev[col]

        # Trend in this position
        if idx >= 3:
            trend = df.iloc[idx-1][col] - df.iloc[idx-3][col]
            features[f'trend_L{pos}'] = trend

    return features


def build_dataset(df, min_idx=30):
    """
    Build training dataset with features and labels
    """
    X = []
    y = []
    dates = []
    wrong_counts = []

    for idx in range(min_idx, len(df)):
        # Get prediction and actual
        pool = get_pool_prediction(df, idx)
        actual_row = df.iloc[idx]
        actual_parts = set([actual_row['L_1'], actual_row['L_2'], actual_row['L_3'],
                          actual_row['L_4'], actual_row['L_5']])

        wrong = evaluate_pool(pool, actual_parts)

        # Extract features
        features = extract_features(df, idx)
        if features is None:
            continue

        # Label: 0 = Use Set A (0-1 wrong), 1 = Use Set B (4-5 wrong), 2 = Abstain (2-3 wrong)
        if wrong <= 1:
            label = 0  # Set A is good
        elif wrong >= 4:
            label = 1  # Set B is good (inverse)
        else:
            label = 2  # Neither - abstain

        X.append(features)
        y.append(label)
        dates.append(actual_row['date'])
        wrong_counts.append(wrong)

    return pd.DataFrame(X), np.array(y), dates, wrong_counts


def evaluate_strategy(y_true, y_pred, wrong_counts):
    """
    Evaluate the dual-model strategy
    """
    results = {
        'total_days': len(y_true),
        'predicted_A': 0,  # Days we predicted to use Set A
        'predicted_B': 0,  # Days we predicted to use Set B
        'predicted_abstain': 0,
        'correct_routing': 0,  # Routed to correct set
        'wrong_routing': 0,  # Routed to wrong set
        'actionable_abstain': 0,  # Abstained but one set was good
        'correct_abstain': 0,  # Abstained correctly (2-3 wrong)
    }

    for i in range(len(y_true)):
        wrong = wrong_counts[i]
        pred = y_pred[i]

        if pred == 0:  # Predicted: Use Set A
            results['predicted_A'] += 1
            if wrong <= 1:
                results['correct_routing'] += 1
            else:
                results['wrong_routing'] += 1

        elif pred == 1:  # Predicted: Use Set B (inverse)
            results['predicted_B'] += 1
            if wrong >= 4:
                results['correct_routing'] += 1
            else:
                results['wrong_routing'] += 1

        else:  # Predicted: Abstain
            results['predicted_abstain'] += 1
            if wrong in [2, 3]:
                results['correct_abstain'] += 1
            else:
                results['actionable_abstain'] += 1

    return results


def main():
    print("=" * 70)
    print("DUAL-MODEL CONFIDENCE CLASSIFIER")
    print("=" * 70)

    df = load_data()
    print(f"Data loaded: {len(df)} days")

    # Build dataset
    print("\nBuilding feature dataset...")
    X, y, dates, wrong_counts = build_dataset(df)
    print(f"Dataset size: {len(X)} samples")

    # Class distribution
    print("\n" + "=" * 70)
    print("BASELINE DISTRIBUTION (Ground Truth)")
    print("=" * 70)
    counter = Counter(y)
    print(f"Set A good (0-1 wrong): {counter[0]} ({counter[0]/len(y)*100:.1f}%)")
    print(f"Set B good (4-5 wrong): {counter[1]} ({counter[1]/len(y)*100:.1f}%)")
    print(f"Neither (2-3 wrong):    {counter[2]} ({counter[2]/len(y)*100:.1f}%)")

    actionable = counter[0] + counter[1]
    print(f"\nActionable days (0-1 or 4-5): {actionable} ({actionable/len(y)*100:.1f}%)")

    # Wrong count distribution
    print("\nWrong count distribution:")
    wrong_counter = Counter(wrong_counts)
    for w in range(6):
        print(f"  {w} wrong: {wrong_counter[w]} ({wrong_counter[w]/len(wrong_counts)*100:.1f}%)")

    # Train classifier
    print("\n" + "=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)

    # Split data (time-based)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    wrong_train = wrong_counts[:split_idx]
    wrong_test = wrong_counts[split_idx:]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Try multiple classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Accuracy
        accuracy = (y_pred == y_test).mean()
        print(f"Accuracy: {accuracy:.3f}")

        # Evaluate strategy
        results = evaluate_strategy(y_test, y_pred, wrong_test)

        print(f"\nStrategy Results:")
        print(f"  Predicted Use A:   {results['predicted_A']:>5} days")
        print(f"  Predicted Use B:   {results['predicted_B']:>5} days")
        print(f"  Predicted Abstain: {results['predicted_abstain']:>5} days")
        total_routed = results['predicted_A'] + results['predicted_B']
        if total_routed > 0:
            print(f"  Correct Routing:   {results['correct_routing']:>5} ({results['correct_routing']/total_routed*100:.1f}% when not abstaining)")
        else:
            print(f"  Correct Routing:   {results['correct_routing']:>5} (N/A - no routing predictions)")
        print(f"  Wrong Routing:     {results['wrong_routing']:>5}")
        print(f"  Correct Abstain:   {results['correct_abstain']:>5}")
        print(f"  Missed Opportunity:{results['actionable_abstain']:>5}")

        # Feature importance
        if hasattr(clf, 'feature_importances_'):
            importances = pd.Series(clf.feature_importances_, index=X.columns)
            print(f"\nTop 10 Features:")
            for feat, imp in importances.nlargest(10).items():
                print(f"  {feat:>20}: {imp:.4f}")

    # Alternative: Confidence-based thresholding
    print("\n" + "=" * 70)
    print("CONFIDENCE-BASED THRESHOLDING")
    print("=" * 70)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # Get probabilities
    probs = clf.predict_proba(X_test)

    # Try different confidence thresholds
    print("\nResults by confidence threshold:")
    print(f"{'Threshold':>10} {'Predictions':>12} {'Correct':>10} {'Precision':>10} {'Coverage':>10}")
    print("-" * 56)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        confident_mask = probs.max(axis=1) >= threshold
        confident_preds = clf.predict(X_test)[confident_mask]
        confident_true = y_test[confident_mask]
        confident_wrong = [wrong_test[i] for i in range(len(wrong_test)) if confident_mask[i]]

        if len(confident_preds) == 0:
            continue

        # Evaluate
        correct = 0
        for i in range(len(confident_preds)):
            pred = confident_preds[i]
            wrong = confident_wrong[i]

            if pred == 0 and wrong <= 1:  # Correctly predicted Set A
                correct += 1
            elif pred == 1 and wrong >= 4:  # Correctly predicted Set B
                correct += 1
            elif pred == 2 and wrong in [2, 3]:  # Correctly abstained
                correct += 1

        precision = correct / len(confident_preds)
        coverage = len(confident_preds) / len(y_test)

        print(f"{threshold:>10.1f} {len(confident_preds):>12} {correct:>10} {precision:>9.1%} {coverage:>9.1%}")

    # Final evaluation: Practical strategy
    print("\n" + "=" * 70)
    print("PRACTICAL STRATEGY EVALUATION")
    print("=" * 70)

    # Strategy: Predict when confidence > 0.5, route to highest probability class
    threshold = 0.5

    actionable_days = 0
    actionable_correct = 0
    abstain_days = 0

    for i in range(len(probs)):
        max_prob = probs[i].max()
        pred_class = probs[i].argmax()
        wrong = wrong_test[i]

        if max_prob >= threshold and pred_class != 2:  # Confident prediction (not abstain)
            actionable_days += 1
            if pred_class == 0 and wrong <= 1:
                actionable_correct += 1
            elif pred_class == 1 and wrong >= 4:
                actionable_correct += 1
        else:
            abstain_days += 1

    print(f"Confidence threshold: {threshold}")
    print(f"Actionable predictions: {actionable_days} ({actionable_days/len(y_test)*100:.1f}%)")
    if actionable_days > 0:
        print(f"Correct actionable:     {actionable_correct} ({actionable_correct/actionable_days*100:.1f}% precision)")
    else:
        print(f"Correct actionable:     0 (N/A)")
    print(f"Abstain days:           {abstain_days} ({abstain_days/len(y_test)*100:.1f}%)")

    # Compare to baseline (always predict)
    baseline_correct = sum(1 for w in wrong_test if w <= 1 or w >= 4)
    print(f"\nBaseline (always predict): {baseline_correct/len(y_test)*100:.1f}% actionable")
    if actionable_days > 0:
        print(f"Our strategy:              {actionable_correct/actionable_days*100:.1f}% precision on {actionable_days/len(y_test)*100:.1f}% of days")
    else:
        print(f"Our strategy:              N/A (no actionable predictions)")


if __name__ == "__main__":
    main()
