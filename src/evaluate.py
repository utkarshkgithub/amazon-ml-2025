"""
Generate Final Submission File: test_out.csv
Uses trained LightGBM model: last-lgb-model-wo-qtyext (1).pkl
"""

import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import time
import os
import sys


def load_predictions_and_metadata():
    """Load model, processed test features, and test metadata"""

    print("\n" + "="*70)
    print("LOADING MODEL AND TEST FEATURES")
    print("="*70)

    # ------------------------------
    # 1️⃣ Load processed test features (X_test)
    # ------------------------------
    test_feature_files = [
        "data/test_features.npy",
        "data/test_processed.npy",
        "test_processed.npy",
        "X_test.npy",
        "features/test_features.npy"
    ]

    X_test = None
    for f in test_feature_files:
        if os.path.exists(f):
            X_test = np.load(f)
            print(f"✓ Processed test features loaded: {f}")
            print(f"   Shape: {X_test.shape}")
            break

    if X_test is None:
        print("❌ ERROR: Processed test feature file not found.")
        print("   You need to generate X_test using the same preprocessing as training.")
        print("   Try running your feature engineering script first.")
        sys.exit(1)

    # ------------------------------
    # 2️⃣ Load model
    # ------------------------------
    try:
        model = joblib.load("last-lgb-model-wo-qtyext (1).pkl")
        print("✓ Model loaded: last-lgb-model-wo-qtyext (1).pkl")
    except FileNotFoundError:
        print("❌ Model file not found! Expected: last-lgb-model-wo-qtyext (1).pkl")
        sys.exit(1)

    # ------------------------------
    # 3️⃣ Load test metadata (sample_id etc.)
    # ------------------------------
    test_meta_files = ["data/test.csv", "test.csv", "data/test_cleaned.csv"]
    test_df = None
    for f in test_meta_files:
        if os.path.exists(f):
            test_df = pd.read_csv(f)
            print(f"✓ Test metadata loaded from: {f}")
            break

    if test_df is None:
        print("❌ Test CSV not found! Tried:")
        for f in test_meta_files:
            print(f"   - {f}")
        sys.exit(1)

    # ------------------------------
    # 4️⃣ Generate predictions
    # ------------------------------
    print(f"\n📊 Generating predictions using LightGBM model...")
    y_pred = model.predict(X_test)
    print(f"✓ Predictions generated: {len(y_pred):,} samples")
    print(f"   Expected SMAPE: ~51.39")

    return y_pred, test_df


def analyze_predictions(y_pred):
    """Analyze prediction statistics"""
    print("\n" + "="*70)
    print("PREDICTION ANALYSIS")
    print("="*70)

    print(f"\nCount: {len(y_pred):,}")
    print(f"Min: ${y_pred.min():.2f}")
    print(f"Max: ${y_pred.max():.2f}")
    print(f"Mean: ${y_pred.mean():.2f}")
    print(f"Median: ${np.median(y_pred):.2f}")
    print(f"Std: ${y_pred.std():.2f}")


def create_submission_file(test_df, y_pred, output_path="test_out.csv"):
    """Create final submission CSV"""

    print("\n" + "="*70)
    print("CREATING SUBMISSION FILE")
    print("="*70)

    # Detect ID column
    possible_ids = ["sample_id", "id", "ID", "product_id", "PRODUCT_ID"]
    id_col = None
    for col in possible_ids:
        if col in test_df.columns:
            id_col = col
            break

    if id_col is None:
        print("⚠️ No ID column found, creating sequential IDs...")
        test_df["sample_id"] = range(len(test_df))
        id_col = "sample_id"

    submission = pd.DataFrame({
        "sample_id": test_df[id_col],
        "price": y_pred
    })

    # Save
    submission.to_csv(output_path, index=False)
    print(f"✓ Saved submission file: {output_path}")
    print(submission.head())

    return submission


def validate_submission(submission):
    """Basic submission file validation"""
    print("\n" + "="*70)
    print("VALIDATING SUBMISSION FILE")
    print("="*70)

    ok = True

    if list(submission.columns) != ["sample_id", "price"]:
        print("❌ Wrong column names. Should be ['sample_id', 'price']")
        ok = False
    if submission.isnull().sum().sum() > 0:
        print("❌ Missing values found")
        ok = False
    if (submission["price"] < 0).any():
        print("❌ Negative prices found")
        ok = False

    print("✅ Validation done." if ok else "⚠️ Validation failed.")
    return ok


def main():
    print("\n" + "="*70)
    print("FINAL SUBMISSION GENERATOR")
    print("Model: last-lgb-model-wo-qtyext (1).pkl")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start = time.time()

    y_pred, test_df = load_predictions_and_metadata()
    analyze_predictions(y_pred)
    submission = create_submission_file(test_df, y_pred, output_path="test_out.csv")
    validate_submission(submission)

    print("\n" + "="*70)
    print("✅ SUBMISSION READY!")
    print("="*70)
    print(f"Total time: {time.time() - start:.2f}s")
    print(f"Output: test_out.csv")
    print(f"Rows: {len(submission):,}")
    print(f"Price range: ${y_pred.min():.2f} - ${y_pred.max():.2f}")
    print("🎯 You can now upload this to the competition platform.")


if __name__ == "__main__":
    main()
