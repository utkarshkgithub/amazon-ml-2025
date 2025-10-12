"""
Generate Final Submission File: test_out.csv
Uses trained LightGBM model to predict prices for test dataset
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime

def load_model_and_features():
    """Load trained model and test features"""
    
    print("\n" + "="*70)
    print("LOADING MODEL AND FEATURES")
    print("="*70)
    
    # Load trained model
    print("\n📦 Loading trained LightGBM model...")
    try:
        with open('models/lightgbm_log_final.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded: {type(model).__name__}")
    except FileNotFoundError:
        print("❌ Model file not found!")
        print("   Expected: models/lightgbm_log_final.pkl")
        print("   Please run train_enhanced.py first!")
        exit(1)
    
    # Load test features
    print("\n📦 Loading test features...")
    try:
        X_test = np.load('data/processed/test_features_final.npy')
        print(f"✓ Test features loaded: {X_test.shape}")
    except FileNotFoundError:
        print("❌ Test features not found!")
        print("   Expected: data/processed/test_features_final.npy")
        print("   Please run integrate_features.py first!")
        exit(1)
    
    # Load test metadata (for sample_ids)
    print("\n📦 Loading test metadata...")
    try:
        test_df = pd.read_csv('data/test_cleaned.csv')
        print(f"✓ Test metadata loaded: {len(test_df):,} samples")
    except FileNotFoundError:
        print("❌ test_cleaned.csv not found!")
        print("   Please check data/test_cleaned.csv exists")
        exit(1)
    
    return model, X_test, test_df


def generate_predictions(model, X_test):
    """Generate price predictions"""
    
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    
    print(f"\n🔮 Predicting prices for {X_test.shape[0]:,} test samples...")
    
    # Predict on log scale
    y_test_log = model.predict(X_test)
    
    print(f"✓ Log predictions generated")
    print(f"   Log range: {y_test_log.min():.3f} to {y_test_log.max():.3f}")
    
    # Transform back to original scale
    print(f"\n🔄 Transforming predictions to original scale...")
    y_test_pred = np.expm1(y_test_log)  # exp(log) - 1
    
    # Clip to reasonable range (no negative prices)
    y_test_pred = np.clip(y_test_pred, 0.01, None)
    
    print(f"✓ Predictions transformed")
    
    return y_test_pred


def analyze_predictions(y_pred):
    """Analyze prediction statistics"""
    
    print("\n" + "="*70)
    print("PREDICTION ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Prediction Statistics:")
    print(f"   Count:      {len(y_pred):>10,}")
    print(f"   Minimum:    ${y_pred.min():>10.2f}")
    print(f"   Maximum:    ${y_pred.max():>10.2f}")
    print(f"   Mean:       ${y_pred.mean():>10.2f}")
    print(f"   Median:     ${np.median(y_pred):>10.2f}")
    print(f"   Std Dev:    ${y_pred.std():>10.2f}")
    
    print(f"\n📊 Price Distribution:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(y_pred, p)
        print(f"   {p:>2}th percentile: ${val:>8.2f}")
    
    print(f"\n📊 Price Range Buckets:")
    ranges = [
        (0, 10, "Very Low"),
        (10, 20, "Low"),
        (20, 50, "Medium"),
        (50, 100, "High"),
        (100, float('inf'), "Very High")
    ]
    
    for lower, upper, label in ranges:
        count = ((y_pred >= lower) & (y_pred < upper)).sum()
        pct = (count / len(y_pred)) * 100
        print(f"   ${lower:>3}-{upper:>3} ({label:>10}): {count:>6,} ({pct:>5.1f}%)")


def create_submission_file(test_df, y_pred, output_path='test_out.csv'):
    """Create final submission file"""
    
    print("\n" + "="*70)
    print("CREATING SUBMISSION FILE")
    print("="*70)
    
    # Verify sample_id column exists
    if 'sample_id' not in test_df.columns:
        print("❌ Error: 'sample_id' column not found in test data!")
        print(f"   Available columns: {list(test_df.columns)}")
        exit(1)
    
    # Create submission dataframe
    print(f"\n📝 Creating submission dataframe...")
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': y_pred
    })
    
    print(f"✓ Submission dataframe created: {submission.shape}")
    
    # Verify no missing values
    missing = submission.isnull().sum()
    if missing.any():
        print(f"⚠️  Warning: Missing values detected!")
        print(missing[missing > 0])
    else:
        print(f"✓ No missing values")
    
    # Save to CSV
    print(f"\n💾 Saving to {output_path}...")
    submission.to_csv(output_path, index=False)
    print(f"✓ File saved successfully!")
    
    # Show sample
    print(f"\n📋 Sample of submission file (first 10 rows):")
    print(submission.head(10).to_string(index=False))
    
    print(f"\n📋 Sample of submission file (last 10 rows):")
    print(submission.tail(10).to_string(index=False))
    
    return submission


def validate_submission(submission):
    """Validate submission file format"""
    
    print("\n" + "="*70)
    print("VALIDATING SUBMISSION")
    print("="*70)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Correct columns
    print(f"\n✓ Check 1: Column names")
    expected_cols = ['sample_id', 'price']
    if list(submission.columns) == expected_cols:
        print(f"   ✅ PASS - Columns: {expected_cols}")
        checks_passed += 1
    else:
        print(f"   ❌ FAIL - Expected {expected_cols}, got {list(submission.columns)}")
    
    # Check 2: No missing values
    print(f"\n✓ Check 2: Missing values")
    missing_count = submission.isnull().sum().sum()
    if missing_count == 0:
        print(f"   ✅ PASS - No missing values")
        checks_passed += 1
    else:
        print(f"   ❌ FAIL - Found {missing_count} missing values")
    
    # Check 3: Positive prices
    print(f"\n✓ Check 3: Price range")
    negative_count = (submission['price'] < 0).sum()
    if negative_count == 0:
        print(f"   ✅ PASS - All prices are positive")
        checks_passed += 1
    else:
        print(f"   ❌ FAIL - Found {negative_count} negative prices")
    
    # Check 4: Reasonable price range
    print(f"\n✓ Check 4: Reasonable range")
    min_price = submission['price'].min()
    max_price = submission['price'].max()
    if 0.01 <= min_price and max_price <= 10000:
        print(f"   ✅ PASS - Range ${min_price:.2f} to ${max_price:.2f}")
        checks_passed += 1
    else:
        print(f"   ⚠️  WARNING - Range might be unusual: ${min_price:.2f} to ${max_price:.2f}")
        checks_passed += 0.5
    
    # Check 5: Correct number of rows
    print(f"\n✓ Check 5: Row count")
    expected_rows = 75000  # Adjust if your test set is different
    actual_rows = len(submission)
    if actual_rows == expected_rows:
        print(f"   ✅ PASS - Correct row count: {actual_rows:,}")
        checks_passed += 1
    else:
        print(f"   ⚠️  WARNING - Expected {expected_rows:,}, got {actual_rows:,}")
        checks_passed += 0.5
    
    # Summary
    print(f"\n" + "="*70)
    print(f"VALIDATION SUMMARY: {checks_passed}/{total_checks} checks passed")
    print("="*70)
    
    if checks_passed >= 4.5:
        print(f"✅ Submission file is ready for upload!")
    elif checks_passed >= 3:
        print(f"⚠️  Submission has warnings but may be acceptable")
    else:
        print(f"❌ Submission has critical issues - please fix!")
    
    return checks_passed >= 3


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("FINAL SUBMISSION GENERATOR")
    print("LightGBM Log-Transformed Model")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    import time
    start_time = time.time()
    
    # Step 1: Load model and features
    model, X_test, test_df = load_model_and_features()
    
    # Step 2: Generate predictions
    y_pred = generate_predictions(model, X_test)
    
    # Step 3: Analyze predictions
    analyze_predictions(y_pred)
    
    # Step 4: Create submission file
    submission = create_submission_file(test_df, y_pred, output_path='test_out.csv')
    
    # Step 5: Validate submission
    valid = validate_submission(submission)
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ SUBMISSION GENERATION COMPLETE!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed_time:.2f} seconds\n")
    
    print(f"📁 Output file: test_out.csv")
    print(f"📊 Predictions: {len(y_pred):,} samples")
    print(f"💰 Price range: ${y_pred.min():.2f} - ${y_pred.max():.2f}")
    
    if valid:
        print(f"\n🎯 Ready to submit to competition! 🚀")
    else:
        print(f"\n⚠️  Please review warnings before submitting")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Review test_out.csv")
    print(f"   2. Upload to competition platform")
    print(f"   3. Submit Documentation.md alongside predictions\n")


if __name__ == "__main__":
    main()
