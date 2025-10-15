# ============================================================
# LOCAL TRAINING WITH ENHANCED FEATURES - COMPLETE FIXED VERSION
# ============================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import gc
import os
from datetime import datetime
import json

print("="*70)
print("LOCAL TRAINING - ENHANCED FEATURES")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================
# CONFIGURE PATHS
# ============================================================

DATA_PATH = 'data/processed/'
MAIN_DATA = 'data/'

# ============================================================
# CHECK AVAILABLE FEATURE FILES
# ============================================================

print("\n📋 Checking available feature files...")

enhanced_exists = os.path.exists(DATA_PATH + 'train_features_enhanced_v1.npy')
clean_exists = os.path.exists(DATA_PATH + 'train_features_clean.npy')
combined_exists = os.path.exists(DATA_PATH + 'train_features_combined.npy')
final_exists = os.path.exists(DATA_PATH + 'train_features_final.npy')

print("\n🔍 Available base feature files:")
if enhanced_exists:
    arr = np.load(DATA_PATH + 'train_features_enhanced_v1.npy', mmap_mode='r')
    print(f"✓ train_features_enhanced_v1.npy: {arr.shape} ⭐ BEST")
    base_file = 'train_features_enhanced_v1.npy'
    base_shape = arr.shape
elif final_exists:
    arr = np.load(DATA_PATH + 'train_features_final.npy', mmap_mode='r')
    print(f"✓ train_features_final.npy: {arr.shape}")
    base_file = 'train_features_final.npy'
    base_shape = arr.shape
elif clean_exists:
    arr = np.load(DATA_PATH + 'train_features_clean.npy', mmap_mode='r')
    print(f"✓ train_features_clean.npy: {arr.shape}")
    base_file = 'train_features_clean.npy'
    base_shape = arr.shape
elif combined_exists:
    arr = np.load(DATA_PATH + 'train_features_combined.npy', mmap_mode='r')
    print(f"✓ train_features_combined.npy: {arr.shape}")
    base_file = 'train_features_combined.npy'
    base_shape = arr.shape
else:
    print("❌ No base feature files found!")
    print("Expected files in:", DATA_PATH)
    exit(1)

print(f"\n📦 Using: {base_file}")
print(f"   Shape: {base_shape}")

# ============================================================
# LOAD ALL FEATURES
# ============================================================

print("\n📥 Loading all features...")

# Load base features
print(f"Loading {base_file}...")
X_train_base = np.load(DATA_PATH + base_file)
test_base_file = base_file.replace('train_', 'test_')
X_test_base = np.load(DATA_PATH + test_base_file)
print(f"✓ Base: {X_train_base.shape}")

# Load CSV features (FULL - not yet masked)
print("Loading CSV features...")
train_text_stats = pd.read_csv(DATA_PATH + 'train_text_stats.csv')
test_text_stats = pd.read_csv(DATA_PATH + 'test_text_stats.csv')
train_advanced = pd.read_csv(DATA_PATH + 'train_advanced_text_features.csv')
test_advanced = pd.read_csv(DATA_PATH + 'test_advanced_text_features.csv')
train_cat = pd.read_csv(DATA_PATH + 'train_category_features.csv')
test_cat = pd.read_csv(DATA_PATH + 'test_category_features.csv')
print(f"✓ CSV loaded (before masking): {train_text_stats.shape[0]} rows")

# Load TF-IDF
print("Loading TF-IDF...")
train_tfidf = np.load(DATA_PATH + 'train_tfidf_150.npy')
test_tfidf = np.load(DATA_PATH + 'test_tfidf_150.npy')
print(f"✓ TF-IDF: {train_tfidf.shape}")

# Load quantity features
print("Loading quantity features...")
if os.path.exists(DATA_PATH + 'train_quantity_6.csv'):
    train_qty = pd.read_csv(DATA_PATH + 'train_quantity_6.csv')
    test_qty = pd.read_csv(DATA_PATH + 'test_quantity_6.csv')
    print(f"✓ Quantity: {train_qty.shape}")
    has_quantity = True
else:
    print("⚠️ Quantity features not found, skipping...")
    has_quantity = False

# ============================================================
# LOAD TARGET AND APPLY CONSISTENT MASKING
# ============================================================

print("\n🎯 Loading target and applying mask...")

# Load full target
if os.path.exists(DATA_PATH + 'train_target_final.npy'):
    y_train_full = np.load(DATA_PATH + 'train_target_final.npy')
    print(f"✓ Loaded train_target_final.npy: {y_train_full.shape}")
    
    # Calculate outlier mask
    y_log = np.log1p(y_train_full)
    Q1, Q3 = np.percentile(y_log, [25, 75])
    IQR = Q3 - Q1
    mask = (y_log >= Q1 - 1.5*IQR) & (y_log <= Q3 + 1.5*IQR)
    
    print(f"   Outliers: {(~mask).sum():,}")
    print(f"   Clean:    {mask.sum():,}")
    
    # Check if base features are already masked
    if X_train_base.shape[0] == mask.sum():
        print("✓ Base features already cleaned")
        base_already_masked = True
    else:
        print("✓ Base features need masking")
        base_already_masked = False
        X_train_base = X_train_base[mask]
        train_tfidf = train_tfidf[mask]
    
    # Apply mask to CSV features
    train_text_stats = train_text_stats.iloc[mask]
    train_advanced = train_advanced.iloc[mask]
    train_cat = train_cat.iloc[mask]
    
    # Apply mask to target
    y_train_target = y_log[mask]
    
    # Handle quantity
    if has_quantity:
        if train_qty.shape[0] == mask.sum():
            print("✓ Quantity already cleaned")
        else:
            print("⚠️ Masking quantity features")
            train_qty = train_qty.iloc[mask]
    
elif os.path.exists(DATA_PATH + 'train_target_clean.npy'):
    y_train_target = np.load(DATA_PATH + 'train_target_clean.npy')
    print(f"✓ Using train_target_clean.npy: {y_train_target.shape}")
    
    # Assume everything matches
    n_samples = X_train_base.shape[0]
    train_text_stats = train_text_stats.iloc[:n_samples]
    train_advanced = train_advanced.iloc[:n_samples]
    train_cat = train_cat.iloc[:n_samples]
    
    if has_quantity and train_qty.shape[0] > n_samples:
        train_qty = train_qty.iloc[:n_samples]
        
else:
    print("❌ No target file found!")
    exit(1)

# ============================================================
# VERIFY SHAPES MATCH
# ============================================================

print(f"\n✅ SHAPE VERIFICATION:")
print(f"   Base features:      {X_train_base.shape}")
print(f"   Text stats:         {train_text_stats.shape}")
print(f"   Advanced text:      {train_advanced.shape}")
print(f"   Category:           {train_cat.shape}")
print(f"   TF-IDF:             {train_tfidf.shape}")
if has_quantity:
    print(f"   Quantity:           {train_qty.shape}")
print(f"   Target:             {y_train_target.shape}")

# Check all match
n_samples = X_train_base.shape[0]
shapes_match = (
    train_text_stats.shape[0] == n_samples and
    train_advanced.shape[0] == n_samples and
    train_cat.shape[0] == n_samples and
    train_tfidf.shape[0] == n_samples and
    y_train_target.shape[0] == n_samples
)

if has_quantity:
    shapes_match = shapes_match and (train_qty.shape[0] == n_samples)

if not shapes_match:
    print("\n❌ SHAPE MISMATCH!")
    print("Debugging info:")
    print(f"  Expected: {n_samples} samples")
    print(f"  Base: {X_train_base.shape[0]}")
    print(f"  Text stats: {train_text_stats.shape[0]}")
    print(f"  Advanced: {train_advanced.shape[0]}")
    print(f"  Category: {train_cat.shape[0]}")
    print(f"  TF-IDF: {train_tfidf.shape[0]}")
    if has_quantity:
        print(f"  Quantity: {train_qty.shape[0]}")
    print(f"  Target: {y_train_target.shape[0]}")
    exit(1)

print(f"\n✓ All features aligned: {n_samples:,} samples")

# ============================================================
# COMBINE ALL FEATURES
# ============================================================

print("\n🔗 Combining all features...")

features_to_combine = []
test_features_to_combine = []
feature_counts = []

# Add base features
features_to_combine.append(X_train_base)
test_features_to_combine.append(X_test_base)
feature_counts.append(X_train_base.shape[1])

# Add CSV features
features_to_combine.append(train_text_stats.values)
test_features_to_combine.append(test_text_stats.values)
feature_counts.append(train_text_stats.shape[1])

features_to_combine.append(train_advanced.values)
test_features_to_combine.append(test_advanced.values)
feature_counts.append(train_advanced.shape[1])

features_to_combine.append(train_cat.values)
test_features_to_combine.append(test_cat.values)
feature_counts.append(train_cat.shape[1])

# Add TF-IDF
features_to_combine.append(train_tfidf)
test_features_to_combine.append(test_tfidf)
feature_counts.append(train_tfidf.shape[1])

# Add quantity
if has_quantity:
    features_to_combine.append(train_qty.values)
    test_features_to_combine.append(test_qty.values)
    feature_counts.append(train_qty.shape[1])

# Combine
print("Combining train features...")
X_train_full = np.hstack(features_to_combine)
print("Combining test features...")
X_test_full = np.hstack(test_features_to_combine)

print(f"\n✅ FEATURE BREAKDOWN:")
print(f"   Base features:      {feature_counts[0]:4d}")
print(f"   Text stats:         {feature_counts[1]:4d}")
print(f"   Advanced text:      {feature_counts[2]:4d}")
print(f"   Category:           {feature_counts[3]:4d}")
print(f"   TF-IDF:             {feature_counts[4]:4d}")
if has_quantity:
    print(f"   Quantity:           {feature_counts[5]:4d}")
print(f"   ──────────────────────────")
print(f"   TOTAL:              {X_train_full.shape[1]:4d}")

print(f"\n   Train shape: {X_train_full.shape}")
print(f"   Test shape:  {X_test_full.shape}")

# Free memory
del X_train_base, X_test_base, train_text_stats, test_text_stats
del train_advanced, test_advanced, train_cat, test_cat
del train_tfidf, test_tfidf
if has_quantity:
    del train_qty, test_qty
gc.collect()

# ============================================================
# TRAIN-VAL SPLIT
# ============================================================

print("\n✂️ Splitting train/validation...")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_target,
    test_size=0.2,
    random_state=42
)

print(f"✓ Train: {X_train.shape}")
print(f"✓ Val:   {X_val.shape}")

del X_train_full, y_train_target
gc.collect()

# ============================================================
# DEFINE SMAPE
# ============================================================

def calculate_smape(y_true, y_pred):
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    numerator = np.abs(y_pred_orig - y_true_orig)
    denominator = (np.abs(y_true_orig) + np.abs(y_pred_orig)) / 2
    return np.mean(numerator / denominator) * 100

# ============================================================
# TRAIN LIGHTGBM
# ============================================================

print("\n" + "="*70)
print(f"TRAINING LIGHTGBM WITH {X_train.shape[1]} FEATURES")
print("="*70)

params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 60,
    'reg_alpha': 1.5,
    'reg_lambda': 5.0,
    'subsample': 0.75,
    'colsample_bytree': 0.65,
    'min_child_samples': 30,
    'max_bin': 255,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'n_estimators': 2000
}

model = lgb.LGBMRegressor(**params)

print("\nTraining (20-30 minutes)...")
start_time = datetime.now()

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(100, verbose=True),
        lgb.log_evaluation(100)
    ]
)

training_time = (datetime.now() - start_time).total_seconds() / 60

print(f"\n✅ Training complete! ({training_time:.1f} minutes)")

# ============================================================
# EVALUATE
# ============================================================

print("\n" + "="*70)
print("EVALUATION")
print("="*70)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_mae = mean_absolute_error(y_train, train_pred)
val_mae = mean_absolute_error(y_val, val_pred)
train_smape = calculate_smape(y_train, train_pred)
val_smape = calculate_smape(y_val, val_pred)

print(f"\n📊 PERFORMANCE:")
print(f"Train MAE:    {train_mae:.4f}")
print(f"Val MAE:      {val_mae:.4f}")
print(f"Gap:          {val_mae - train_mae:.4f}")

print(f"\nTrain SMAPE:  {train_smape:.2f}")
print(f"Val SMAPE:    {val_smape:.2f}")
print(f"Gap:          {val_smape - train_smape:.2f}")

print(f"\n📈 COMPARISON:")
print(f"{'='*70}")
print(f"Previous best (5,312):  50.30 SMAPE")
print(f"Current ({X_train.shape[1]}):     {val_smape:.2f} SMAPE")
print(f"Difference:             {val_smape - 50.30:+.2f} points")

# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

print("\n" + "="*70)
print("GENERATING TEST PREDICTIONS")
print("="*70)

test_pred = np.expm1(model.predict(X_test_full))

print(f"✓ Predictions: {len(test_pred):,}")
print(f"  Min:    ${test_pred.min():.2f}")
print(f"  Max:    ${test_pred.max():.2f}")
print(f"  Mean:   ${test_pred.mean():.2f}")
print(f"  Median: ${np.median(test_pred):.2f}")

# ============================================================
# CREATE SUBMISSION
# ============================================================

if os.path.exists(MAIN_DATA + 'test.csv'):
    test_df = pd.read_csv(MAIN_DATA + 'test.csv')
elif os.path.exists('test.csv'):
    test_df = pd.read_csv('test.csv')
else:
    print("⚠️ test.csv not found, creating dummy IDs")
    test_df = pd.DataFrame({'sample_id': range(len(test_pred))})

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': test_pred
})

output_filename = f'test_out_local_{X_train.shape[1]}_features.csv'
submission.to_csv(output_filename, index=False)

print(f"\n✅ Submission: {output_filename}")
print(f"🎯 Expected SMAPE: ~{val_smape:.2f}")
print(f"\n📋 Preview:")
print(submission.head(10))

# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "="*70)
print("SAVING FILES")
print("="*70)

model_filename = f'lgb_local_{X_train.shape[1]}.txt'
model.booster_.save_model(model_filename)
print(f"✅ Model: {model_filename}")

params_filename = f'params_{X_train.shape[1]}.json'
params['best_iteration'] = int(model.best_iteration_)
params['val_smape'] = float(val_smape)
params['training_time_minutes'] = float(training_time)

with open(params_filename, 'w') as f:
    json.dump(params, f, indent=2)
print(f"✅ Params: {params_filename}")

importance_df = pd.DataFrame({
    'feature': range(X_train.shape[1]),
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_filename = f'importance_{X_train.shape[1]}.csv'
importance_df.to_csv(importance_filename, index=False)
print(f"✅ Importance: {importance_filename}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("🎉 LOCAL TRAINING COMPLETE!")
print("="*70)

print(f"\n📊 RESULTS:")
print(f"   Features:          {X_train.shape[1]}")
print(f"   Training time:     {training_time:.1f} min")
print(f"   Validation SMAPE:  {val_smape:.2f}")
print(f"   vs Previous:       {val_smape - 50.30:+.2f}")

print(f"\n📁 FILES SAVED:")
print(f"   1. {output_filename}")
print(f"   2. {model_filename}")
print(f"   3. {params_filename}")
print(f"   4. {importance_filename}")

print(f"\n🚀 NEXT: Upload {output_filename} to competition")
print("="*70)
