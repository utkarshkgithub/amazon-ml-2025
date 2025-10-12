"""
Train Enhanced Models with LOG-TRANSFORMED PRICES
Critical fix for extreme price range ($0.13 - $2,796)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


def calculate_smape(y_true, y_pred):
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape


def load_data():
    """Load integrated features with LOG transformation"""
    print("\n" + "="*70)
    print("LOADING INTEGRATED FEATURES + LOG TRANSFORMATION")
    print("="*70)
    
    start = time.time()
    
    print("\nLoading features...")
    X_train_full = np.load('data/processed/train_features_final.npy')
    y_train_full = np.load('data/processed/train_target_final.npy')
    X_test = np.load('data/processed/test_features_final.npy')
    
    print(f"✓ Loaded in {time.time()-start:.2f}s")
    
    # 🔥 LOG TRANSFORMATION - CRITICAL FIX 🔥
    print("\n🔄 Applying LOG transformation to prices...")
    print(f"   Original price range: ${y_train_full.min():.2f} - ${y_train_full.max():.2f}")
    
    y_train_full_log = np.log1p(y_train_full)  # log(1 + price)
    
    print(f"   Log-transformed range: {y_train_full_log.min():.3f} - {y_train_full_log.max():.3f}")
    print(f"   Compression ratio: {(y_train_full.max()-y_train_full.min())/(y_train_full_log.max()-y_train_full_log.min()):.1f}x")
    
    # Split train/validation
    print("\nSplitting train/validation (80/20)...")
    X_train, X_val, y_train, y_val, y_train_log, y_val_log = train_test_split(
        X_train_full, y_train_full, y_train_full_log, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data Summary:")
    print(f"   Training:   {X_train.shape[0]:>6,} samples × {X_train.shape[1]:>5,} features")
    print(f"   Validation: {X_val.shape[0]:>6,} samples × {X_val.shape[1]:>5,} features")
    print(f"   Test:       {X_test.shape[0]:>6,} samples × {X_test.shape[1]:>5,} features")
    
    return X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test


def train_lightgbm_log(X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test):
    """Train LightGBM on LOG-TRANSFORMED prices"""
    
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM (LOG-TRANSFORMED PRICES)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    start = time.time()
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=50,
        min_split_gain=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
        metric='mae'
    )
    
    print("📋 Model Configuration:")
    print(f"   Features:           {X_train.shape[1]:,}")
    print(f"   Target:             LOG(price) - Handles extreme range!")
    print(f"   Max iterations:     {model.n_estimators}")
    print(f"   Learning rate:      {model.learning_rate}\n")
    
    print("🚀 Training on LOG-TRANSFORMED prices...\n")
    
    # 🔥 TRAIN ON LOG PRICES 🔥
    model.fit(
        X_train, y_train_log,  # ← LOG TARGET
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
        eval_metric='mae',
        eval_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50)
        ]
    )
    
    best_iteration = model.best_iteration_
    print(f"\n✅ Training completed!")
    print(f"   Best iteration: {best_iteration}")
    print(f"   Training time:  {(time.time()-start)/60:.2f} minutes\n")
    
    # Evaluate
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70 + "\n")
    
    # 🔥 PREDICT LOG, THEN TRANSFORM BACK 🔥
    train_pred_log = model.predict(X_train)
    val_pred_log = model.predict(X_val)
    
    # Transform back to original price scale
    train_pred = np.expm1(train_pred_log)  # exp(log) - 1 = original
    val_pred = np.expm1(val_pred_log)
    
    # Metrics on ORIGINAL scale
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    train_smape = calculate_smape(y_train, train_pred)
    
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    val_smape = calculate_smape(y_val, val_pred)
    
    # Metrics on LOG scale (what model optimized)
    val_r2_log = r2_score(y_val_log, val_pred_log)
    
    print("📊 Training Set (Original $ Scale):")
    print(f"   MAE:   ${train_mae:>8.2f}")
    print(f"   RMSE:  ${train_rmse:>8.2f}")
    print(f"   R²:    {train_r2:>9.4f}")
    print(f"   SMAPE: {train_smape:>8.2f}%")
    
    print("\n📊 Validation Set (Original $ Scale):")
    print(f"   MAE:   ${val_mae:>8.2f}")
    print(f"   RMSE:  ${val_rmse:>8.2f}")
    print(f"   R²:    {val_r2:>9.4f}")
    print(f"   SMAPE: {val_smape:>8.2f}% 🎯")
    
    print(f"\n📊 Validation (Log Scale - What Model Optimized):")
    print(f"   R² (log): {val_r2_log:>7.4f}")
    
    # Overfitting analysis
    mae_gap = ((train_mae - val_mae) / val_mae) * 100
    r2_gap = train_r2 - val_r2
    smape_gap = val_smape - train_smape
    
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   MAE Gap:   {abs(mae_gap):>6.1f}%")
    print(f"   R² Gap:    {abs(r2_gap):>7.3f}")
    print(f"   SMAPE Gap: {smape_gap:>6.2f}%")
    
    if abs(mae_gap) < 25:
        print(f"   Status: ✅ GOOD generalization")
    elif abs(mae_gap) < 35:
        print(f"   Status: ⚠️  Some overfitting")
    else:
        print(f"   Status: ❌ Severe overfitting")
    
    # SMAPE Assessment
    print(f"\n📈 SMAPE Assessment:")
    if val_smape < 30:
        print(f"   ⭐⭐⭐ EXCELLENT ({val_smape:.1f}%) - Top 10-20%!")
    elif val_smape < 35:
        print(f"   ⭐⭐ VERY GOOD ({val_smape:.1f}%) - Top 20-30%")
    elif val_smape < 40:
        print(f"   ⭐ GOOD ({val_smape:.1f}%) - Top 30-40%")
    elif val_smape < 45:
        print(f"   ✓ DECENT ({val_smape:.1f}%) - Top 40-50%")
    else:
        print(f"   ⚠️  NEEDS MORE WORK ({val_smape:.1f}%)")
    
    # Improvement comparison
    print(f"\n📈 Improvement vs Linear Scale:")
    print(f"   Previous SMAPE (linear): 61.16%")
    print(f"   Current SMAPE (log):     {val_smape:.2f}%")
    if val_smape < 61:
        print(f"   Improvement: {((61.16 - val_smape) / 61.16 * 100):.1f}% better! 🎉")
    
    # Generate predictions
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70 + "\n")
    
    # 🔥 PREDICT LOG, TRANSFORM BACK 🔥
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.clip(test_pred, 0.01, None)
    
    print(f"✓ Generated {len(test_pred):,} predictions")
    print(f"\n📈 Prediction Statistics:")
    print(f"   Min:    ${test_pred.min():>8.2f}")
    print(f"   Max:    ${test_pred.max():>8.2f}")
    print(f"   Mean:   ${test_pred.mean():>8.2f}")
    print(f"   Median: ${np.median(test_pred):>8.2f}")
    
    # Save outputs
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70 + "\n")
    
    Path('models').mkdir(exist_ok=True)
    
    model_path = 'models/lightgbm_log_final.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model: {model_path}")
    
    test_df = pd.read_csv('data/test_cleaned.csv')
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_pred
    })
    pred_path = 'models/lightgbm_log_predictions.csv'
    submission.to_csv(pred_path, index=False)
    print(f"✓ Predictions: {pred_path}")
    
    summary = {
        'model': 'LightGBM_Log',
        'features': X_train.shape[1],
        'train_smape': train_smape,
        'val_smape': val_smape,
        'val_r2': val_r2,
        'val_r2_log': val_r2_log,
        'mae_gap_%': abs(mae_gap),
        'improvement_vs_linear_%': (61.16 - val_smape) / 61.16 * 100,
        'training_time_min': (time.time() - start) / 60
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('models/lightgbm_log_summary.csv', index=False)
    
    print(f"\n⏱️  Total time: {(time.time() - start)/60:.2f} minutes")
    
    return model, val_smape, val_r2


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LOG-TRANSFORMED ENHANCED MODEL")
    print("Critical Fix: Handles $0.13 - $2,796 price range")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    overall_start = time.time()
    
    # Load with log transformation
    X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test = load_data()
    
    # Train
    model, smape, r2 = train_lightgbm_log(X_train, X_val, y_train, y_val, 
                                          y_train_log, y_val_log, X_test)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total time: {(time.time()-overall_start)/60:.2f} minutes\n")
    
    print("📁 Output Files:")
    print("   models/lightgbm_log_predictions.csv  ← SUBMIT THIS!")
    print("   models/lightgbm_log_summary.csv\n")
    
    print(f"🎯 Final SMAPE: {smape:.2f}%")
    print(f"   Expected Rank: Top 30-40%\n")
