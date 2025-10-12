"""
Train XGBoost with Log Transform + Create Ensemble
Combines LightGBM + XGBoost for better SMAPE
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


def calculate_smape(y_true, y_pred):
    """Calculate SMAPE"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape


def load_data():
    """Load integrated features with LOG transformation"""
    print("\n" + "="*70)
    print("LOADING DATA FOR XGBOOST")
    print("="*70)
    
    start = time.time()
    
    X_train_full = np.load('data/processed/train_features_final.npy')
    y_train_full = np.load('data/processed/train_target_final.npy')
    X_test = np.load('data/processed/test_features_final.npy')
    
    print(f"✓ Loaded in {time.time()-start:.2f}s")
    
    # Log transformation
    print("\n🔄 Applying LOG transformation...")
    y_train_full_log = np.log1p(y_train_full)
    
    print(f"   Original range: ${y_train_full.min():.2f} - ${y_train_full.max():.2f}")
    print(f"   Log range:      {y_train_full_log.min():.3f} - {y_train_full_log.max():.3f}")
    
    # Split
    X_train, X_val, y_train, y_val, y_train_log, y_val_log = train_test_split(
        X_train_full, y_train_full, y_train_full_log, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, {X_test.shape[0]:,} test")
    
    return X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test


def train_xgboost_log(X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test):
    """Train XGBoost on log-transformed prices"""
    
    print("\n" + "="*70)
    print("TRAINING XGBOOST (LOG-TRANSFORMED)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    start = time.time()
    
    # XGBoost optimized for log-scale
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    print("📋 XGBoost Configuration:")
    print(f"   Features:       {X_train.shape[1]:,}")
    print(f"   Max iterations: {model.n_estimators}")
    print(f"   Learning rate:  {model.learning_rate}")
    print(f"   Max depth:      {model.max_depth}\n")
    
    print("🚀 Training XGBoost on LOG prices...\n")
    
    # Train on log
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
        verbose=50
    )
    
    print(f"\n✅ XGBoost training completed in {(time.time()-start)/60:.2f} min\n")
    
    # Evaluate
    print("="*70)
    print("XGBOOST VALIDATION RESULTS")
    print("="*70 + "\n")
    
    # Predict and transform back
    train_pred_log = model.predict(X_train)
    val_pred_log = model.predict(X_val)
    
    train_pred = np.expm1(train_pred_log)
    val_pred = np.expm1(val_pred_log)
    
    # Metrics
    train_smape = calculate_smape(y_train, train_pred)
    val_smape = calculate_smape(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"📊 Training SMAPE:   {train_smape:>6.2f}%")
    print(f"📊 Validation SMAPE: {val_smape:>6.2f}% 🎯")
    print(f"📊 Validation MAE:   ${val_mae:>7.2f}")
    print(f"📊 Validation R²:    {val_r2:>8.4f}")
    
    smape_gap = val_smape - train_smape
    print(f"\n🔍 SMAPE Gap: {smape_gap:>5.2f}%")
    
    # Generate predictions
    print("\n" + "="*70)
    print("GENERATING XGBOOST PREDICTIONS")
    print("="*70 + "\n")
    
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.clip(test_pred, 0.01, None)
    
    print(f"✓ Predictions: Min=${test_pred.min():.2f}, Max=${test_pred.max():.2f}, Mean=${test_pred.mean():.2f}")
    
    # Save
    Path('models').mkdir(exist_ok=True)
    
    with open('models/xgboost_log_final.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved: models/xgboost_log_final.pkl")
    
    test_df = pd.read_csv('data/test_cleaned.csv')
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_pred
    })
    submission.to_csv('models/xgboost_log_predictions.csv', index=False)
    print(f"✓ Saved: models/xgboost_log_predictions.csv")
    
    print(f"\n⏱️  XGBoost total time: {(time.time()-start)/60:.2f} minutes")
    
    return model, val_smape, test_pred


def create_ensemble():
    """Create ensemble of LightGBM + XGBoost"""
    
    print("\n" + "="*70)
    print("CREATING ENSEMBLE (LightGBM + XGBoost)")
    print("="*70 + "\n")
    
    # Load individual predictions
    lgb_pred = pd.read_csv('models/lightgbm_log_predictions.csv')['price'].values
    xgb_pred = pd.read_csv('models/xgboost_log_predictions.csv')['price'].values
    
    print(f"✓ Loaded LightGBM predictions (SMAPE: 54.81%)")
    print(f"✓ Loaded XGBoost predictions")
    
    # Simple average (can be weighted later)
    ensemble_pred = (lgb_pred * 0.5 + xgb_pred * 0.5)
    
    # Alternative: weighted average favoring better model
    # ensemble_pred = (lgb_pred * 0.6 + xgb_pred * 0.4)
    
    print(f"\n📊 Ensemble Statistics:")
    print(f"   Min:    ${ensemble_pred.min():>8.2f}")
    print(f"   Max:    ${ensemble_pred.max():>8.2f}")
    print(f"   Mean:   ${ensemble_pred.mean():>8.2f}")
    print(f"   Median: ${np.median(ensemble_pred):>8.2f}")
    
    # Save ensemble
    test_df = pd.read_csv('data/test_cleaned.csv')
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': ensemble_pred
    })
    submission.to_csv('models/ensemble_lgb_xgb_predictions.csv', index=False)
    
    print(f"\n✅ Saved: models/ensemble_lgb_xgb_predictions.csv")
    print(f"\n📈 Expected Ensemble SMAPE: 50-53% (3-5% better than single models)")
    
    return ensemble_pred


def evaluate_on_validation():
    """Evaluate ensemble on validation set"""
    
    print("\n" + "="*70)
    print("EVALUATING ENSEMBLE ON VALIDATION SET")
    print("="*70 + "\n")
    
    # Load data
    X_train_full = np.load('data/processed/train_features_final.npy')
    y_train_full = np.load('data/processed/train_target_final.npy')
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    # Load models
    with open('models/lightgbm_log_final.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    
    with open('models/xgboost_log_final.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    # Predict
    lgb_val_pred_log = lgb_model.predict(X_val)
    xgb_val_pred_log = xgb_model.predict(X_val)
    
    lgb_val_pred = np.expm1(lgb_val_pred_log)
    xgb_val_pred = np.expm1(xgb_val_pred_log)
    
    # Ensemble
    ensemble_val_pred = (lgb_val_pred * 0.5 + xgb_val_pred * 0.5)
    
    # Calculate SMAPE
    lgb_smape = calculate_smape(y_val, lgb_val_pred)
    xgb_smape = calculate_smape(y_val, xgb_val_pred)
    ensemble_smape = calculate_smape(y_val, ensemble_val_pred)
    
    print("📊 Validation SMAPE Comparison:")
    print(f"   LightGBM:  {lgb_smape:>6.2f}%")
    print(f"   XGBoost:   {xgb_smape:>6.2f}%")
    print(f"   Ensemble:  {ensemble_smape:>6.2f}% 🎯")
    
    improvement = min(lgb_smape, xgb_smape) - ensemble_smape
    print(f"\n✨ Ensemble improvement: {improvement:+.2f}% SMAPE")
    
    if ensemble_smape < 50:
        print(f"   🎉 SMAPE < 50% - Competitive for Top 30-40%!")
    elif ensemble_smape < 53:
        print(f"   ✅ SMAPE < 53% - Good submission")
    else:
        print(f"   ⚠️  SMAPE still high, need more tuning")
    
    return ensemble_smape


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XGBOOST TRAINING + ENSEMBLE CREATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    overall_start = time.time()
    
    # Load data
    X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test = load_data()
    
    # Train XGBoost
    xgb_model, xgb_smape, xgb_pred = train_xgboost_log(
        X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test
    )
    
    # Create ensemble
    ensemble_pred = create_ensemble()
    
    # Evaluate ensemble on validation
    final_smape = evaluate_on_validation()
    
    # Summary
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total time: {(time.time()-overall_start)/60:.2f} minutes\n")
    
    print("📁 Generated Files:")
    print("   models/xgboost_log_predictions.csv")
    print("   models/ensemble_lgb_xgb_predictions.csv  ← SUBMIT THIS! 🎯\n")
    
    print(f"🎯 Final Ensemble SMAPE: {final_smape:.2f}%")
    print(f"   Expected Competition Rank: Top 30-40%\n")
    
    print("🚀 Next steps:")
    print("   1. Submit ensemble_lgb_xgb_predictions.csv")
    print("   2. If time permits: Hyperparameter tuning (2-3% gain)")
    print("   3. Feature selection (1-2% gain)\n")
