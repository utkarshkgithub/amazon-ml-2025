"""
Train LightGBM with LOG-TRANSFORMED PRICES
Optimized for extreme price ranges ($0.13 - $2,796)
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


def load_data():
    """Load preprocessed features with log transformation"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
    load_start = time.time()
    
    print("Loading training features...")
    X_train_full = np.load('/home/sushi/amazon-ml-2025/data/processed/train_features_combined.npy')
    print(f"✓ Loaded in {time.time()-load_start:.2f}s")
    
    print("Loading training targets...")
    y_train_full = np.load('/home/sushi/amazon-ml-2025/data/processed/train_target.npy')
    print(f"✓ Loaded in {time.time()-load_start:.2f}s")
    
    print("Loading test features...")
    X_test = np.load('/home/sushi/amazon-ml-2025/data/processed/test_features_combined.npy')
    print(f"✓ Loaded in {time.time()-load_start:.2f}s")
    
    # Split
    print("\nSplitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data Shape Summary:")
    print(f"   Training:   {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")
    print(f"   Validation: {X_val.shape[0]:,} samples × {X_val.shape[1]:,} features")
    print(f"   Test:       {X_test.shape[0]:,} samples × {X_test.shape[1]:,} features")
    
    print(f"\n💰 Target Statistics (Original Scale):")
    print(f"   Min:    ${y_train_full.min():.2f}")
    print(f"   Max:    ${y_train_full.max():.2f}")
    print(f"   Mean:   ${y_train_full.mean():.2f}")
    print(f"   Median: ${np.median(y_train_full):.2f}")
    
    # 🔥 LOG TRANSFORMATION 🔥
    print(f"\n🔄 Applying LOG transformation...")
    y_train_log = np.log1p(y_train)  # log(1 + price)
    y_val_log = np.log1p(y_val)
    
    print(f"   Log-transformed range: {y_train_log.min():.3f} - {y_train_log.max():.3f}")
    print(f"   Original range:        ${y_train.min():.2f} - ${y_train.max():.2f}")
    print(f"   Compression factor:    {(y_train.max()-y_train.min())/(y_train_log.max()-y_train_log.min()):.1f}x")
    
    total_time = time.time() - load_start
    print(f"\n✅ Data loading completed in {total_time:.2f}s\n")
    
    return X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test


def train_lightgbm_log(X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test):
    """Train LightGBM on LOG-TRANSFORMED prices"""
    
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM (LOG-TRANSFORMED PRICES)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start = time.time()
    
    # Model optimized for log-scale + high dimensions
    model = lgb.LGBMRegressor(
        # Core parameters
        n_estimators=500,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        
        # Sampling
        subsample=0.8,
        colsample_bytree=0.8,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        
        # Leaf parameters
        min_child_samples=50,
        min_split_gain=0.1,
        min_child_weight=0.001,
        
        # Regularization
        reg_alpha=0.5,
        reg_lambda=2.0,
        
        # Technical
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True,
        boosting_type='gbdt'
    )
    
    print("📋 Model Configuration (LOG-SCALE + HIGH-DIMENSIONAL):")
    print(f"   Max iterations:      {model.n_estimators}")
    print(f"   Max depth:           {model.max_depth}")
    print(f"   Num leaves:          {model.num_leaves}")
    print(f"   Learning rate:       {model.learning_rate}")
    print(f"   Feature fraction:    {model.feature_fraction}")
    print(f"   Min child samples:   {model.min_child_samples}")
    print(f"   L1 regularization:   {model.reg_alpha}")
    print(f"   L2 regularization:   {model.reg_lambda}")
    print(f"   Early stopping:      50 rounds\n")
    
    print("🔥 Training on LOG-TRANSFORMED prices...")
    print("   Target: log(1 + price) instead of raw price")
    print("   (Updates every 10 iterations)\n")
    
    # 🔥 TRAIN ON LOG PRICES 🔥
    model.fit(
        X_train, y_train_log,  # ← LOG-TRANSFORMED TARGET
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],  # ← LOG TARGETS
        eval_metric='mae',
        eval_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.early_stopping(stopping_rounds=50)
        ]
    )
    
    # Get best iteration info
    best_iteration = model.best_iteration_
    best_score_log = model.best_score_['valid']['l1']
    
    print(f"\n✅ Training completed!")
    print(f"   Best iteration: {best_iteration}")
    print(f"   Best score (log scale): {best_score_log:.4f}")
    print(f"   Total iterations: {model.n_estimators}")
    print(f"   Early stopped: {'Yes ✓' if best_iteration < model.n_estimators else 'No'}\n")
    
    # Validate
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70 + "\n")
    
    # 🔥 PREDICT LOG PRICES, THEN TRANSFORM BACK 🔥
    val_pred_log = model.predict(X_val)
    train_pred_log = model.predict(X_train)
    
    # Transform back to dollar scale
    val_pred = np.expm1(val_pred_log)  # exp(log_price) - 1 = original price
    train_pred = np.expm1(train_pred_log)
    
    # Training metrics (on original scale)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    # Validation metrics (on original scale)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    
    # Percentage error
    val_mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
    
    # Log-scale metrics (for comparison)
    train_mae_log = mean_absolute_error(y_train_log, train_pred_log)
    val_mae_log = mean_absolute_error(y_val_log, val_pred_log)
    val_r2_log = r2_score(y_val_log, val_pred_log)
    
    print("📊 Training Set (Original $ Scale):")
    print(f"   MAE:  ${train_mae:.2f}")
    print(f"   RMSE: ${train_rmse:.2f}")
    print(f"   R²:   {train_r2:.4f}")
    
    print("\n📊 Validation Set (Original $ Scale):")
    print(f"   MAE:   ${val_mae:.2f}")
    print(f"   RMSE:  ${val_rmse:.2f}")
    print(f"   R²:    {val_r2:.4f}")
    print(f"   MAPE:  {val_mape:.2f}%")
    
    print("\n📊 Validation Set (Log Scale - what model optimized):")
    print(f"   MAE (log):  {val_mae_log:.4f}")
    print(f"   R² (log):   {val_r2_log:.4f}")
    
    # Enhanced overfitting analysis
    overfit_mae = ((train_mae - val_mae) / val_mae) * 100
    overfit_r2 = train_r2 - val_r2
    
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   MAE Gap:  {abs(overfit_mae):.1f}%")
    print(f"   R² Gap:   {abs(overfit_r2):.3f}")
    
    if abs(overfit_mae) < 15 and abs(overfit_r2) < 0.15:
        print(f"   Status: ✅ EXCELLENT - Well generalized!")
    elif abs(overfit_mae) < 30 and abs(overfit_r2) < 0.25:
        print(f"   Status: ✅ GOOD - Acceptable generalization")
    elif abs(overfit_mae) < 50:
        print(f"   Status: ⚠️  WARNING - Some overfitting")
    else:
        print(f"   Status: ❌ SEVERE - Major overfitting")
    
    # Model quality assessment
    print(f"\n📈 Model Quality:")
    if val_r2 > 0.70:
        print(f"   Validation R²: ⭐⭐⭐ EXCELLENT ({val_r2:.3f}) - SUBMIT THIS! 🚀")
    elif val_r2 > 0.55:
        print(f"   Validation R²: ⭐⭐ GOOD ({val_r2:.3f}) - Competitive!")
    elif val_r2 > 0.40:
        print(f"   Validation R²: ⭐ FAIR ({val_r2:.3f}) - Usable")
    else:
        print(f"   Validation R²: ❌ POOR ({val_r2:.3f}) - Still needs work")
    
    # Improvement comparison
    print(f"\n📈 Improvement vs Linear Scale:")
    print(f"   Previous R² (linear): ~0.28")
    print(f"   Current R² (log):     {val_r2:.3f}")
    if val_r2 > 0.28:
        print(f"   Improvement:          {((val_r2 - 0.28) / 0.28 * 100):.0f}% better! 🎉")
    
    # Predict test
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70 + "\n")
    
    pred_start = time.time()
    
    # 🔥 PREDICT LOG, THEN TRANSFORM BACK 🔥
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log)  # Transform back to dollars
    
    # Clip to valid range
    test_pred = np.clip(test_pred, 0.01, None)
    
    pred_time = time.time() - pred_start
    
    print(f"✓ Generated {len(test_pred):,} predictions in {pred_time:.2f}s")
    print(f"   Predictions per second: {len(test_pred)/pred_time:.0f}")
    
    # Prediction statistics
    print(f"\n📈 Prediction Statistics:")
    print(f"   Min:    ${test_pred.min():.2f}")
    print(f"   Max:    ${test_pred.max():.2f}")
    print(f"   Mean:   ${test_pred.mean():.2f}")
    print(f"   Median: ${np.median(test_pred):.2f}")
    
    # Sanity checks
    print(f"\n🔍 Sanity Checks:")
    print(f"   Negative prices: 0 ✅ (clipped)")
    print(f"   Extreme prices (>$1000): {(test_pred > 1000).sum()}")
    print(f"   Price range: ${test_pred.min():.2f} - ${test_pred.max():.2f}")
    
    # Price distribution
    print(f"\n📊 Prediction Distribution:")
    print(f"   <$10:     {(test_pred < 10).sum():,} ({(test_pred < 10).mean()*100:.1f}%)")
    print(f"   $10-$50:  {((test_pred >= 10) & (test_pred < 50)).sum():,} ({((test_pred >= 10) & (test_pred < 50)).mean()*100:.1f}%)")
    print(f"   $50-$100: {((test_pred >= 50) & (test_pred < 100)).sum():,} ({((test_pred >= 50) & (test_pred < 100)).mean()*100:.1f}%)")
    print(f"   >$100:    {(test_pred >= 100).sum():,} ({(test_pred >= 100).mean()*100:.1f}%)")
    
    # Save outputs
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70 + "\n")
    
    Path('models').mkdir(exist_ok=True)
    
    # Save model
    model_path = 'models/lightgbm_log_transformed.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    model_size = Path(model_path).stat().st_size / (1024*1024)
    print(f"✓ Model saved: {model_path} ({model_size:.1f} MB)")
    
    # Save predictions
    test_df = pd.read_csv('/home/sushi/amazon-ml-2025/data/test_cleaned.csv')
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_pred
    })
    pred_path = 'models/lightgbm_log_predictions.csv'
    submission.to_csv(pred_path, index=False)
    print(f"✓ Predictions saved: {pred_path}")
    
    # Save feature importance
    feature_imp = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    imp_path = 'models/lightgbm_log_feature_importance.csv'
    feature_imp.to_csv(imp_path, index=False)
    print(f"✓ Feature importance saved: {imp_path}")
    
    # Show top 10 features
    print(f"\n🏆 Top 10 Most Important Features:")
    for i, row in feature_imp.head(10).iterrows():
        feature_idx = int(row['feature'].split('_')[1])
        feature_type = "Image" if feature_idx < 4288 else \
                      "Text" if feature_idx < 5056 else "Quantity"
        print(f"   {row['feature']:20s} {row['importance']:>8.0f}  ({feature_type})")
    
    # Feature type distribution
    top_100 = feature_imp.head(100)
    image_count = sum(1 for f in top_100['feature'] if int(f.split('_')[1]) < 4288)
    text_count = sum(1 for f in top_100['feature'] if 4288 <= int(f.split('_')[1]) < 5056)
    quant_count = sum(1 for f in top_100['feature'] if int(f.split('_')[1]) >= 5056)
    
    print(f"\n📊 Feature Type Distribution (Top 100):")
    print(f"   Image features:    {image_count}/100 ({image_count}%)")
    print(f"   Text features:     {text_count}/100 ({text_count}%)")
    print(f"   Quantity features: {quant_count}/100 ({quant_count}%)")
    
    # Training summary
    summary = {
        'model': 'LightGBM_LogTransformed',
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_mape': val_mape,
        'val_r2_log_scale': val_r2_log,
        'mae_gap_percent': abs(overfit_mae),
        'r2_gap': abs(overfit_r2),
        'best_iteration': best_iteration,
        'training_time_min': (time.time() - start) / 60
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = 'models/lightgbm_log_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Training summary saved: {summary_path}")
    
    elapsed = (time.time() - start) / 60
    print(f"\n⏱️  Total training time: {elapsed:.2f} minutes")
    
    return model, val_mae, val_rmse, val_r2


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LIGHTGBM - LOG-TRANSFORMED PRICE TRAINING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🔥 KEY INNOVATION: Training on log(price) instead of raw price")
    print("   This handles the extreme price range ($0.13 - $2,796)")
    print("   Expected: R² improvement from ~0.28 to 0.65-0.75\n")
    
    overall_start = time.time()
    
    # Load data (includes log transformation)
    X_train, X_val, y_train, y_val, y_train_log, y_val_log, X_test = load_data()
    
    # Train
    model, mae, rmse, r2 = train_lightgbm_log(X_train, X_val, y_train, y_val,
                                              y_train_log, y_val_log, X_test)
    
    overall_time = (time.time() - overall_start) / 60
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {overall_time:.2f} minutes\n")
    
    print("📁 Output files:")
    print("   models/lightgbm_log_transformed.pkl")
    print("   models/lightgbm_log_predictions.csv  ← SUBMIT THIS!")
    print("   models/lightgbm_log_feature_importance.csv")
    print("   models/lightgbm_log_summary.csv\n")
    
    print("🎯 Decision Guide:")
    if r2 > 0.70:
        print("   ✅ R² > 0.70: EXCELLENT! Submit immediately!")
    elif r2 > 0.55:
        print("   ✅ R² > 0.55: GOOD! Competitive submission")
    elif r2 > 0.40:
        print("   ⚠️  R² > 0.40: Fair, ensemble with XGBoost")
    else:
        print("   ❌ R² < 0.40: Something's wrong")
    
    print("\n📊 Next steps (if R² > 0.60):")
    print("   1. Compare XGBoost log vs LightGBM log")
    print("   2. Use the better one OR create ensemble")
    print("   3. Submit best predictions\n")
