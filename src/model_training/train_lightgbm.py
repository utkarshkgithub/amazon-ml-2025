"""
Train LightGBM Model Only
Enhanced with progress tracking and early stopping
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
    """Load preprocessed features"""
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
    
    print(f"\n💰 Target Statistics (Training):")
    print(f"   Min:    ${y_train_full.min():.2f}")
    print(f"   Max:    ${y_train_full.max():.2f}")
    print(f"   Mean:   ${y_train_full.mean():.2f}")
    print(f"   Median: ${np.median(y_train_full):.2f}")
    
    total_time = time.time() - load_start
    print(f"\n✅ Data loading completed in {total_time:.2f}s\n")
    
    return X_train, X_val, y_train, y_val, X_test


def train_lightgbm(X_train, X_val, y_train, y_val, X_test):
    """Train LightGBM with progress tracking and early stopping"""
    
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start = time.time()
    
    # Define model with enhanced parameters
    model = lgb.LGBMRegressor(
        n_estimators=1000,           # Max iterations
        max_depth=8,                 # Tree depth
        learning_rate=0.05,          # Step size
        subsample=0.8,               # Row sampling
        colsample_bytree=0.8,        # Column sampling
        min_child_samples=20,        # Minimum samples in leaf
        reg_alpha=0.1,               # L1 regularization
        reg_lambda=1.0,              # L2 regularization
        random_state=42,
        n_jobs=-1,                   # Use all CPU cores
        verbose=-1                   # Suppress warnings
    )
    
    print("📋 Model Configuration:")
    print(f"   Max iterations:  {model.n_estimators}")
    print(f"   Max depth:       {model.max_depth}")
    print(f"   Learning rate:   {model.learning_rate}")
    print(f"   Early stopping:  50 rounds")
    print(f"   Eval metric:     MAE")
    print(f"   CPU cores:       All available\n")
    
    print("🚀 Training started...")
    print("   (Updates every 10 iterations)\n")
    
    # Train with callbacks
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.early_stopping(stopping_rounds=50)
        ]
    )
    
    # Get best iteration info
    best_iteration = model.best_iteration_
    best_score = model.best_score_['valid_0']['l1']  # MAE is l1 in LightGBM
    
    print(f"\n✅ Training completed!")
    print(f"   Best iteration: {best_iteration}")
    print(f"   Best validation score: {best_score:.4f}")
    print(f"   Total iterations: {model.n_estimators}")
    print(f"   Early stopped: {'Yes' if best_iteration < model.n_estimators else 'No'}\n")
    
    # Validate
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70 + "\n")
    
    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)
    
    # Training metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    # Validation metrics
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    
    # Percentage error
    val_mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
    
    print("📊 Training Set:")
    print(f"   MAE:  ${train_mae:.2f}")
    print(f"   RMSE: ${train_rmse:.2f}")
    print(f"   R²:   {train_r2:.4f}")
    
    print("\n📊 Validation Set:")
    print(f"   MAE:   ${val_mae:.2f}")
    print(f"   RMSE:  ${val_rmse:.2f}")
    print(f"   R²:    {val_r2:.4f}")
    print(f"   MAPE:  {val_mape:.2f}%")
    
    # Overfitting check
    overfit_mae = ((train_mae - val_mae) / val_mae) * 100
    print(f"\n🔍 Overfitting Check:")
    if abs(overfit_mae) < 10:
        print(f"   Status: ✅ Good (MAE diff: {abs(overfit_mae):.1f}%)")
    elif abs(overfit_mae) < 20:
        print(f"   Status: ⚠️  Slight overfitting (MAE diff: {abs(overfit_mae):.1f}%)")
    else:
        print(f"   Status: ❌ Overfitting detected (MAE diff: {abs(overfit_mae):.1f}%)")
    
    # Predict test
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70 + "\n")
    
    pred_start = time.time()
    test_pred = model.predict(X_test)
    pred_time = time.time() - pred_start
    
    print(f"✓ Generated {len(test_pred):,} predictions in {pred_time:.2f}s")
    print(f"   Predictions per second: {len(test_pred)/pred_time:.0f}")
    
    # Prediction statistics
    print(f"\n📈 Prediction Statistics:")
    print(f"   Min:    ${test_pred.min():.2f}")
    print(f"   Max:    ${test_pred.max():.2f}")
    print(f"   Mean:   ${test_pred.mean():.2f}")
    print(f"   Median: ${np.median(test_pred):.2f}")
    
    # Save model
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70 + "\n")
    
    Path('models').mkdir(exist_ok=True)
    
    # Save model
    model_path = 'models/lightgbm_baseline.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    model_size = Path(model_path).stat().st_size / (1024*1024)  # MB
    print(f"✓ Model saved: {model_path} ({model_size:.1f} MB)")
    
    # Save predictions
    test_df = pd.read_csv('/home/sushi/amazon-ml-2025/data/test_cleaned.csv')
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_pred
    })
    pred_path = 'models/lightgbm_predictions.csv'
    submission.to_csv(pred_path, index=False)
    print(f"✓ Predictions saved: {pred_path}")
    
    # Save feature importance
    feature_imp = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    imp_path = 'models/lightgbm_feature_importance.csv'
    feature_imp.to_csv(imp_path, index=False)
    print(f"✓ Feature importance saved: {imp_path}")
    
    # Show top 10 features
    print(f"\n🏆 Top 10 Most Important Features:")
    for i, row in feature_imp.head(10).iterrows():
        print(f"   {row['feature']:20s} {row['importance']:.6f}")
    
    # Training summary
    summary = {
        'model': 'LightGBM',
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_mape': val_mape,
        'best_iteration': best_iteration,
        'training_time_min': (time.time() - start) / 60
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = 'models/lightgbm_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Training summary saved: {summary_path}")
    
    elapsed = (time.time() - start) / 60
    print(f"\n⏱️  Total training time: {elapsed:.2f} minutes")
    print(f"   ({elapsed*60:.0f} seconds)")
    
    return model, val_mae, val_rmse, val_r2


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LIGHTGBM PRICE PREDICTION TRAINING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    # Load data
    X_train, X_val, y_train, y_val, X_test = load_data()
    
    # Train
    model, mae, rmse, r2 = train_lightgbm(X_train, X_val, y_train, y_val, X_test)
    
    overall_time = (time.time() - overall_start) / 60
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {overall_time:.2f} minutes\n")
    
    print("📁 Output files:")
    print("   models/lightgbm_baseline.pkl")
    print("   models/lightgbm_predictions.csv  ← Submit this!")
    print("   models/lightgbm_feature_importance.csv")
    print("   models/lightgbm_summary.csv\n")
    
    print("🎯 Next steps:")
    print("   1. Compare with XGBoost results")
    print("   2. Train CatBoost for third opinion")
    print("   3. Create ensemble of all models\n")
