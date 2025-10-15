import numpy as np

# Load validation data and predictions
X_train_full = np.load('data/processed/train_features_clean.npy')
y_train_full = np.load('data/processed/train_target_clean.npy')

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Load predictions
lgb_val_pred = np.load('lightgbm_val_predictions.npy')
# Need to load XGBoost validation predictions - let me generate them first

# Load models and make XGBoost validation predictions
import xgboost as xgb
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgboost_model_v1.json')
xgb_val_pred = xgb_model.predict(X_val)

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE - convert from log scale first"""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    
    numerator = np.abs(y_pred_orig - y_true_orig)
    denominator = (np.abs(y_true_orig) + np.abs(y_pred_orig)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape

# Test different ensemble weights on validation set
print("="*60)
print("🔬 TESTING ENSEMBLE WEIGHTS")
print("="*60)

weights = [
    (1.0, 0.0),  # 100% LightGBM
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
    (0.0, 1.0)   # 100% XGBoost
]

best_smape = float('inf')
best_weight = None

for lgb_w, xgb_w in weights:
    ensemble_val_pred = lgb_w * lgb_val_pred + xgb_w * xgb_val_pred
    ensemble_smape = calculate_smape(y_val, ensemble_val_pred)
    
    print(f"LGB:{lgb_w:.1f} + XGB:{xgb_w:.1f} → SMAPE: {ensemble_smape:.2f}", end="")
    
    if ensemble_smape < best_smape:
        best_smape = ensemble_smape
        best_weight = (lgb_w, xgb_w)
        print(" ✅ NEW BEST!")
    else:
        print()

print("\n" + "="*60)
print("🎯 BEST ENSEMBLE RESULT")
print("="*60)
print(f"Optimal Weights: LightGBM {best_weight[0]:.1f} + XGBoost {best_weight[1]:.1f}")
print(f"Validation SMAPE: {best_smape:.2f}")
print(f"\n📊 Comparison:")
print(f"   LightGBM alone:  51.39")
print(f"   XGBoost alone:   54.17")
print(f"   Ensemble:        {best_smape:.2f}")
print(f"\n🚀 Improvement: {51.39 - best_smape:.2f} points from best single model")
print("="*60)

# Create final test predictions with best ensemble
lgb_test_pred = np.load('lightgbm_test_predictions.npy')
xgb_test_pred = np.load('xgboost_test_predictions.npy')

final_ensemble_pred = best_weight[0] * lgb_test_pred + best_weight[1] * xgb_test_pred

print(f"\n📈 FINAL ENSEMBLE TEST PREDICTIONS")
print("="*60)
print(f"Total predictions: {len(final_ensemble_pred)}")
print(f"Min predicted price: ${final_ensemble_pred.min():.2f}")
print(f"Max predicted price: ${final_ensemble_pred.max():.2f}")
print(f"Mean predicted price: ${final_ensemble_pred.mean():.2f}")
print(f"Median predicted price: ${np.median(final_ensemble_pred):.2f}")

# Save final ensemble predictions
np.save('ensemble_test_predictions_best.npy', final_ensemble_pred)
print(f"\n✅ Final ensemble predictions saved!")
print(f"   - ensemble_test_predictions_best.npy")
print(f"\n🎉 Expected leaderboard SMAPE: ~{best_smape:.2f}")
