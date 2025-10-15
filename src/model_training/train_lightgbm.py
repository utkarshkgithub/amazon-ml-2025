import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load cleaned data (same data you used for XGBoost)
X_train_full = np.load('data/processed/train_features_clean.npy')
y_train_full = np.load('data/processed/train_target_clean.npy')

print(f"Loaded: X={X_train_full.shape}, y={y_train_full.shape}")

# Split for validation (use same split as XGBoost for fair comparison)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# LightGBM parameters (optimized for speed and performance)
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

print("\n" + "="*60)
print("🚀 Training LightGBM Model...")
print("="*60)

# Train LightGBM
lgb_model = lgb.LGBMRegressor(**lgb_params)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100)
    ]
)

print("\n✅ Training completed!")

# Evaluate on validation set
train_pred = lgb_model.predict(X_train)
val_pred = lgb_model.predict(X_val)

train_mae = mean_absolute_error(y_train, train_pred)
val_mae = mean_absolute_error(y_val, val_pred)

# Calculate SMAPE
def calculate_smape(y_true, y_pred):
    """Calculate SMAPE - convert from log scale first"""
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    
    numerator = np.abs(y_pred_orig - y_true_orig)
    denominator = (np.abs(y_true_orig) + np.abs(y_pred_orig)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape

val_smape = calculate_smape(y_val, val_pred)

print("\n" + "="*60)
print("📊 LIGHTGBM VALIDATION RESULTS")
print("="*60)
print(f"Train MAE (log scale): {train_mae:.4f}")
print(f"Val MAE (log scale): {val_mae:.4f}")
print(f"Overfitting gap: {val_mae - train_mae:.4f}")
print(f"\nValidation SMAPE: {val_smape:.2f}")

print("\n" + "="*60)
print("🔍 MODEL COMPARISON")
print("="*60)
print(f"XGBoost SMAPE:  54.17")
print(f"LightGBM SMAPE: {val_smape:.2f}")
print(f"Difference:     {abs(54.17 - val_smape):.2f}")

if val_smape < 54.17:
    print(f"✅ LightGBM is {54.17 - val_smape:.2f} points better!")
elif val_smape > 54.17:
    print(f"⚠️ XGBoost is {val_smape - 54.17:.2f} points better")
else:
    print("⚖️ Both models perform similarly")

print("="*60)

# Make predictions on test set
X_test = np.load('data/processed/test_features_final.npy')
print(f"\nMaking predictions on test set: {X_test.shape}")

lgb_test_pred_log = lgb_model.predict(X_test)
lgb_test_pred_original = np.expm1(lgb_test_pred_log)

print(f"\n📈 LightGBM Test Predictions:")
print(f"Total predictions: {len(lgb_test_pred_original)}")
print(f"Min predicted price: ${lgb_test_pred_original.min():.2f}")
print(f"Max predicted price: ${lgb_test_pred_original.max():.2f}")
print(f"Mean predicted price: ${lgb_test_pred_original.mean():.2f}")
print(f"Median predicted price: ${np.median(lgb_test_pred_original):.2f}")

# Save model and predictions
np.save('lightgbm_test_predictions.npy', lgb_test_pred_original)
lgb_model.booster_.save_model('lightgbm_model_v1.txt')

print(f"\n✅ LightGBM model and predictions saved!")
print(f"   - lightgbm_test_predictions.npy")
print(f"   - lightgbm_model_v1.txt")

# Store validation predictions for ensemble
np.save('lightgbm_val_predictions.npy', val_pred)
print(f"   - lightgbm_val_predictions.npy (for ensemble tuning)")

print("\n🎯 Ready for ensemble creation!")
