"""
Integrate all extracted features
Combines: Original (5068) + Text (28) + Category (13) = 5109 features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def integrate_all_features():
    """Combine all feature sets"""
    
    print("="*70)
    print("FEATURE INTEGRATION")
    print("="*70)
    
    # 1. Load original features (image + text embeddings + quantity)
    print("\n📦 Loading original features...")
    X_train_original = np.load('data/processed/train_features_combined.npy')
    X_test_original = np.load('data/processed/test_features_combined.npy')
    print(f"✓ Train: {X_train_original.shape}")
    print(f"✓ Test:  {X_test_original.shape}")
    
    # 2. Load advanced text features
    print("\n📝 Loading advanced text features...")
    train_text = pd.read_csv('data/processed/train_advanced_text_features.csv')
    test_text = pd.read_csv('data/processed/test_advanced_text_features.csv')
    print(f"✓ Train: {train_text.shape}")
    print(f"✓ Test:  {test_text.shape}")
    
    # 3. Load category features
    print("\n📂 Loading category features...")
    train_category = pd.read_csv('data/processed/train_category_features.csv')
    test_category = pd.read_csv('data/processed/test_category_features.csv')
    print(f"✓ Train: {train_category.shape}")
    print(f"✓ Test:  {test_category.shape}")
    
    # 4. Combine all features
    print("\n🔗 Combining all features...")
    X_train_combined = np.concatenate([
        X_train_original,
        train_text.values,
        train_category.values
    ], axis=1)
    
    X_test_combined = np.concatenate([
        X_test_original,
        test_text.values,
        test_category.values
    ], axis=1)
    
    print(f"\n✅ Final combined shape:")
    print(f"   Train: {X_train_combined.shape}")
    print(f"   Test:  {X_test_combined.shape}")
    
    # 5. Load target
    print("\n🎯 Loading target...")
    y_train = np.load('data/processed/train_target.npy')
    print(f"✓ Target shape: {y_train.shape}")
    
    # 6. Save combined features
    print("\n💾 Saving combined features...")
    np.save('data/processed/train_features_final.npy', X_train_combined)
    np.save('data/processed/test_features_final.npy', X_test_combined)
    np.save('data/processed/train_target_final.npy', y_train)
    
    print("✅ Saved:")
    print("   data/processed/train_features_final.npy")
    print("   data/processed/test_features_final.npy")
    print("   data/processed/train_target_final.npy")
    
    # 7. Create feature names
    feature_names = []
    feature_names.extend([f'original_{i}' for i in range(X_train_original.shape[1])])
    feature_names.extend(train_text.columns.tolist())
    feature_names.extend(train_category.columns.tolist())
    
    with open('data/processed/feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    print(f"\n✅ Saved feature names ({len(feature_names)} features)")
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION SUMMARY")
    print("="*70)
    print(f"Original features:      {X_train_original.shape[1]:>6,}")
    print(f"Text features:          {train_text.shape[1]:>6,}")
    print(f"Category features:      {train_category.shape[1]:>6,}")
    print(f"-" * 70)
    print(f"TOTAL FEATURES:         {X_train_combined.shape[1]:>6,}")
    print(f"Training samples:       {X_train_combined.shape[0]:>6,}")
    print(f"Test samples:           {X_test_combined.shape[0]:>6,}")
    
    return X_train_combined, X_test_combined, y_train


if __name__ == "__main__":
    integrate_all_features()
