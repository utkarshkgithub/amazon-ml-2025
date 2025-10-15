# Run this on your LOCAL machine

import pandas as pd
import numpy as np

# Load cleaned data
train_cleaned = pd.read_csv('data/train_cleaned.csv')
test_cleaned = pd.read_csv('data/test_cleaned.csv')

# Extract quantity columns
qty_cols = ['quantity_value', 'quantity_normalized', 'quantity_log', 
            'quantity_sqrt', 'has_quantity', 'unit_multiplier']

train_qty = train_cleaned[qty_cols]
test_qty = test_cleaned[qty_cols]

# Apply same outlier mask as train_features_clean.npy
y = train_cleaned['price'].values
y_log = np.log1p(y)
Q1, Q3 = np.percentile(y_log, [25, 75])
IQR = Q3 - Q1
mask = (y_log >= Q1 - 1.5*IQR) & (y_log <= Q3 + 1.5*IQR)

train_qty_clean = train_qty[mask].reset_index(drop=True)

print(f"Train: {train_qty_clean.shape}")  # Should be (74758, 6)
print(f"Test: {test_qty.shape}")           # Should be (75000, 6)

# Save
train_qty_clean.to_csv('train_quantity_6.csv', index=False)
test_qty.to_csv('test_quantity_6.csv', index=False)

print("✅ Saved! Upload to Kaggle.")
