"""
Advanced Text Feature Extraction Using Discovered Patterns
Based on data-driven discovery from training set
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class AdvancedTextFeatureExtractor:
    """Extract features using discovered price-predictive patterns"""
    
    def __init__(self):
        # Load discovered patterns
        print("Loading discovered patterns...")
        
        self.premium_keywords = self._load_keywords('data/processed/premium_keywords_discovered.csv')
        self.budget_keywords = self._load_keywords('data/processed/budget_keywords_discovered.csv')
        self.materials = self._load_materials('data/processed/material_price_analysis.csv')
        self.brands = self._load_brands('data/processed/brand_price_analysis.csv')
        self.units = self._load_units('data/processed/unit_price_analysis.csv')
        
        print(f"✓ Loaded {len(self.premium_keywords)} premium keywords")
        print(f"✓ Loaded {len(self.budget_keywords)} budget keywords")
        print(f"✓ Loaded {len(self.materials)} materials")
        print(f"✓ Loaded {len(self.brands)} brands")
        print(f"✓ Loaded {len(self.units)} unit patterns")
    
    def _load_keywords(self, path):
        """Load discovered keywords"""
        try:
            df = pd.read_csv(path)
            # Return dict: word -> correlation strength
            return dict(zip(df['word'], df['correlation']))
        except:
            return {}
    
    def _load_materials(self, path):
        """Load material price patterns"""
        try:
            df = pd.read_csv(path)
            # Normalize by average
            avg_price = df['avg_price'].mean()
            return dict(zip(df['material'], df['avg_price'] / avg_price))
        except:
            return {}
    
    def _load_brands(self, path):
        """Load brand price patterns"""
        try:
            df = pd.read_csv(path)
            # Normalize by average
            avg_price = df['avg_price'].mean()
            return dict(zip(df['brand'].str.lower(), df['avg_price'] / avg_price))
        except:
            return {}
    
    def _load_units(self, path):
        """Load unit price patterns"""
        try:
            df = pd.read_csv(path)
            
            # Check if file exists and has data
            if df.empty:
                print(f"   ⚠️  Unit CSV is empty")
                return {}
            
            # Normalize by average price
            if 'avg_price' not in df.columns:
                print(f"   ⚠️  'avg_price' column not found in {path}")
                print(f"   Available columns: {list(df.columns)}")
                return {}
            
            avg_price = df['avg_price'].mean()
            
            # Create unit dictionary
            unit_col = 'Unit' if 'Unit' in df.columns else 'unit'
            if unit_col not in df.columns:
                print(f"   ⚠️  Neither 'Unit' nor 'unit' column found")
                return {}
            
            units_dict = dict(zip(df[unit_col], df['avg_price'] / avg_price))
            
            return units_dict
            
        except Exception as e:
            print(f"   ⚠️  Error loading units: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def extract_features(self, row):
        """Extract all features from a single row"""
        
        # Get text
        item_name = str(row.get('item_name', '')).lower()
        bullet_points = str(row.get('bullet_points', '')).lower()
        combined_text = f"{item_name} {bullet_points}"
        unit = str(row.get('unit', ''))
        
        # Extract words
        words = set(re.findall(r'\b[a-z]{3,}\b', combined_text))
        
        features = {}
        
        # 1. Premium keyword score (weighted by correlation)
        premium_score = sum(self.premium_keywords.get(word, 0) for word in words)
        premium_count = sum(1 for word in words if word in self.premium_keywords)
        features['premium_score'] = premium_score
        features['premium_count'] = premium_count
        features['has_premium'] = int(premium_count > 0)
        
        # 2. Budget keyword score
        budget_score = sum(abs(self.budget_keywords.get(word, 0)) for word in words)
        budget_count = sum(1 for word in words if word in self.budget_keywords)
        features['budget_score'] = budget_score
        features['budget_count'] = budget_count
        features['has_budget'] = int(budget_count > 0)
        
        # 3. Net premium score (premium - budget)
        features['net_premium_score'] = premium_score - budget_score
        
        # 4. Material score
        material_score = 0
        material_count = 0
        for material, score in self.materials.items():
            if material in combined_text:
                material_score += score
                material_count += 1
        features['material_score'] = material_score
        features['material_count'] = material_count
        features['has_premium_material'] = int(material_score > 1.5)
        
        # 5. Brand detection
        brand_score = 0
        brand_found = 0
        # Extract capitalized words from item_name
        item_name_orig = str(row.get('item_name', ''))
        cap_words = re.findall(r'\b[A-Z][a-z]{2,}\b', item_name_orig)
        for word in cap_words:
            if word.lower() in self.brands:
                brand_score = max(brand_score, self.brands[word.lower()])
                brand_found = 1
        features['brand_score'] = brand_score
        features['has_known_brand'] = brand_found
        features['is_luxury_brand'] = int(brand_score > 2.0)
        
        # 6. Unit price indicator
        unit_score = self.units.get(unit, 1.0)  # Default to 1.0 (average)
        features['unit_price_score'] = unit_score
        features['is_expensive_unit'] = int(unit_score > 1.5)
        
        # 7. Text statistics
        features['text_length'] = len(combined_text)
        features['word_count'] = len(words)
        features['item_name_length'] = len(item_name)
        features['has_bullet_points'] = int(len(bullet_points) > 10)
        
        # 8. Special patterns
        features['has_organic'] = int('organic' in combined_text)
        features['has_natural'] = int('natural' in combined_text)
        features['has_pack'] = int('pack' in combined_text)
        features['has_bulk'] = int('bulk' in combined_text)
        features['has_frozen'] = int('frozen' in combined_text)
        features['has_fresh'] = int('fresh' in combined_text)
        
        # 9. Category indicators (from top correlations)
        features['is_tea_related'] = int('tea' in combined_text)
        features['is_flower_related'] = int('flower' in combined_text or 'floral' in combined_text)
        features['is_food_related'] = int(any(w in combined_text for w in ['sauce', 'seasoning', 'snack']))
        
        return features
    
    def transform(self, df):
        """Transform entire dataframe"""
        print(f"\n🔍 Extracting advanced text features from {len(df):,} samples...")
        
        all_features = []
        for idx, row in df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processed {idx:,}/{len(df):,} rows...")
            
            features = self.extract_features(row)
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        
        print(f"\n✅ Extracted {len(features_df.columns)} advanced features")
        return features_df


def main():
    """Extract features for train and test"""
    print("="*70)
    print("ADVANCED TEXT FEATURE EXTRACTION")
    print("Using Data-Driven Discovered Patterns")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    train_df = pd.read_csv('data/train_cleaned.csv')
    test_df = pd.read_csv('data/test_cleaned.csv')
    print(f"✓ Train: {len(train_df):,} samples")
    print(f"✓ Test:  {len(test_df):,} samples")
    
    # Initialize extractor
    extractor = AdvancedTextFeatureExtractor()
    
    # Extract train features
    print("\n" + "="*70)
    print("PROCESSING TRAIN DATA")
    print("="*70)
    train_features = extractor.transform(train_df)
    
    # Extract test features
    print("\n" + "="*70)
    print("PROCESSING TEST DATA")
    print("="*70)
    test_features = extractor.transform(test_df)
    
    # Save
    print("\n" + "="*70)
    print("SAVING FEATURES")
    print("="*70)
    train_features.to_csv('data/processed/train_advanced_text_features.csv', index=False)
    test_features.to_csv('data/processed/test_advanced_text_features.csv', index=False)
    
    print(f"✅ Saved train features: {train_features.shape}")
    print(f"✅ Saved test features:  {test_features.shape}")
    
    # Show sample
    print("\n📊 Sample Features (first 3 rows):")
    print(train_features.head(3).T)
    
    # Statistics
    print("\n📈 Feature Statistics:")
    print(train_features.describe().T[['mean', 'std', 'min', 'max']].head(15))
    
    print("\n🎯 Next step:")
    print("   Combine these with your existing features and retrain!")


if __name__ == "__main__":
    main()
