"""
Data-Driven Category Detection
Automatically discovers product categories from training data
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class DataDrivenCategoryDetector:
    """Automatically discover and detect product categories"""
    
    def __init__(self, train_df=None):
        self.categories = {}
        
        if train_df is not None:
            print("\n🔍 Discovering categories from training data...")
            self.discover_categories(train_df)
        else:
            # Load pre-discovered categories
            self.load_categories()
    
    def discover_categories(self, train_df):
        """Automatically discover categories from data"""
        
        print("Analyzing word co-occurrence patterns...")
        
        # Extract all text
        all_text = []
        prices = []
        for idx, row in train_df.iterrows():
            text = f"{row.get('item_name', '')} {row.get('bullet_points', '')}".lower()
            all_text.append(text)
            prices.append(row.get('price', 0))
        
        # Find common word pairs (potential category indicators)
        word_pairs = []
        for text in all_text:
            words = re.findall(r'\b[a-z]{3,}\b', text)
            word_pairs.extend(words)
        
        # Get most common words
        word_counts = Counter(word_pairs)
        top_words = [w for w, c in word_counts.most_common(200) if c >= 100]
        
        print(f"Found {len(top_words)} common product words")
        
        # Group related words into categories using co-occurrence
        categories_discovered = self._cluster_words(all_text, top_words, prices)
        
        self.categories = categories_discovered
        
        # Save discovered categories
        self.save_categories()
        
        print(f"\n✅ Discovered {len(self.categories)} categories:")
        for cat_name, cat_info in self.categories.items():
            print(f"   {cat_name:25s} Avg: ${cat_info['avg_price']:>7.2f}  "
                  f"Keywords: {', '.join(cat_info['keywords'][:5])}")
    
    def _cluster_words(self, all_text, top_words, prices):
        """Cluster words that appear together into categories"""
        
        # Manual category definitions based on YOUR discovery results
        # These are based on the actual patterns you found
        
        categories = {}
        
        # Tea/Beverage category (you found 'tea' appears in 15.85%)
        tea_words = [w for w in top_words if any(kw in w for kw in 
                    ['tea', 'coffee', 'beverage', 'drink', 'herbal', 'brew'])]
        if len(tea_words) > 0:
            tea_mask = [any(w in text for w in tea_words) for text in all_text]
            tea_prices = [p for p, m in zip(prices, tea_mask) if m]
            categories['tea_beverage'] = {
                'keywords': tea_words[:10],
                'avg_price': np.mean(tea_prices) if tea_prices else 25.0,
                'weight': 1.0,
                'coverage': sum(tea_mask) / len(all_text)
            }
        
        # Flowers/Decor (you found 'flower' keywords)
        flower_words = [w for w in top_words if any(kw in w for kw in 
                       ['flower', 'floral', 'rose', 'bouquet', 'arrangement', 'stem'])]
        if len(flower_words) > 0:
            flower_mask = [any(w in text for w in flower_words) for text in all_text]
            flower_prices = [p for p, m in zip(prices, flower_mask) if m]
            categories['flowers_decor'] = {
                'keywords': flower_words[:10],
                'avg_price': np.mean(flower_prices) if flower_prices else 80.0,
                'weight': 1.2,
                'coverage': sum(flower_mask) / len(all_text)
            }
        
        # Food/Snacks (you found negative correlation words)
        food_words = [w for w in top_words if any(kw in w for kw in 
                     ['sauce', 'seasoning', 'snack', 'chip', 'cookie', 'pasta', 'soup'])]
        if len(food_words) > 0:
            food_mask = [any(w in text for w in food_words) for text in all_text]
            food_prices = [p for p, m in zip(prices, food_mask) if m]
            categories['food_snacks'] = {
                'keywords': food_words[:10],
                'avg_price': np.mean(food_prices) if food_prices else 18.0,
                'weight': 0.8,
                'coverage': sum(food_mask) / len(all_text)
            }
        
        # Organic/Health (you found 'organic' in materials)
        organic_words = [w for w in top_words if any(kw in w for kw in 
                        ['organic', 'natural', 'gluten', 'vegan', 'health'])]
        if len(organic_words) > 0:
            organic_mask = [any(w in text for w in organic_words) for text in all_text]
            organic_prices = [p for p, m in zip(prices, organic_mask) if m]
            categories['organic_health'] = {
                'keywords': organic_words[:10],
                'avg_price': np.mean(organic_prices) if organic_prices else 25.0,
                'weight': 1.1,
                'coverage': sum(organic_mask) / len(all_text)
            }
        
        # Bulk/Survival (you found 'bulk', 'case', 'pound' premium keywords)
        bulk_words = [w for w in top_words if any(kw in w for kw in 
                     ['bulk', 'case', 'emergency', 'survival', 'pound', 'wholesale'])]
        if len(bulk_words) > 0:
            bulk_mask = [any(w in text for w in bulk_words) for text in all_text]
            bulk_prices = [p for p, m in zip(prices, bulk_mask) if m]
            categories['bulk_products'] = {
                'keywords': bulk_words[:10],
                'avg_price': np.mean(bulk_prices) if bulk_prices else 100.0,
                'weight': 1.5,
                'coverage': sum(bulk_mask) / len(all_text)
            }
        
        # Gourmet/Premium (from your premium keywords)
        premium_words = [w for w in top_words if any(kw in w for kw in 
                        ['gourmet', 'artisan', 'imported', 'premium', 'deluxe', 'luxury'])]
        if len(premium_words) > 0:
            premium_mask = [any(w in text for w in premium_words) for text in all_text]
            premium_prices = [p for p, m in zip(prices, premium_mask) if m]
            categories['gourmet_premium'] = {
                'keywords': premium_words[:10],
                'avg_price': np.mean(premium_prices) if premium_prices else 60.0,
                'weight': 1.4,
                'coverage': sum(premium_mask) / len(all_text)
            }
        
        return categories
    
    def save_categories(self):
        """Save discovered categories"""
        categories_data = []
        for cat_name, cat_info in self.categories.items():
            categories_data.append({
                'category': cat_name,
                'keywords': ','.join(cat_info['keywords']),
                'avg_price': cat_info['avg_price'],
                'weight': cat_info['weight'],
                'coverage': cat_info.get('coverage', 0)
            })
        
        df = pd.DataFrame(categories_data)
        df.to_csv('data/processed/categories_discovered.csv', index=False)
        print(f"\n✅ Saved discovered categories to data/processed/categories_discovered.csv")
    
    def load_categories(self):
        """Load pre-discovered categories"""
        try:
            df = pd.read_csv('data/processed/categories_discovered.csv')
            for _, row in df.iterrows():
                self.categories[row['category']] = {
                    'keywords': row['keywords'].split(','),
                    'avg_price': row['avg_price'],
                    'weight': row['weight'],
                    'coverage': row.get('coverage', 0)
                }
            print(f"✓ Loaded {len(self.categories)} pre-discovered categories")
        except:
            print("⚠️  No pre-discovered categories found, using defaults")
            self.categories = {}
    
    def detect_categories(self, text: str) -> dict:
        """Detect which categories this product belongs to"""
        
        if pd.isna(text):
            text = ""
        
        text = str(text).lower()
        
        detected = {}
        category_scores = {}
        
        for cat_name, cat_info in self.categories.items():
            # Count matching keywords
            matches = sum(1 for kw in cat_info['keywords'] if kw in text)
            
            detected[cat_name] = int(matches > 0)
            category_scores[cat_name] = matches
        
        return detected, category_scores
    
    def extract_features(self, row):
        """Extract category features"""
        
        # Get text
        item_name = str(row.get('item_name', ''))
        bullet_points = str(row.get('bullet_points', ''))
        combined_text = f"{item_name} {bullet_points}".lower()
        
        # Detect categories
        detected, scores = self.detect_categories(combined_text)
        
        features = {}
        
        # Binary flags
        for cat in self.categories.keys():
            features[f'is_{cat}'] = detected.get(cat, 0)
            features[f'score_{cat}'] = scores.get(cat, 0)
        
        # Aggregate
        features['category_count'] = sum(detected.values())
        features['has_any_category'] = int(sum(detected.values()) > 0)
        
        # Weighted avg price
        total_weight = 0
        weighted_price = 0
        
        for cat_name, cat_info in self.categories.items():
            if detected.get(cat_name, 0):
                weight = cat_info['weight'] * scores.get(cat_name, 0)
                weighted_price += cat_info['avg_price'] * weight
                total_weight += weight
        
        features['avg_category_price'] = weighted_price / max(total_weight, 1)
        
        return features
    
    def transform(self, df):
        """Transform dataframe"""
        print(f"\n📂 Detecting categories for {len(df):,} samples...")
        
        all_features = []
        for idx, row in df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processed {idx:,}/{len(df):,} rows...")
            
            features = self.extract_features(row)
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        print(f"\n✅ Extracted {len(features_df.columns)} category features")
        
        return features_df


def main():
    """Discover categories and extract features"""
    print("="*70)
    print("PHASE 2: DATA-DRIVEN CATEGORY DETECTION")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    train_df = pd.read_csv('data/train_cleaned.csv')
    test_df = pd.read_csv('data/test_cleaned.csv')
    
    # Discover categories from training data
    detector = DataDrivenCategoryDetector(train_df=train_df)
    
    # Extract features
    print("\n" + "="*70)
    print("PROCESSING TRAIN DATA")
    print("="*70)
    train_features = detector.transform(train_df)
    
    print("\n" + "="*70)
    print("PROCESSING TEST DATA")
    print("="*70)
    test_features = detector.transform(test_df)
    
    # Save
    print("\n" + "="*70)
    print("SAVING FEATURES")
    print("="*70)
    train_features.to_csv('data/processed/train_category_features.csv', index=False)
    test_features.to_csv('data/processed/test_category_features.csv', index=False)
    
    print(f"✅ Saved: {train_features.shape}")
    
    # Stats
    print("\n📊 Category Coverage:")
    category_cols = [col for col in train_features.columns if col.startswith('is_')]
    for col in category_cols:
        coverage = train_features[col].mean() * 100
        print(f"   {col:30s} {coverage:>6.1f}%")


if __name__ == "__main__":
    main()
