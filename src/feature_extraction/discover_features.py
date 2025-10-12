"""
Data-Driven Feature Discovery - CUSTOMIZED FOR YOUR DATA
Columns: sample_id, item_name, value, unit, bullet_points, image_link, price
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class DataDrivenFeatureDiscovery:
    """Discover price-predictive keywords from training data"""
    
    def __init__(self, train_df):
        self.train_df = train_df
        self.price_col = 'price'  # ✅ Your price column
        self.text_columns = ['item_name', 'bullet_points']  # ✅ Your text columns
        
        print(f"✓ Using price column: '{self.price_col}'")
        print(f"✓ Using text columns: {self.text_columns}")
        print(f"✓ Dataset size: {len(train_df):,} samples")
        
    def analyze_word_correlations(self, min_frequency=30):
        """Find words that correlate with price"""
        print("\n" + "="*70)
        print("DISCOVERING PRICE-PREDICTIVE KEYWORDS FROM DATA")
        print("="*70)
        
        # Combine all text
        print("\nCombining text fields...")
        all_text = []
        for idx, row in self.train_df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processed {idx:,}/{len(self.train_df):,} rows...")
            
            # Combine item_name + bullet_points
            item_name = str(row.get('item_name', ''))
            bullet_points = str(row.get('bullet_points', ''))
            combined = f"{item_name} {bullet_points}".lower()
            all_text.append(combined)
        
        # Extract all words
        print("\nExtracting words...")
        all_words = []
        for text in all_text:
            words = re.findall(r'\b[a-z]{3,}\b', text)
            all_words.extend(words)
        
        # Count word frequencies
        print("Counting word frequencies...")
        word_counts = Counter(all_words)
        
        # Filter by frequency
        common_words = {word: count for word, count in word_counts.items() 
                       if count >= min_frequency}
        
        print(f"\n✓ Found {len(common_words):,} words appearing ≥{min_frequency} times")
        
        if len(common_words) == 0:
            print("❌ No common words found!")
            print(f"   Sample text: {all_text[0][:200]}")
            return [], []
        
        # Calculate correlation with price
        print("\nCalculating price correlations...")
        print(f"   Analyzing top {min(5000, len(common_words))} most common words...")
        
        word_correlations = {}
        prices = self.train_df[self.price_col].values
        
        # Sort by frequency and take top N
        top_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:5000]
        
        for idx, (word, freq) in enumerate(top_words):
            if idx % 500 == 0 and idx > 0:
                print(f"   Analyzed {idx}/{len(top_words)} words...")
            
            # Create binary feature
            word_present = np.array([int(word in text) for text in all_text])
            
            # Skip if too rare or too common
            presence_rate = word_present.mean()
            if presence_rate < 0.005 or presence_rate > 0.95:
                continue
            
            # Calculate correlation
            try:
                corr, p_value = pearsonr(word_present, prices)
                if abs(corr) > 0.03 and p_value < 0.05:
                    word_correlations[word] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'frequency': freq,
                        'presence_rate': presence_rate
                    }
            except:
                continue
        
        print(f"\n✓ Found {len(word_correlations)} price-correlated words")
        
        if len(word_correlations) == 0:
            print("⚠️  No significant correlations found.")
            return [], []
        
        # Sort by correlation
        sorted_words = sorted(word_correlations.items(), 
                            key=lambda x: abs(x[1]['correlation']), 
                            reverse=True)
        
        # Separate positive and negative
        premium_words = [(w, d) for w, d in sorted_words if d['correlation'] > 0]
        budget_words = [(w, d) for w, d in sorted_words if d['correlation'] < 0]
        
        # Display
        print("\n" + "="*70)
        print(f"TOP {min(50, len(premium_words))} PREMIUM INDICATORS")
        print("="*70)
        print(f"{'Word':<25} {'Correlation':>12} {'Frequency':>12} {'Presence %':>12}")
        print("-"*70)
        for word, data in premium_words[:50]:
            print(f"{word:<25} {data['correlation']:>12.4f} {data['frequency']:>12,} {data['presence_rate']*100:>11.2f}%")
        
        print("\n" + "="*70)
        print(f"TOP {min(50, len(budget_words))} BUDGET INDICATORS")
        print("="*70)
        print(f"{'Word':<25} {'Correlation':>12} {'Frequency':>12} {'Presence %':>12}")
        print("-"*70)
        for word, data in budget_words[:50]:
            print(f"{word:<25} {data['correlation']:>12.4f} {data['frequency']:>12,} {data['presence_rate']*100:>11.2f}%")
        
        # Save
        if len(premium_words) > 0:
            premium_df = pd.DataFrame([
                {'word': w, 'correlation': d['correlation'], 'frequency': d['frequency']}
                for w, d in premium_words[:200]
            ])
            premium_df.to_csv('data/processed/premium_keywords_discovered.csv', index=False)
            print(f"\n✅ Saved {len(premium_df)} premium keywords")
        
        if len(budget_words) > 0:
            budget_df = pd.DataFrame([
                {'word': w, 'correlation': d['correlation'], 'frequency': d['frequency']}
                for w, d in budget_words[:200]
            ])
            budget_df.to_csv('data/processed/budget_keywords_discovered.csv', index=False)
            print(f"✅ Saved {len(budget_df)} budget keywords")
        
        return premium_words, budget_words
    
    def analyze_units(self):
        """Analyze unit-price relationships"""
        print("\n" + "="*70)
        print("ANALYZING UNIT-PRICE RELATIONSHIPS")
        print("="*70)
        
        # Group by unit
        unit_stats = self.train_df.groupby('unit')['price'].agg([
            ('avg_price', 'mean'),
            ('median_price', 'median'),
            ('count', 'count')
        ]).reset_index()
        
        # Filter units with at least 20 samples
        unit_stats = unit_stats[unit_stats['count'] >= 20]
        unit_stats = unit_stats.sort_values('avg_price', ascending=False)
        
        print(f"\n{'Unit':<20} {'Avg Price':>12} {'Median Price':>15} {'Count':>10}")
        print("-"*70)
        for _, row in unit_stats.iterrows():
            print(f"{row['unit']:<20} ${row['avg_price']:>11.2f} ${row['median_price']:>14.2f} {row['count']:>10,}")
        
        # Save
        unit_stats.to_csv('data/processed/unit_price_analysis.csv', index=False)
        print(f"\n✅ Saved {len(unit_stats)} unit patterns")
        
        return unit_stats
    
    def analyze_materials(self):
        """Discover material-price relationships from text"""
        print("\n" + "="*70)
        print("DISCOVERING MATERIAL INDICATORS")
        print("="*70)
        
        material_patterns = [
            'gold', 'silver', 'platinum', 'diamond', 'leather', 'silk',
            'cotton', 'wool', 'plastic', 'rubber', 'wood', 'metal',
            'stainless', 'steel', 'aluminum', 'glass', 'ceramic',
            'organic', 'natural', 'synthetic'
        ]
        
        material_prices = {}
        
        for material in material_patterns:
            # Search in both columns
            mask = (self.train_df['item_name'].astype(str).str.lower().str.contains(material, na=False) | 
                   self.train_df['bullet_points'].astype(str).str.lower().str.contains(material, na=False))
            
            if mask.sum() >= 10:
                prices = self.train_df[mask]['price']
                material_prices[material] = {
                    'avg_price': prices.mean(),
                    'median_price': prices.median(),
                    'count': mask.sum()
                }
        
        if len(material_prices) == 0:
            print("⚠️  No materials found")
            return []
        
        # Sort
        sorted_materials = sorted(material_prices.items(), 
                                 key=lambda x: x[1]['avg_price'], 
                                 reverse=True)
        
        print(f"\n{'Material':<15} {'Avg Price':>12} {'Median Price':>15} {'Count':>10}")
        print("-"*70)
        for material, data in sorted_materials:
            print(f"{material:<15} ${data['avg_price']:>11.2f} ${data['median_price']:>14.2f} {data['count']:>10,}")
        
        # Save
        materials_df = pd.DataFrame([
            {'material': m, 'avg_price': d['avg_price'], 'count': d['count']}
            for m, d in sorted_materials
        ])
        materials_df.to_csv('data/processed/material_price_analysis.csv', index=False)
        print(f"\n✅ Saved {len(materials_df)} material patterns")
        
        return sorted_materials
    
    def analyze_brands(self):
        """Extract brands from item names"""
        print("\n" + "="*70)
        print("DISCOVERING BRAND INDICATORS")
        print("="*70)
        
        brand_prices = {}
        
        print("Extracting brands from item names...")
        for idx, row in self.train_df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processed {idx:,}/{len(self.train_df):,} rows...")
            
            item_name = str(row.get('item_name', ''))
            price = row['price']
            
            # Extract capitalized words (potential brands)
            brands = re.findall(r'\b[A-Z][a-z]{2,}\b', item_name)
            
            for brand in brands:
                if brand not in brand_prices:
                    brand_prices[brand] = []
                brand_prices[brand].append(price)
        
        # Calculate stats
        brand_stats = {}
        for brand, prices in brand_prices.items():
            if len(prices) >= 15:
                brand_stats[brand] = {
                    'avg_price': np.mean(prices),
                    'median_price': np.median(prices),
                    'count': len(prices)
                }
        
        if len(brand_stats) == 0:
            print("⚠️  No brands found")
            return []
        
        # Sort
        sorted_brands = sorted(brand_stats.items(), 
                              key=lambda x: x[1]['avg_price'], 
                              reverse=True)
        
        print(f"\n{'Brand':<20} {'Avg Price':>12} {'Median Price':>15} {'Count':>10}")
        print("-"*70)
        for brand, data in sorted_brands[:50]:
            print(f"{brand:<20} ${data['avg_price']:>11.2f} ${data['median_price']:>14.2f} {data['count']:>10,}")
        
        # Save
        brands_df = pd.DataFrame([
            {'brand': b, 'avg_price': d['avg_price'], 'count': d['count']}
            for b, d in sorted_brands[:200]
        ])
        brands_df.to_csv('data/processed/brand_price_analysis.csv', index=False)
        print(f"\n✅ Saved {len(brands_df)} brand patterns")
        
        return sorted_brands


def main():
    """Run discovery process"""
    print("="*70)
    print("DATA-DRIVEN FEATURE DISCOVERY")
    print("Customized for Amazon ML Challenge 2025")
    print("="*70)
    
    # Load data
    print("\nLoading training data...")
    train_df = pd.read_csv('data/train_cleaned.csv')
    print(f"✓ Loaded {len(train_df):,} training samples")
    print(f"✓ Columns: {list(train_df.columns)}")
    
    # Initialize
    discoverer = DataDrivenFeatureDiscovery(train_df)
    
    # Run discoveries
    premium_words, budget_words = discoverer.analyze_word_correlations(min_frequency=30)
    units = discoverer.analyze_units()
    materials = discoverer.analyze_materials()
    brands = discoverer.analyze_brands()
    
    print("\n" + "="*70)
    print("✅ DISCOVERY COMPLETE!")
    print("="*70)
    print("\nGenerated files in data/processed/:")
    print("  • premium_keywords_discovered.csv")
    print("  • budget_keywords_discovered.csv")
    print("  • unit_price_analysis.csv")
    print("  • material_price_analysis.csv")
    print("  • brand_price_analysis.csv")
    
    print("\n🎯 Next steps:")
    print("  1. Review the discovered patterns")
    print("  2. Use them in TextFeatureExtractor")
    print("  3. Extract advanced features")
    print("  4. Retrain models with new features!")


if __name__ == "__main__":
    main()
