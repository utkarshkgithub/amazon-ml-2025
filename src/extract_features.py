"""
Feature Extraction Pipeline
Combines image features, text embeddings, and engineered features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    def __init__(self, data_dir='/home/sushi/amazon-ml-2025/data', processed_dir='/home/sushi/amazon-ml-2025/data/processed'):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Processed directory: {self.processed_dir}\n")
        
        # Load text model
        print("Loading Sentence Transformer...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded\n")
    
    def extract_text_embeddings(self, df, dataset_name='train'):
        """Extract embeddings from item_name and bullet_points"""
        print(f"=== Extracting Text Embeddings for {dataset_name} ===\n")
        
        # Item name embeddings
        print("1. Item name embeddings...")
        item_names = df['item_name'].fillna('Unknown Product').astype(str).tolist()
        item_embeddings = self.text_model.encode(
            item_names,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        save_path = self.processed_dir / f'{dataset_name}_item_embeddings.npy'
        np.save(save_path, item_embeddings)
        print(f"   ✓ Saved: {save_path} | Shape: {item_embeddings.shape}\n")
        
        # Bullet point embeddings
        print("2. Bullet point embeddings...")
        bullets = df['bullet_points'].fillna('').astype(str).tolist()
        bullet_embeddings = self.text_model.encode(
            bullets,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        save_path = self.processed_dir / f'{dataset_name}_bullet_embeddings.npy'
        np.save(save_path, bullet_embeddings)
        print(f"   ✓ Saved: {save_path} | Shape: {bullet_embeddings.shape}\n")
        
        return item_embeddings, bullet_embeddings
    
    def extract_text_stats(self, df, dataset_name='train'):
        """Create text statistical features"""
        print(f"=== Creating Text Statistics for {dataset_name} ===\n")
        
        text_stats = pd.DataFrame()
        
        # Item name features
        text_stats['item_name_length'] = df['item_name'].str.len()
        text_stats['item_name_word_count'] = df['item_name'].str.split().str.len()
        text_stats['item_name_has_numbers'] = df['item_name'].str.contains(r'\d', regex=True).astype(int)
        
        # Bullet features
        text_stats['has_bullets'] = (df['bullet_points'].str.len() > 0).astype(int)
        text_stats['bullet_length'] = df['bullet_points'].str.len()
        text_stats['bullet_word_count'] = df['bullet_points'].str.split().str.len()
        
        save_path = self.processed_dir / f'{dataset_name}_text_stats.csv'
        text_stats.to_csv(save_path, index=False)
        print(f"✓ Saved: {save_path} | Shape: {text_stats.shape}\n")
        
        return text_stats
    
    def combine_all_features(self, dataset_name='train'):
        """Combine all features into final feature matrix"""
        print(f"\n=== Combining All Features for {dataset_name} ===\n")
        
        # Load cleaned data
        df = pd.read_csv(self.data_dir / f'{dataset_name}_cleaned.csv')
        print(f"Loaded {dataset_name} data: {df.shape}")
        
        # Load image features
        img_path = self.processed_dir / f'{dataset_name}_multi_image_features.npy'
        print(f"Looking for image features at: {img_path}")
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image features not found at {img_path}\nPlease check the file exists.")
        
        img_features = np.load(img_path)
        print(f"✓ Image features: {img_features.shape}")
        
        # Load text embeddings
        item_emb = np.load(self.processed_dir / f'{dataset_name}_item_embeddings.npy')
        bullet_emb = np.load(self.processed_dir / f'{dataset_name}_bullet_embeddings.npy')
        print(f"✓ Item embeddings: {item_emb.shape}")
        print(f"✓ Bullet embeddings: {bullet_emb.shape}")
        
        # Load text stats
        text_stats = pd.read_csv(self.processed_dir / f'{dataset_name}_text_stats.csv')
        print(f"✓ Text statistics: {text_stats.shape}")
        
        # Extract quantity features from cleaned CSV
        quantity_features = df[[
            'quantity_value', 'unit_multiplier', 'quantity_normalized',
            'quantity_log', 'quantity_sqrt', 'has_quantity'
        ]].values
        print(f"✓ Quantity features: {quantity_features.shape}")
        
        # Combine everything
        X = np.hstack([
            img_features,           # 4288 features
            item_emb,              # 384 features
            bullet_emb,            # 384 features
            text_stats.values,     # 6 features
            quantity_features      # 6 features
        ])
        
        print(f"\n✅ Combined feature matrix: {X.shape}")
        print(f"   Total features: {X.shape[1]}")
        
        # Save combined features
        save_path = self.processed_dir / f'{dataset_name}_features_combined.npy'
        np.save(save_path, X)
        print(f"   Saved: {save_path}")
        
        # Also save target (price) for training data
        if dataset_name == 'train':
            y = df['price'].values
            np.save(self.processed_dir / 'train_target.npy', y)
            print(f"   Saved target: {self.processed_dir / 'train_target.npy'}")
        
        return X, df
    
    def process_dataset(self, dataset_name='train'):
        """Complete feature extraction pipeline"""
        print(f"\n{'='*70}")
        print(f"FEATURE EXTRACTION PIPELINE - {dataset_name.upper()}")
        print(f"{'='*70}\n")
        
        # Load data
        df = pd.read_csv(self.data_dir / f'{dataset_name}_cleaned.csv')
        print(f"✓ Loaded {dataset_name} data: {df.shape}\n")
        
        # Extract text embeddings
        self.extract_text_embeddings(df, dataset_name)
        
        # Extract text stats
        self.extract_text_stats(df, dataset_name)
        
        # Combine all features
        X, df = self.combine_all_features(dataset_name)
        
        print(f"\n{'='*70}")
        print(f"✅ {dataset_name.upper()} FEATURE EXTRACTION COMPLETE!")
        print(f"{'='*70}\n")
        
        return X, df


if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Process training data
    X_train, train_df = extractor.process_dataset('train')
    
    # Process test data
    X_test, test_df = extractor.process_dataset('test')
    
    print("\n🎉 All features extracted successfully!")
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
