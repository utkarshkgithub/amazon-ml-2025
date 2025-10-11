"""
advanced_feature_extraction.py - Multi-model approach for best results
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


class MultiModelFeatureExtractor:
    """Extract features from multiple models for best performance"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {}
        self.feature_dims = {}
        
        print("Loading multiple models...")
        
        # 1. EfficientNet-B0 (Good for general product features)
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.models['efficientnet'] = nn.Sequential(*list(efficientnet.children())[:-1])
        self.feature_dims['efficientnet'] = 1280
        
        # 2. ResNet50 (Good for detailed features)
        resnet = models.resnet50(pretrained=True)
        self.models['resnet'] = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dims['resnet'] = 2048
        
        # 3. MobileNet-V3 (Good for packaging/text detection)
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.models['mobilenet'] = nn.Sequential(*list(mobilenet.children())[:-1])
        self.feature_dims['mobilenet'] = 960
        
        # Move all to device and set to eval
        for name, model in self.models.items():
            model.to(device)
            model.eval()
            print(f"✓ {name}: {self.feature_dims[name]} features")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.total_dims = sum(self.feature_dims.values())
        print(f"\nTotal feature dimension: {self.total_dims}")
    
    def extract_single(self, image_path):
        """Extract features from all models for one image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            all_features = []
            
            with torch.no_grad():
                for name, model in self.models.items():
                    features = model(img_tensor)
                    features = features.squeeze().cpu().numpy()
                    all_features.append(features)
            
            # Concatenate all features
            combined = np.concatenate(all_features)
            return combined
        
        except Exception as e:
            return np.zeros(self.total_dims)
    
    def extract_all(self, csv_file, images_dir, output_file, batch_size=16):
        """Extract features for all images"""
        df = pd.read_csv(csv_file)
        images_path = Path(images_dir)
        
        print(f"\nExtracting from {len(df)} images...")
        
        all_features = []
        missing_count = 0
        
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i:i+batch_size]
            
            for _, row in batch_df.iterrows():
                sample_id = row['sample_id']
                
                img_path = None
                for ext in ['jpg', 'jpeg', 'png', 'webp']:
                    potential_path = images_path / f"{sample_id}.{ext}"
                    if potential_path.exists():
                        img_path = potential_path
                        break
                
                if img_path is None:
                    missing_count += 1
                    all_features.append(np.zeros(self.total_dims))
                else:
                    features = self.extract_single(img_path)
                    all_features.append(features)
            
            if self.device == 'cuda' and i % 500 == 0:
                torch.cuda.empty_cache()
        
        features_array = np.array(all_features)
        np.save(output_file, features_array)
        
        print(f"✓ Shape: {features_array.shape}")
        print(f"✗ Missing: {missing_count}")
        
        return features_array


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    extractor = MultiModelFeatureExtractor(device=device)
    
    # Extract training features
    print("\n" + "="*60)
    print("EXTRACTING TRAINING FEATURES (3 MODELS)")
    print("="*60)
    train_features = extractor.extract_all(
        csv_file='data/train.csv',
        images_dir='data/images/train',
        output_file='data/processed/train_multi_image_features.npy',
        batch_size=16  # Lower batch size for 3 models
    )
    
    # Extract test features
    print("\n" + "="*60)
    print("EXTRACTING TEST FEATURES (3 MODELS)")
    print("="*60)
    test_features = extractor.extract_all(
        csv_file='data/test.csv',
        images_dir='data/images/test',
        output_file='data/processed/test_multi_image_features.npy',
        batch_size=16
    )
    
    print("\n" + "="*60)
    print("MULTI-MODEL EXTRACTION COMPLETE")
    print("="*60)
    print(f"Feature dimension: {extractor.total_dims}")


if __name__ == "__main__":
    main()
