# 🛒 Amazon ML Challenge 2025: From Data to Top 15% in 3 Days  
### A Deep Dive into Multi-Modal Product Price Prediction

Welcome to the repository for my **Top 24% (Rank 1,267 / 7000+)** solution for the **Amazon ML Challenge 2025**.  
In just **3 days**, this project evolved from a simple baseline into a **multi-modal fusion pipeline** that combines product images and text descriptions to predict prices accurately.

---

## 🚀 The Journey: 3-Day Sprint to 50.31 SMAPE

Every ML competition is a story of iteration.  
We started simple and improved step-by-step — validating every choice.

| Stage | Technique Added | Features | Validation SMAPE | Δ Improvement |
|:------|:----------------|:----------:|:----------------:|:-------------:|
| Baseline | Simple CSV Features | ~1,000 | 51.82 | — |
| Stage 1 | + Multi-Model Image Embeddings | 5,159 | 51.20 | ↓ 0.62 |
| Stage 2 | + Advanced TF-IDF Features | 5,309 | 50.61 | ↓ 0.59 |
| Stage 3 | + Optuna Hyperparameter Tuning | 5,312 | 50.31 | ↓ 0.30 |
| Stage 4 | − Removed Low-Impact Features | 5,306 | 50.31 | ✓ No change |
| Stage 5 | Experimental (Rejected) | 5,359 | 50.48 | ↑ 0.17 |

📊 **Total Improvement:** 51.82 → 50.31 SMAPE (−1.51)

---

## 🗓️ Competition Timeline

| Date | Highlights |
|------|-------------|
| **Oct 10** | 🏁 Competition Kick-off & Initial EDA |
| **Oct 11** | 💡 Built baseline (51.82 SMAPE) |
| **Oct 12** | 🖼️ Added 4 image embedding models → 50.61 |
| **Oct 13** | ⚙️ Optuna tuning → 50.31 and 🏆 Final submission & ensemble confirmed |
---

## 🧠 Solution Architecture: Multi-Modal Fusion Pipeline
```ruby
📦 Input Data (Images, Text, Metadata)
┣━━ 🖼️ Image Pipeline
┃ ┣• Pretrained Models: ResNet50, EfficientNetB0, ViT, DenseNet121
┃ ┗• Output → 5,109-dim embeddings
┣━━ 📝 Text Pipeline
┃ ┣• TF-IDF (150 feats)
┃ ┣• Word/Char counts (6 feats)
┃ ┣• Engineered NLP feats (47 feats)
┃ ┗• Output → 197-dim text/meta feats
┗━━ 🧠 Fusion & Modeling
┣• Merge → 5,306 features
┣• Gradient Boosting Core → LightGBM
┣• Ensemble → 70% LGBM + 30% XGBoost
┗• 💰 Final Price Predictions
```
---

## 💡 Key Learnings

- **Quality > Quantity** → 5,306 well-selected features beat 5,359 noisy ones.  
- **Strong validation = confidence** → our local CV matched leaderboard.  
- **Iterate wisely** → start simple, measure, then scale.  
- **Smart ensembling** → 70/30 LGBM-XGB worked better than 50/50.

---

## 🔧 Technology Stack

| Category | Tools |
|-----------|-------|
| Data Handling | Python 3.10, NumPy, Pandas |
| ML Modeling | LightGBM 3.3.5, XGBoost 1.7.0, scikit-learn |
| Deep Learning | PyTorch, Hugging Face Transformers, Torchvision |
| Tuning | Optuna |
| NLP | NLTK |

---

## 🧩 Feature Engineering Highlights

### 🖼️ Multi-Model Image Embeddings  
Combined **ResNet**, **EfficientNet**, **ViT**, and **DenseNet** for diverse visual signals.

### 📝 Text & Metadata Features  
- **Premium Keywords:** detect “luxury”, “professional”, etc.  
- **Budget Keywords:** detect “cheap”, “basic”, etc.  
- **Quantity Extraction:** regex-based “500g”, “2 pcs” → normalized numeric feature.  

---

## 📊 Model Performance Deep Dive

| Model | Features | Train SMAPE | Val SMAPE | Gap |
|:------|:----------:|:-------------:|:------------:|:------:|
| LightGBM | 5,306 | 12.05 | **50.31 ✓** | 38.26 |
| XGBoost | 5,359 | 5.70 | 50.50 | 44.80 |

✅ **Chosen:** LightGBM → better generalization.

---

## 🔝 Top 10 Important Features

| Source | Feature | Importance |
|:--------|:------------------------------|:-----------:|
| 🖼️ Image | Image_ResNet_feature_1024 | 0.0234 |
| 🖼️ Image | Image_EfficientNet_feat_512 | 0.0198 |
| 📝 Text | TF-IDF_luxury | 0.0156 |
| 📝 Text | Text_char_count | 0.0142 |
| 📊 Metadata | Category_electronics | 0.0128 |
| 🖼️ Image | Image_ViT_feature_384 | 0.0115 |
| 📝 Text | Premium_score | 0.0109 |
| ⚖️ Engineered | Quantity_value | 0.0098 |
| 📝 Text | TF-IDF_brand | 0.0087 |
| 🖼️ Image | Image_DenseNet_feat_768 | 0.0081 |

---

## 🔮 Future Improvements

### Short-Term (~49.5 SMAPE)
- [ ] Replace TF-IDF with **DeBERTa / Sentence-Transformer** embeddings.  
- [ ] Category-wise specialized models.  
- [ ] Meta-model stacking ensemble.

### Long-Term (~48.5 SMAPE)
- [ ] **Vision-Language models (CLIP)** for joint embeddings.  
- [ ] **External data**: competitor pricing, brand info, etc.

---

## ⚙️ Setup & Usage

### Prerequisites
- Python 3.8+
- CUDA 11.8+
- 16GB+ RAM, 60GB+ Disk

### 🧭 Quick Start

```bash
# 1️⃣ Clone
git clone https://github.com/yourusername/amazon-ml-2025.git
cd amazon-ml-2025

# 2️⃣ Setup environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 3️⃣ Install dependencies
pip install -r requirements.txt
🏃 Run the Pipeline
bash
Copy code
# Image features (~4h)
python src/features/image_embeddings.py --data_dir data/raw --output_dir data/processed

# Text features (~30m)
python src/features/text_features.py --train_file data/train.csv --output_dir data/processed

# Train model (~10m)
python src/models/train_lgb.py --feature_dir data/processed --output models/lgb_model.pkl

# Generate predictions
python inference.py \
  --model models/lgb_optimized.pkl \
  --test_features data/processed/test_features_final.npy \
  --output test_predictions.csv
<details> <summary>📁 <strong>Project Structure</strong></summary>
css
Copy code
</details>
📚 References
Papers

DeBERTa: Decoding-enhanced BERT with Disentangled Attention

EfficientNet: Rethinking Model Scaling

Vision Transformer (ViT)

Libraries

LightGBM Docs

Hugging Face Transformers

🤝 Contributing & Contact
Contributions welcome!

Author: Dhanu Gupta,Utkarsh Kumar,Somil Gupta, Ishant Singh [Team: PENTAGON]

GitHub: @Dhanugupta0

LinkedIn: [dhanugupta0](https://www.linkedin.com/in/dhanugupta0/)
```
<div align="center">
Made with ❤️ and lots of ☕

⭐ If you found this project insightful, consider giving it a star! ⭐

</div>
