# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Pentagon  
**Team Members:** Utkarsh Kumar, Dhanu Gupta, Ishant Singh, Somil Gupta  
**Submission Date:** October 12, 2025

---

## 1. Executive Summary

Our solution employs a **log-transformed gradient boosting approach** combining multimodal feature extraction (CLIP image embeddings + BERT text embeddings) with data-driven text pattern discovery. By leveraging automated feature engineering that discovers price-predictive keywords, materials, brands, and product categories from training data, we achieved a validated **SMAPE score of 54.81%**. Our key innovation lies in correlation-based feature weighting and log-scale optimization to handle extreme price ranges ($0.13 - $2,796), demonstrating that systematic data analysis can reveal pricing patterns without domain expertise.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

The Smart Product Pricing Challenge required predicting e-commerce product prices from catalog metadata (title, description, bullet points, quantity) and product images. Through extensive exploratory data analysis, we identified several critical patterns:

**Key Observations:**

- **Extreme price distribution:** Training data spans $0.13 to $2,796 (21,500× range), creating severe challenges for linear regression models
- **Skewed distribution:** Median price $14.00 vs mean $23.65 indicates right-skewed distribution requiring log transformation
- **Multimodal signals:** Product information encoded in both visual (product images) and textual (titles, descriptions) formats with complementary information
- **Category clustering:** Products naturally group into distinct categories with different pricing patterns:
  - Tea/Beverage: 26.6% coverage, $28.22 average
  - Food/Snacks: 30.7% coverage, $20.89 average  
  - Organic/Health: 45.1% coverage, $25.05 average
- **Unit standardization critical:** Same products listed in different units (fluid ounces vs ounces, pounds vs grams) required normalization
- **Premium text indicators:** Keywords like "organic" (r=+0.19), "bulk" (+0.09), "premium" correlate positively with price
- **Budget indicators:** Keywords like "ounce" (r=-0.074), "snack" (-0.03), "seasoning" negatively correlate with price
- **Brand value:** Discovered 200 brands with distinct price tiers (Caviar: $233 avg vs generic: $18 avg)
- **Material pricing:** Silver ($33.73), wood ($33.44), organic ($24.90) materials command premium prices
- **Quantity patterns:** Strong correlation between quantity_normalized and price (R²=0.44 on log scale)

### 2.2 Solution Strategy

**Approach Type:** Single Model (LightGBM) with Log Transformation + Data-Driven Feature Discovery  
**Core Innovation:** Automated discovery of price-predictive patterns through correlation analysis, eliminating need for domain expertise

**Pipeline Overview:**

1. **Multimodal Feature Extraction:** CLIP (512-dim image embeddings) + BERT (384-dim text embeddings) for baseline features
2. **Data-Driven Pattern Discovery:** Automated analysis of 75,000 training samples to discover:
   - 200 premium keywords with positive price correlation
   - 21 budget keywords with negative price correlation
   - 19 materials with price multipliers
   - 200 brands with pricing tiers
   - 5 product categories with average prices
3. **Feature Engineering:** 41 advanced text features based on discovered patterns
4. **Log Transformation:** Compress extreme price range (21,500× → 66× on log scale)
5. **Model Training:** LightGBM regression optimized for MAE on log-transformed prices
6. **Inverse Transformation:** Convert log predictions back to original price scale

---

## 3. Model Architecture

### 3.1 Architecture Overview

