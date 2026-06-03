# Multi-classification-Mobile-Price-Prediction
# 📱 Mobile Price Prediction · Multiclass Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Dash](https://img.shields.io/badge/Plotly%20Dash-2.x-008DE4?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

> **Predict whether a phone is Budget, Economy, Mid-Range, or Premium — using RAM, battery, pixels and more.**  
> Three classifiers. One interactive dashboard. Zero guesswork.

</div>

---

## 🗂️ Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models & Results](#-models--results)
- [Visual Highlights](#-visual-highlights)
- [Dashboard](#-interactive-dashboard)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Feature Engineering](#-feature-engineering)
- [Contributing](#-contributing)

---

## 🔍 Overview

This project tackles a real-world multiclass classification problem: **predicting mobile phone price categories** based on hardware specifications. Four price tiers are modelled:

| Class | Label | Description |
|-------|-------|-------------|
| `0` | 💚 Budget | Entry-level devices |
| `1` | 🔵 Economy | Mid-low segment |
| `2` | 🟡 Mid-Range | Performance value |
| `3` | 🔴 Premium | Flagship devices |

Three production-ready ML algorithms are benchmarked head-to-head with 5-fold cross-validation.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Samples** | 2,000 |
| **Features** | 20 hardware specifications |
| **Target** | `price_range` (0 – 3) |
| **Missing Values** | None |
| **Class Balance** | Perfectly stratified |

### Key Features
```
battery_power  · blue        · clock_speed  · dual_sim
fc             · four_g      · int_memory   · m_dep
mobile_wt      · n_cores     · pc           · px_height
px_width       · ram ★       · sc_h         · sc_w
talk_time      · three_g     · touch_screen · wifi
```
> ★ **RAM** is the strongest single predictor — correlation with price_range ≈ **0.92**

---

## 🏆 Models & Results

### Accuracy Comparison

```
┌──────────────────────┬──────────────┬───────────┬─────────┐
│ Model                │ Test Acc     │ CV Mean   │ CV Std  │
├──────────────────────┼──────────────┼───────────┼─────────┤
│ KNN (k=5)            │   91.50 %    │  90.83 %  │ ±0.82 % │
│ Logistic Regression  │   97.25 %    │  96.94 %  │ ±0.48 % │
│ SVM (RBF kernel)     │   97.75 % 🏆 │  97.31 %  │ ±0.41 % │
└──────────────────────┴──────────────┴───────────┴─────────┘
```

### SVM – Best Model Classification Report

```
               precision    recall  f1-score   support

     Budget       0.98      0.98      0.98       100
    Economy       0.97      0.96      0.97       100
  Mid-Range       0.97      0.97      0.97       100
    Premium       0.99      0.99      0.99       100

   accuracy                           0.977       400
  macro avg       0.978     0.977     0.977       400
weighted avg       0.978     0.977     0.977       400
```

### 📈 Visual: Model Accuracy Bar Chart

```
 Accuracy
 100% ┤
  97% ┤          ████████   ████████
  94% ┤          ████████   ████████
  91% ┤ ████████ ████████   ████████
  88% ┤ ████████ ████████   ████████
      └──────────────────────────────
        KNN      LogReg      SVM ← 🏆
```

---

## 🎨 Visual Highlights

### 🔥 Correlation Heatmap

The heatmap reveals that **RAM dominates** all other features in its relationship with price range. `px_width`, `battery_power`, and `px_height` also show strong positive correlations.

```
Features most correlated with price_range:
  ram            ████████████████████  +0.917 ★
  px_width       ████████████░░░░░░░░  +0.165
  battery_power  ████████████░░░░░░░░  +0.200
  px_height      ███████████░░░░░░░░░  +0.149
  int_memory     ██████████░░░░░░░░░░  +0.044

Features negatively correlated:
  mobile_wt      ░░░░░░░░░███████████  -0.030
  m_dep          ░░░░░░░░░███████████  -0.019
```

### 📦 RAM vs Price Range (Box Plot)

```
RAM (MB)
 4000 ┤                              ╔═══╗
 3000 ┤                    ╔═══╗     ║   ║
 2000 ┤         ╔═══╗      ║   ║     ║   ║
 1000 ┤╔═══╗    ║   ║      ║   ║     ║   ║
    0 ┤║   ║    ║   ║      ║   ║     ║   ║
      └──────────────────────────────────
       Budget  Economy  Mid-Range  Premium
```

> Higher price tiers clearly cluster at higher RAM values — a beautifully linear relationship.

---

## 🖥️ Interactive Dashboard

A full **Plotly Dash** dashboard is included, featuring:

| Panel | Content |
|-------|---------|
| 📊 KPI Cards | Best accuracy, dataset size, feature count, class count |
| 📉 Accuracy Bar | Test vs CV accuracy for all 3 models, side-by-side |
| 🕸️ Radar Chart | Test vs CV mean across models |
| 🔲 Confusion Matrix | Interactive — switch between models |
| 📊 Per-Class F1 | Bar chart tied to confusion matrix selector |
| 🌡️ Correlation Heatmap | Full 20×20 feature correlation matrix |
| 📦 RAM Box Plot | Distribution of RAM by price tier |
| 📋 Metrics Table | Accuracy, CV, Precision, Recall, F1 (weighted) |

**Dashboard Preview:**
```
╔═══════════════════════════════════════════════════════════╗
║ 📱 Mobile Price Prediction             🏆 Best: SVM       ║
╠════════════╦═══════════╦═════════════╦════════════════════╣
║ Best Acc   ║  Dataset  ║  Features   ║  Price Classes     ║
║  97.75%    ║  2,000    ║    20       ║      4             ║
╠════════════╩═══════════╩═════════════╩════════════════════╣
║  [Accuracy Bar Chart]          [Radar: Test vs CV]        ║
╠═══════════════════════════════════════════════════════════╣
║  [Confusion Matrix – toggle]   [Per-Class F1-Score]       ║
╠═══════════════════════════════════════════════════════════╣
║  [Feature Importance]          [RAM Box + Price Dist]     ║
╠═══════════════════════════════════════════════════════════╣
║  [Full Correlation Heatmap]                               ║
╠═══════════════════════════════════════════════════════════╣
║  [Model Metrics Table: Acc · CV · Precision · Recall · F1]║
╚═══════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
mobile-price-prediction/
│
├── 📓 MOBILE_PRICE_PREDICTION-MULTICLASS_CLASSIFICATION.ipynb
│       Full exploratory + modelling notebook
│
├── 🖥️  dashboard.py
│       Plotly Dash interactive dashboard (this file)
│
├── 📄  mobile_price.csv
│       Dataset (2000 rows × 21 columns)
│
├── 📋  requirements.txt
│       All dependencies
│
└── 📖  README.md
```

---

## ⚡ Quick Start

### 1 — Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/mobile-price-prediction.git
cd mobile-price-prediction

pip install -r requirements.txt
```

### 2 — Run the Dashboard

```bash
python dashboard.py
# → Open http://127.0.0.1:8050
```

### 3 — Explore the Notebook

```bash
jupyter notebook MOBILE_PRICE_PREDICTION-MULTICLASS_CLASSIFICATION.ipynb
```

### requirements.txt
```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.3
plotly>=5.18
dash>=2.14
dash-bootstrap-components>=1.5
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

---

## 🔬 Feature Engineering

### Pipeline

```
Raw CSV
   │
   ▼
train_test_split (80/20, stratified)
   │
   ▼
StandardScaler (fit on train only)
   │
   ├──► KNeighborsClassifier(k=5)
   ├──► LogisticRegression(max_iter=1000)
   └──► SVC(kernel='rbf')
         │
         ▼
   accuracy_score + classification_report + cross_val_score(cv=5)
```

### Why StandardScaler?
- KNN is **distance-based** — features at different scales heavily distort neighbour ranking.  
- Logistic Regression and SVM both benefit from feature normalization for faster, stable convergence.
- Scaler is **fit only on training data** to prevent data leakage.

### Why SVM wins?
The RBF kernel maps features into a higher-dimensional space where the four price classes become **linearly separable** — a textbook strength of kernel SVMs on structured tabular data.

---

## 🤝 Contributing

Contributions welcome! Ideas to extend this project:

- [ ] Add Random Forest, XGBoost, LightGBM comparison
- [ ] Hyperparameter tuning (GridSearchCV / Optuna)
- [ ] SHAP feature importance visualization
- [ ] Deploy dashboard to Heroku / Render
- [ ] Add real-time prediction form to dashboard

```bash
# Fork → Branch → PR
git checkout -b feature/random-forest-comparison
```

---

## 📜 License

MIT © 2024 — Free to use, modify, and distribute.

---

<div align="center">

**Built with ❤️ using Python · Scikit-Learn · Plotly Dash .DONE BY HARINI.P**

⭐ Star this repo if it helped you!

</div>
