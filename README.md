# 🧠 Hybrid Contrastive–Classification for Detecting Troll-like Behavior

📖 **Overview**

This repository contains the official implementation of the research paper:

**Hybrid Contrastive–Classification for Detecting Troll-like Behavior:  
Integrating Psycholinguistic Features with Calibration-Aware Learning**

Online trolling presents a significant challenge for automated moderation systems,
requiring models that are both accurate and reliable under ambiguous discourse.
This work proposes a **hybrid contrastive–classification framework** that integrates
contextual transformer embeddings with text-derived psycholinguistic and stylistic
features.

The proposed framework jointly optimizes supervised contrastive learning and cross-entropy loss to simultaneously improve representation quality, predictive performance, and confidence reliability. Unlike conventional text classification methods, the framework explicitly incorporates calibration-aware evaluation through Expected Calibration Error (ECE) and Brier Score, providing a more reliable solution for large-scale online content moderation.
## 🔍 Key Contributions

- Hybrid representation combining transformer embeddings with psycholinguistic features  
- Dual-loss optimization (Cross-Entropy + Supervised Contrastive Learning)  
- Calibration-aware evaluation using ECE and Brier Score  
- Competitive performance compared to traditional machine learning, transformer-based, and representative hybrid architectures
- Ablation analysis validating feature integration and loss design  

## 📈 Benchmark Models

The proposed framework is evaluated against representative benchmark models spanning multiple methodological categories.

| Category | Models |
|----------|--------|
| Traditional Machine Learning | Logistic Regression, Linear SVM, Random Forest |
| Transformer Models | BERT, DistilBERT |
| Hybrid Architecture | BCBGA (Bao et al.) |
| Ablation | Proposed (Cross-Entropy Only) |
| Proposed | Hybrid Contrastive–Classification |


## 📂 Datasets

The proposed framework is evaluated on two publicly available English-language datasets representing complementary abusive language detection scenarios.

### Reddit Troll Dataset
- Primary benchmark for troll-like behavior detection
- Binary classification
- Comment-level annotations
- Used for model development and primary evaluation

### TRAC Dataset
- Independent benchmark for aggression detection
- Binary aggression classification
- Used to evaluate cross-dataset generalization and robustness
- Provides a complementary annotation paradigm for online abusive language

Both datasets are publicly available for research purposes. Detailed preprocessing and data partitioning procedures are described in the accompanying paper.
  
  ## ⚠️ Dataset Notes

The datasets are derived from publicly available online discussions and represent annotated troll-like or aggressive behavior. Since abusive language annotation is inherently subjective, labels should be interpreted as approximations rather than absolute ground truth.

All features are extracted strictly at the comment level without using user-level metadata, temporal information, or label-derived attributes.

## 🔐 Information Leakage Prevention

- Dataset partitioning is performed before feature extraction and model training. Feature normalization statistics are computed exclusively on the training set and subsequently applied to the validation and test sets. No information from the validation or test partitions is used during model optimization.
- No information from validation or test sets is used during training
- No user history, author metadata, or platform-specific features are included

This ensures that reported results reflect genuine generalization performance.

---

### 📊 Experimental Results

### Reddit Benchmark

- Accuracy: 97.0%
- F1-score: 0.96
- ROC-AUC: ~0.99
- ECE: 0.009
- Brier Score: 0.027

### Cross-Dataset Evaluation (TRAC)

The proposed framework maintains competitive predictive performance and satisfactory calibration on the independent TRAC aggression benchmark, demonstrating its robustness and generalization capability across complementary abusive language detection tasks.

## 🧩 Repository Structure


HybridContrastiveTrollDetection/
│
├── data/
│   ├── reddit/
│   ├── trac/
│   └── processed/
│
├── baselines/
│   ├── logistic_regression.py
│   ├── svm.py
│   ├── random_forest.py
│   ├── bert.py
│   ├── distilbert.py
│   └── bcbga.py
│
├── models/
│   └── hybrid_best.pt
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── calibration_curve.png
│   ├── confidence_distribution.png
│   ├── reliability_diagram.png
│   └── metrics_report.txt
│
├── src/
│   ├── model_architecture.py
│   ├── hybrid_embedding.py
│   └── utils/
│       ├── losses.py
│       ├── metrics.py
│       ├── calibration.py
│       └── visualization.py
│
├── preprocess.py
├── train_hybrid_model.py
├── evaluate.py
├── requirements.txt
└── README.md


## 🚀 Run Instructions

### 1. Clone the repository

git clone https://github.com/sanower786/HybridContrastiveTrollDetection.git

cd HybridContrastiveTrollDetection


### 2. Create environment and install dependencies

python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt


### 3. Train the model

python train_hybrid_model.py


### 4. Evaluate the model

python evaluate.py


---

## 🧪 Execution Proof

The training pipeline has been tested on sample data.

- Training script executes successfully  
- Evaluation metrics and visualizations are generated  
- Results are saved automatically in the `results/` directory  

- Tested with

 Python 3.11

 PyTorch 2.5.1

 CUDA 12.1

 NVIDIA RTX A4000 GPU



## 🧩 Model Overview

The proposed framework is designed to provide robust and reliable troll detection by integrating complementary linguistic representations within a unified calibration-aware learning framework. The model is evaluated on both the Reddit Troll Dataset and the TRAC Aggression Dataset to assess predictive performance, confidence reliability, and cross-dataset generalization.

The framework consists of the following stages:

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Tokenization and feature preparation

2. **Hybrid Feature Representation**
   - Contextual transformer embeddings (MPNet)
   - Psycholinguistic features
   - Stylistic features
   - Lexical and readability features

3. **Hybrid Contrastive–Classification Architecture**
   - Shared feature encoder
   - Projection head for supervised contrastive learning
   - Classification head for troll prediction

4. **Dual-Loss Optimization**
   - Cross-Entropy Loss
   - Supervised Contrastive Loss
   - Joint optimization through adaptive loss weighting

5. **Evaluation and Reliability Analysis**
   - Classification metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC)
   - Calibration metrics (ECE and Brier Score)
   - Confidence distribution analysis
   - Cross-dataset evaluation on the TRAC benchmark
   ## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Epochs | 5 |
| Batch Size | 16 |
| Evaluation Batch Size | 32 |
| Maximum Sequence Length | 128 |
| Random Seed | 42 |

## 📉 Reliability Evaluation

Beyond conventional classification metrics, the framework evaluates prediction reliability using:

- Expected Calibration Error (ECE)

- Brier Score

- Reliability Diagram

- Confidence Distribution Analysis

 ## 🔬 Reproducibility

To facilitate reproducible research:

• Fixed random seed (42)

• Stratified Train/Validation/Test split

• Unified preprocessing pipeline

• Standardized hyperparameters across transformer baselines

• Evaluation on independent datasets

• Public implementation

This project is intended for academic and research purposes.


