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

The framework employs a **dual-loss learning strategy** combining supervised
contrastive learning and cross-entropy loss to improve both representation quality
and classification reliability, while maintaining interpretability and computational efficiency.

---

## 🔍 Key Contributions

- Hybrid representation combining transformer embeddings with psycholinguistic features  
- Dual-loss optimization (Cross-Entropy + Supervised Contrastive Learning)  
- Calibration-aware evaluation using ECE and Brier Score  
- Competitive performance compared to traditional ML, transformer, and LLM baselines  
- Ablation analysis validating feature integration and loss design  

---

## ⚠️ Dataset Note

The dataset is derived from publicly available Reddit discussions and represents
**flagged or troll-like behavior**. Labels may reflect subjective interpretations
of online discourse and should be considered as approximations rather than absolute ground truth.

All features are extracted strictly at the **comment level**, without using
user-level metadata, temporal signals, or label-derived attributes.

---

## 🔐 Information Leakage Prevention

- Dataset splits are created **prior to feature extraction**
- No information from validation or test sets is used during training
- No user history, author metadata, or platform-specific features are included

This ensures that reported results reflect genuine generalization performance.

---

## 📊 Key Results

- **Accuracy:** 97.0%  
- **F1-Score:** 0.96  
- **ROC–AUC:** ~0.99  
- **ECE (Expected Calibration Error):** 0.009  
- **Brier Score:** 0.027  

These results demonstrate **strong and competitive performance** relative to
traditional machine learning models, transformer-based approaches, and a
state-of-the-art LLM embedding baseline.

---

## 🧩 Repository Structure
HybridContrastiveTrollDetection/
│
├── data/
│ ├── sample_data.csv
│ └── processed_data.npz
│
├── models/
│ └── hybrid_best.pt
│
├── results/
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ ├── calibration_curve.png
│ └── metrics_report.txt
│
├── src/
│ ├── model_architecture.py
│ └── utils/
│ ├── losses.py
│ ├── metrics.py
│
├── preprocess.py
├── train_hybrid_model.py
├── evaluate.py
├── requirements.txt
└── README.md


---

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

Tested with:
- Python 3.11  
- PyTorch 2.0.1 (CPU)

---

## 🧩 Model Overview

The proposed framework consists of:

1. **Preprocessing**: Text cleaning and normalization  
2. **Hybrid Embedding Construction**:  
   - Transformer embeddings (MPNet)  
   - Psycholinguistic and stylistic features  
3. **Dual-Branch Architecture**:  
   - Projection head (contrastive learning)  
   - Classifier head (cross-entropy loss)  
4. **Dual-Loss Optimization**:  
   - Adaptive weighting between objectives  

---

## 📌 Notes

- The framework does not rely on user-level metadata  
- Designed for **interpretability and deployment efficiency**  
- Emphasizes **calibration and reliability**, not only accuracy  
## 📜 License

This project is intended for academic and research purposes.


