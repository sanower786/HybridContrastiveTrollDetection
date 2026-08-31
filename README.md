# Hybrid Contrastive–Classification for Detecting Troll-like Behavior

## Overview

Online trolling presents a challenging problem for automated content
moderation because troll-like behavior can exhibit substantial linguistic
variability and overlap with non-trolling discourse.

This repository implements a hybrid contrastive–classification framework
that combines contextual transformer representations with text-derived
linguistic features.

The framework jointly optimizes supervised contrastive learning and
cross-entropy classification. The study evaluates the resulting model using
conventional predictive metrics together with representation-level,
calibration, confidence, and cross-dataset analyses.

The proposed framework is compared with traditional machine-learning
baselines, transformer-based baselines, a pretrained language-embedding
baseline, and a representative hybrid architecture.

## Key Contributions

- Hybrid representation integrating contextual transformer embeddings with
  complementary linguistic features.
- Joint optimization using supervised contrastive learning and
  cross-entropy loss.
- Projection head for representation-level contrastive learning.
- Classification head for troll/non-troll prediction.
- Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Calibration analysis using Expected Calibration Error (ECE) and Brier Score.
- Representation analysis using embedding visualization and cluster-validity
  measures.
- Controlled comparison with a classification-only baseline.
- Separate pretrained MiniLM + Logistic Regression baseline.
- Independent cross-dataset evaluation on the TRAC aggression dataset.

# Experimental Setting

## Canonical Reddit Dataset

The primary experiments use a deduplicated Reddit dataset containing:

- **7,924 unique text instances**
- Binary classification
- **5,546 training instances**
- **1,189 validation instances**
- **1,189 test instances**

The canonical test set contains:

| Class | Test Instances |
|-------|---------------:|
| Non-Troll | 219 |
| Troll | 970 |
| **Total** | **1,189** |

The train/validation/test partitions are stratified using a fixed random seed
of **42**.

The canonical test partition is frozen and is used consistently for the
principal model comparisons.

# Benchmark Models

The study evaluates models from several methodological categories.

| Category | Model |
|----------|-------|
| Traditional Machine Learning | TF-IDF + Logistic Regression |
| Traditional Machine Learning | TF-IDF + SGD |
| Transformer Baseline | BERT |
| Transformer Baseline | DistilBERT |
| Pretrained Language Embedding | all-MiniLM-L6-v2 + Logistic Regression |
| Hybrid Baseline | BCBGA |
| Controlled Ablation | Classification-only |
| Proposed | Hybrid Contrastive–Classification |

  Input Comment
                         |
                         v
                Text Preprocessing
                         |
              +----------+----------+
              |                     |
              v                     v
       MPNet Representation    Linguistic Features
              |                     |
              +----------+----------+
                         |
                         v
               Hybrid Representation
                         |
                         v
                  Shared Network
                         |
                +--------+--------+
                |                 |
                v                 v
         Projection Head    Classification Head
                |                 |
                v                 v
          SupCon Loss          CE Loss
                |                 |
                +--------+--------+
                         |
                         v
                 Joint Optimization

## Information Leakage Prevention

The experimental pipeline follows strict separation between training, validation, and test data.

- Dataset partitioning is performed before model training and feature learning.
- No validation or test samples are used during optimization.
- Feature normalization statistics are obtained from the training partition and subsequently applied to validation and test data.
- No user history or author-level metadata is used.
- No temporal or label-derived attributes are incorporated.
- All models are evaluated on the same canonical Reddit test partition.

These procedures provide a consistent basis for comparing the evaluated approaches.

---


# Pretrained Language-Embedding Baseline

A separate pretrained language-embedding baseline is included to evaluate the contribution of contextual semantic representations independently of the proposed hybrid architecture.

### Model

**sentence-transformers/all-MiniLM-L6-v2**

Configuration:

- Embedding dimension: **384**
- Device: CUDA
- Batch size: **32**
- Normalized embeddings: **True**
- Classifier: Logistic Regression

The embeddings are generated independently for the training, validation, and canonical test partitions and stored for reproducibility.

### Final Reddit Test Performance

| Metric | MiniLM + Logistic Regression |
|--------|-----------------------------:|
| Accuracy | **92.43%** |
| Precision | **0.9297** |
| Recall | **0.9814** |
| F1-score | **0.9549** |
| ROC-AUC | **0.9609** |

The corresponding canonical test confusion matrix is:

| | Predicted Non-Troll | Predicted Troll |
|---|---:|---:|
| **True Non-Troll** | 147 | 72 |
| **True Troll** | 18 | 952 |

The ROC curve and confusion matrix for this pretrained language-embedding baseline are provided separately in the repository results.

---

# Main Experimental Results

The final canonical Reddit test results reported in the manuscript are summarized below.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|---------:|----------:|-------:|----:|--------:|
| TF-IDF + Logistic Regression | 88.14% | 0.8772 | 0.9938 | 0.9319 | 0.9292 |
| TF-IDF + SGD | 90.08% | 0.8981 | 0.9907 | 0.9422 | 0.9314 |
| BERT | 69.30% | 0.8122 | 0.8113 | 0.8118 | 0.4901 |
| DistilBERT | 81.58% | 0.8158 | 1.0000 | 0.8986 | 0.5214 |
| Classification-only | 97.56% | 0.9846 | 0.9856 | 0.9851 | 0.9957 |
| **Proposed Hybrid** | **97.31%** | **0.9796** | **0.9876** | **0.9836** | **0.9947** |

The pretrained MiniLM baseline is reported separately because it constitutes an additional language-embedding baseline:

**MiniLM + Logistic Regression: Accuracy = 92.43%, F1 = 0.9549, ROC-AUC = 0.9609.**

---

# Confusion Matrix and ROC Analysis

The repository contains separate visualizations for the pretrained MiniLM baseline and the main experimental models.

### Pretrained MiniLM Baseline

- `results/minilm_confusion_matrix.png`
- `results/minilm_roc_curve.png`

The MiniLM baseline achieves a ROC-AUC of **0.9609** on the canonical Reddit test set.

### Main Model Comparisons

ROC curves are provided for:

- TF-IDF + Logistic Regression
- TF-IDF + SGD
- BERT
- DistilBERT
- Classification-only ablation
- Proposed Hybrid

The multi-panel ROC analysis uses the same canonical Reddit test partition (**N = 1,189**) for all models.

---

# Ablation Analysis

Controlled ablation experiments are performed to examine the contribution of the proposed learning components.

The classification-only configuration provides a direct comparison with the complete hybrid contrastive–classification framework.

The ablation analysis is intended to determine whether the observed behavior can be attributed solely to the use of a hybrid neural architecture or whether the learning formulation and representation design also contribute to the resulting performance.

---

# Cross-Dataset Evaluation

The proposed framework is additionally evaluated on the independent TRAC aggression dataset.

On TRAC, the model obtained:

- **Accuracy:** 57.96%
- **Macro-F1:** 0.4098
- **ROC-AUC:** 0.4985

These results indicate substantial distributional and task differences between the Reddit troll-detection setting and the TRAC aggression benchmark.

The cross-dataset results are therefore reported as an independent robustness assessment and should not be interpreted as evidence of uniformly strong transfer across datasets.

---

# Model Architecture

The proposed framework consists of the following major stages.

### 1. Text Preprocessing

- Text cleaning and normalization
- Tokenization
- Linguistic feature preparation

### 2. Hybrid Representation

The framework combines:

- Contextual transformer representations
- Psycholinguistic features
- Stylistic features
- Lexical and readability-related features

### 3. Shared Representation Network

The resulting representation is processed through a shared learning architecture containing:

- A projection head for supervised contrastive learning
- A classification head for troll prediction

### 4. Joint Optimization

The model is trained using a combined objective consisting of:

- Cross-Entropy Loss
- Supervised Contrastive Loss

The two objectives are jointly optimized through the loss-weighting formulation described in the paper.

### 5. Evaluation

The framework is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Expected Calibration Error (ECE)
- Brier Score
- Confusion matrices
- ROC curves
- Confidence analysis
- Cross-dataset evaluation


# Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | `2e-5` |
| Weight Decay | `0.01` |
| Maximum Epochs | `5` |
| Batch Size | `16` |
| Evaluation Batch Size | `32` |
| Maximum Sequence Length | `128` |
| Random Seed | `42` |
| Early Stopping Patience | `2` |

Early stopping is based on validation F1-score, and the checkpoint achieving the best validation F1-score is retained for final evaluation on the held-out canonical test set.


# Reliability Evaluation

In addition to classification performance, the study considers prediction reliability using:

- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams
- Confidence distribution analysis

These measures complement conventional classification metrics by examining the quality of model confidence.



HybridContrastiveTrollDetection/
│
├── data/
│   ├── reddit/
│   ├── trac/
│   └── processed/
│
├── Image/
│
├── results/
│   ├── Figure_MiniLM_ROC_Curve.png
│   ├── Figure_MiniLM_Confusion_Matrix.png
│   ├── confusion_matrix/
│   ├── roc/
│   
│
├── src/
│   ├── __init__.py
│   ├── model_architecture.py
│   └── utils/
│       
│
├── preprocess.py
├── train_hybrid_model.py
├── evaluate.py
│
├── smoke_test.py
├── smoke_test.sh
├── run_smoke.sh
│
├── requirements.txt
├── .gitignore
└── README.md