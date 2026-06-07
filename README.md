# Bakery Product Quality Classification Using CNN

> **ML Course – Phase 2 | Idea 34 of 75**  
> CNN-Based Image Analysis for Bakery Product Quality — Detecting Good vs. Bad Egg Bread Toast

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/sairam3999devineni/bakery-product-quality-classification-using-cnn)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange?logo=kaggle)](https://www.kaggle.com/datasets/jocelyndumlao/good-and-bad-classification-of-egg-bread-toast)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Overview

This project develops a **CNN-based image classification system** to automatically distinguish between **good-quality** and **bad-quality** egg bread toast. It applies supervised deep learning techniques including custom CNNs and transfer learning (MobileNetV2, EfficientNetB0) to solve a real-world food quality control problem.

| Property | Value |
|---|---|
| **Task** | Binary Image Classification |
| **Classes** | `Good` (832 images) / `Bad` (650 images) |
| **Total Images** | 1,482 |
| **Input Size** | 224 × 224 × 3 |
| **Framework** | TensorFlow / Keras |
| **Environment** | Kaggle Notebook (GPU T4) |

---

## Repository Structure

```
bakery-product-quality-classification/
│
├── bakery_quality_classification.ipynb   # Main notebook (all code)
├── proposal/
│   └── Phase2_Proposal.docx             # Complete proposal document
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
└── outputs/                             # Generated figures (after running)
    ├── class_distribution.png
    ├── sample_images.png
    ├── augmented_samples.png
    ├── custom_cnn_curves.png
    ├── mobilenetv2_curves.png
    ├── efficientnetb0_curves.png
    ├── custom_cnn_cm.png
    ├── mobilenetv2_cm.png
    ├── efficientnetb0_cm.png
    ├── roc_curves.png
    ├── model_comparison_bar.png
    ├── model_comparison.csv
    ├── custom_cnn_gradcam.png
    ├── mobilenetv2_gradcam.png
    ├── efficientnetb0_gradcam.png
    └── error_analysis.png
```

---

## Quick Start

### Option A — Run on Kaggle (Recommended)

1. Open the [Kaggle Notebook](https://www.kaggle.com/code/sairam3999devineni/bakery-product-quality-classification-using-cnn)
2. In the right panel → **Add Input** → search `jocelyndumlao/good-and-bad-classification-of-egg-bread-toast` → Add
3. Go to **Session options → Internet → ON** *(required for ImageNet weights)*
4. Select **Accelerator → GPU T4 x2**
5. Click **Run All**

### Option B — Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/sairam3999devineni/bakery-product-quality-classification.git
cd bakery-product-quality-classification

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Visit: https://www.kaggle.com/datasets/jocelyndumlao/good-and-bad-classification-of-egg-bread-toast
# Place the extracted folder so the structure is:
#   ./Good and Bad classification of  Egg Bread Toast/
#       Bad egg bread/
#       good egg bread/

# 5. Launch Jupyter
jupyter notebook bakery_quality_classification.ipynb
```

> **Note:** The notebook auto-detects the dataset path using `rglob`. No manual path configuration needed.

---

## Models

### 1. Custom CNN (Baseline)
Four-block convolutional network built from scratch:

```
Input (224×224×3)
  → RandomRotation + RandomZoom (augmentation)
  → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
  → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
  → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25)
  → Conv2D(256) → BN → ReLU → MaxPool → Dropout(0.25)
  → GlobalAveragePooling2D
  → Dense(256, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)
```

### 2. MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet | Last 30 layers fine-tuned
- Input: rescaled to `[-1, 1]`
- ~2.3M parameters | Fast inference

### 3. EfficientNetB0 (Transfer Learning)
- Pre-trained on ImageNet | Last 20 layers fine-tuned
- Input: rescaled to `[0, 255]`
- ~4.1M parameters | Best accuracy/efficiency trade-off

---

## Results (Expected)

| Model | Accuracy | F1-Score | ROC-AUC | Params | Infer (ms) |
|---|---|---|---|---|---|
| Custom CNN | ~88% | ~0.87 | ~0.95 | ~0.46M | ~2ms |
| MobileNetV2 | ~93% | ~0.92 | ~0.98 | ~2.3M | ~3ms |
| EfficientNetB0 | ~95% | ~0.94 | ~0.99 | ~4.1M | ~4ms |

> *Actual results depend on training run. Enable internet on Kaggle for ImageNet weights.*

---

## Key Features

- ✅ **Auto-detect dataset path** — works on Kaggle regardless of dataset slug
- ✅ **Keras 3 compatible** — `get_last_conv_layer` uses sub-model probing
- ✅ **Internet-aware** — graceful fallback if ImageNet weights unavailable
- ✅ **Grad-CAM explainability** — visualises what each model focuses on
- ✅ **Class weighting** — handles mild imbalance (56/44 split)
- ✅ **Complete evaluation** — confusion matrix, ROC, classification report, error analysis
- ✅ **All outputs saved** — figures and models persisted to `/kaggle/working/outputs/`

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Good and Bad Classification of Egg Bread Toast |
| **Source** | [Kaggle — jocelyndumlao](https://www.kaggle.com/datasets/jocelyndumlao/good-and-bad-classification-of-egg-bread-toast) |
| **Total** | 1,482 images |
| **Good** | 832 images (56.1%) |
| **Bad** | 650 images (43.9%) |
| **Format** | JPEG, RGB |

---

## Research Questions

| RQ | Question |
|---|---|
| **RQ1** | How accurately can CNNs classify bakery product quality? |
| **RQ2** | Custom CNN vs. transfer learning — best performance/efficiency balance? |
| **RQ3** | How do augmentation and class-weighting affect robustness? |
| **RQ4** | Do Grad-CAM maps focus on meaningful visual quality regions? |
| **RQ5** | What are the main failure modes and deployment limitations? |

---

## Requirements

```
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
```

Install with: `pip install -r requirements.txt`

---

## Submission Details

| Item | Link |
|---|---|
| **Kaggle Notebook** | https://www.kaggle.com/code/sairam3999devineni/bakery-product-quality-classification-using-cnn |
| **Dataset** | https://www.kaggle.com/datasets/jocelyndumlao/good-and-bad-classification-of-egg-bread-toast |
| **GitHub Repo** | https://github.com/sairam3999devineni/bakery-product-quality-classification |

---

## References

- LeCun et al. (2015). Deep learning. *Nature*, 521, 436–444.
- Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR 2018*.
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019*.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017*.
- Dumlao, J. (2025). Good and Bad Classification of Egg Bread Toast. *Kaggle Dataset*.

---

## License

This project is submitted as academic coursework. Dataset used under Kaggle terms of service.
