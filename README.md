# Microchip Defect Detection

A deep learning-based computer vision pipeline for automated microchip surface defect detection and classification.

This project focuses on transfer learning, image classification, and industrial visual inspection workflows using lightweight convolutional neural networks optimized for fast training and inference.

---

# Features

- Automated defect classification across multiple defect categories
- Transfer learning with pretrained MobileNet architectures
- End-to-end training and evaluation pipeline
- Confusion matrix and classification report generation
- Data augmentation for improved model robustness
- Model checkpointing and reproducible training
- Benchmarking between multiple CNN architectures
- Lightweight deployment-friendly inference pipeline
- Apple Silicon (MPS), CUDA, and CPU support

---

# Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **100.00%** (3000/3000) |
| Per-class Accuracy | **100% across all classes** |

The model converges rapidly due to the consistency of the dataset and the effectiveness of transfer learning for texture and surface pattern recognition tasks.

---

# Defect Samples & Model Performance

<table align="center">
<tr>
<td align="center">
<b>Crack</b><br><br>
<img src="assets/crack_sample.png" width="230"/>
</td>

<td align="center">
<b>Hole</b><br><br>
<img src="assets/hole_sample.png" width="230"/>
</td>

<td align="center">
<b>Normal</b><br><br>
<img src="assets/normal_sample.png" width="230"/>
</td>
</tr>

<tr>
<td align="center">
<b>Rust</b><br><br>
<img src="assets/rust_sample.png" width="230"/>
</td>

<td align="center">
<b>Scratch</b><br><br>
<img src="assets/scratch_sample.png" width="230"/>
</td>

<td align="center">
<b>Confusion Matrix</b><br><br>
<img src="results/confusion_matrix.png" width="230"/>
</td>
</tr>
</table>

---

The model demonstrates strong classification performance across all manufacturing defect categories with near-perfect separation between classes.

---

# Tech Stack

- Python
- PyTorch
- Torchvision
- MobileNetV2 / MobileNetV3
- NumPy
- Scikit-learn
- Matplotlib
- PIL
- CUDA / Apple Silicon (MPS)

---

# Dataset Structure

```bash
dataset/
├── train/
│   ├── crack/
│   ├── hole/
│   ├── normal/
│   ├── rust/
│   └── scratch/
├── test/
│   ├── crack/
│   ├── hole/
│   ├── normal/
│   ├── rust/
│   └── scratch/
└── metadata.csv
```

The dataset contains categorized microchip surface images used for supervised defect classification.

---

# Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

# Training

Train the model:

```bash
python3 train.py --data_dir ./dataset --output_dir ./results
```

Run evaluation only using a saved checkpoint:

```bash
python3 train.py --skip_train --data_dir ./dataset --output_dir ./results
```

---

# Available Arguments

```bash
--data_dir      Path to dataset
--output_dir    Output directory
--epochs        Number of training epochs
--batch_size    Batch size
--lr            Learning rate
--seed          Random seed
--skip_train    Skip training and load checkpoint
```

---

# Model Architecture

## MobileNetV2 Transfer Learning

The primary model uses a pretrained MobileNetV2 backbone with a custom classification head for multi-class defect recognition.

Key advantages:

- Fast convergence
- Lightweight architecture (~3.4M parameters)
- Efficient inference
- Strong transfer learning performance
- Suitable for edge and embedded deployment

---

# Data Augmentation

Training augmentations include:

- Horizontal flips
- Vertical flips
- Mild color jitter

More aggressive augmentations were intentionally avoided to preserve important defect morphology and texture characteristics.

---

# Training Strategy

- Optimizer: Adam
- Weight decay: 1e-4
- Cosine annealing learning rate schedule
- Best checkpoint selected using validation accuracy
- Fully reproducible training using fixed random seeds

---

# Generated Outputs

| File | Description |
|------|-------------|
| `predictions.csv` | Predicted labels for test images |
| `confusion_matrix.png` | Confusion matrix visualization |
| `classification_report.txt` | Precision, recall, and F1 metrics |
| `model_checkpoint.pth` | Best saved model |
| `training_history.json` | Epoch-by-epoch training metrics |

---

# Additional Experiments

## MobileNet Benchmarking

Additional benchmarking was performed comparing:

- MobileNetV2
- MobileNetV3-Small

across multiple epoch configurations to evaluate training efficiency and convergence speed.

## CNN From Scratch

A custom CNN trained without pretrained weights was also implemented as a baseline comparison.

While the scratch model achieved strong performance, transfer learning consistently provided:

- Faster convergence
- Higher efficiency
- Better early-stage accuracy

---

# Future Improvements

- Real-world semiconductor inspection datasets
- Real-time inference optimization
- Defect localization using object detection
- Vision Transformer (ViT) experimentation
- Quantization and edge deployment
- ONNX / TensorRT export support

---

# Conclusion

This project demonstrates how transfer learning and lightweight CNN architectures can be effectively applied to automated visual inspection tasks. The pipeline achieves high classification accuracy while remaining computationally efficient and deployment-friendly.
