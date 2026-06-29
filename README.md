# Waste Material Segregation Using CNN

**Author:** Deepa Aggarwal  
**Domain:** Computer Vision / Deep Learning  
**Framework:** TensorFlow 2.18 / Keras 3.8

---

## Problem Statement

Manual waste sorting is labour-intensive, error-prone, and costly. This project builds an AI-powered image classification system to automatically categorise waste into recyclable and non-recyclable groups, enabling:

- Automated waste sorting at recycling facilities
- Smart bin integration for real-time classification
- Reduced landfill waste through improved recyclable identification

---

## Dataset

| Property | Detail |
|----------|--------|
| Total Images | 7,625 |
| Image Size | 256×256 (original), resized to 128×128 |
| Input Format | RGB JPG/PNG |
| Classes | 7 |
| Label Source | Directory folder names |

### Class Distribution

| Class | Count |
|-------|-------|
| Plastic | 2,295 |
| Paper | 1,030 |
| Other | 1,010 |
| Food Waste | 1,000 |
| Metal | 1,000 |
| Glass | 750 |
| Cardboard | 540 |

> Note: Dataset is imbalanced — Plastic has ~4x more samples than Cardboard.

---

## Project Structure

```
cnn_assignment/
├── Dataset_Waste_Segregation.zip
└── Unzipped_Waste_Segregation/
    └── data/
        ├── Cardboard/
        ├── Food_Waste/
        ├── Glass/
        ├── Metal/
        ├── Other/
        ├── Paper/
        └── Plastic/

dl/
├── CNN_Waste_Segregation_Starter_Deepa_Aggarwal.ipynb   ← Main notebook
├── README.md                                             ← This file
└── Interview_Guide_CNN_Waste_Segregation.md              ← Interview prep guide
```

---

## Methodology

### Phase 1: Data Preparation

1. **Load images** — Custom function reads images from class-named subdirectories using PIL, returns RGB arrays and integer labels.
2. **Visualise distribution** — Bar chart of class counts, sample grid display per class.
3. **Resize** — All images resized from 256×256 to 128×128 to balance quality and compute cost.
4. **Encode labels** — Integer labels converted to one-hot vectors using `to_categorical` → shape (7625, 7).
5. **Split** — 80% train (6100 samples) / 20% validation (1525 samples) with `stratify=True` to preserve class ratios.

### Phase 2: Baseline Custom CNN

```
Input (128×128×3)
    ↓
Conv2D(32, 3×3) + BatchNorm + MaxPool  → 64×64×32
    ↓
Conv2D(64, 3×3) + BatchNorm + MaxPool  → 32×32×64
    ↓
Conv2D(128, 3×3) + BatchNorm + MaxPool → 16×16×128
    ↓
GlobalAveragePooling2D                  → 128
    ↓
Dense(128, relu) + BatchNorm + Dropout(0.6)
    ↓
Dense(7, softmax)
```

- Total parameters: 112,071
- Optimizer: Adam (lr=1e-4), L2 regularisation (1e-4)
- Result: ~60% validation accuracy after 35 epochs

### Phase 3: Transfer Learning — MobileNetV2

```
Input (128×128×3)
    ↓
MobileNetV2 preprocessing (scale to [-1, 1])
    ↓
MobileNetV2 base (frozen) → 4×4×1280
    ↓
GlobalAveragePooling2D → 1280
    ↓
Dense(128, relu) + Dropout(0.5)
    ↓
Dense(7, softmax)
```

- Trainable parameters: 164,871 of 2,422,855 total
- Optimizer: Adam (lr=1e-4)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Cyclical LR
- Result: **84.2% validation accuracy**

---

## Results

### Training Comparison

| Model | Train Accuracy | Val Accuracy | Val Loss |
|-------|---------------|--------------|----------|
| Custom CNN (35 epochs) | ~60% | ~60% | ~1.23 |
| MobileNetV2 Transfer (30 epochs) | — | **84.2%** | **0.48** |

### Per-Class Performance (Transfer Learning Model)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cardboard | 92.3% | 88.9% | ~90% |
| Metal | — | 88% | >87% |
| Food Waste | — | — | >87% |
| Plastic | 84.9% | 84.9% | ~85% |
| Paper | — | — | ~76% |

---

## Key Design Decisions

**Why GlobalAveragePooling2D over Flatten?**  
GAP reduces 16×16×128 feature maps to 128 values (one per channel), dramatically cutting parameters and acting as a spatial regulariser. Flatten would produce 32,768 features going into Dense, causing memory bloat and overfitting.

**Why Transfer Learning after the custom CNN?**  
The custom CNN plateaued at ~60% validation accuracy despite 35 epochs. MobileNetV2 pre-trained on ImageNet provides rich feature representations (edges, textures, shapes) that transfer to waste images, immediately pushing accuracy to 84%+.

**Why Stratified Split?**  
Stratification ensures both train and validation sets have the same class proportions as the full dataset. Without it, the validation set might under-represent rare classes like Cardboard, giving a misleading accuracy estimate.

**Why Cyclical Learning Rate?**  
Uses cosine annealing `abs(cos(epoch/10 * π))` to periodically increase then decrease LR, helping the model escape local minima and converge to a better solution than a monotonically decaying LR alone.

---

## Requirements

```
numpy==1.26.4
pandas==2.2.2
seaborn==0.13.2
matplotlib==3.10.0
Pillow==11.1.0
tensorflow==2.18.0
keras==3.8.0
scikit-learn==1.6.1
```

Install via:
```bash
pip install numpy==1.26.4 pandas==2.2.2 seaborn==0.13.2 matplotlib==3.10.0 Pillow==11.1.0 tensorflow==2.18.0 scikit-learn==1.6.1
```

---

## How to Run

1. Download and place the dataset zip at:  
   `C:/Users/deepa/OneDrive/Desktop/cnn_assignment/Dataset_Waste_Segregation.zip`

2. Open the notebook:  
   `CNN_Waste_Segregation_Starter_Deepa_Aggarwal.ipynb`

3. Run all cells sequentially. The zip is auto-extracted on first run.

4. Best model weights are saved automatically as `best_model.keras`.

---

## Future Improvements

- **Fine-tune MobileNetV2** — Unfreeze the top 30% of layers and retrain at lr=1e-5 for an estimated 3-5% accuracy gain.
- **Data Augmentation** — Random rotation, horizontal flip, brightness, zoom, and shear to improve generalisation and address class imbalance.
- **Class Weighting** — Apply `class_weight` in `model.fit()` to reduce bias toward the majority Plastic class.
- **Upgrade Backbone** — Try EfficientNetB3 or ConvNeXt for higher accuracy at comparable compute.
- **Edge Deployment** — Convert to TensorFlow Lite with INT8 quantisation for smart bin / Jetson Nano deployment.

---

## Conclusions

This project demonstrates a complete deep learning pipeline from raw image data to a production-ready classifier. The two-phase approach — establishing a custom CNN baseline, then leveraging MobileNetV2 transfer learning — illustrates a sound engineering methodology. The final model at **84.2% validation accuracy** is suitable for real-world automated waste segregation applications, with the architectural choice of MobileNetV2 enabling future edge deployment without architectural redesign.
