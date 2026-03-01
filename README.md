#  BraTS 2020 — Brain Tumor Segmentation with SwinUNETR

> 3D volumetric brain tumor segmentation on the BraTS 2020 dataset using **SwinUNETR** (Swin Transformer U-Net), built with PyTorch and MONAI

# BraTS 2020 — Brain Tumor Segmentation with SwinUNETR

3D volumetric brain tumor segmentation on the BraTS 2020 dataset using **SwinUNETR** (Swin Transformer U-Net), built with PyTorch and MONAI.

---

## Results

| Metric | Score |
|--------|--------|
| **Best Mean Dice** | **0.4244** |
| Best Epoch | 12 / 15 |
| Final Training Loss | 0.4263 |

---

## Class-wise Dice Scores

| Segmentation Class | Dice Score |
|--------------------|------------|
| NCR/NET – Necrotic Core (Class 1) | — |
| Edema (Class 2) | — |
| Enhancing Tumor (ET) (Class 3) | — |
| **Mean Dice (foreground)** | **0.4244** |
> Per-class scores are printed after running the Phase 5 evaluation.

---

##  Dataset

**BraTS 2020** — Multimodal Brain Tumor Segmentation Challenge 2020

- **Source:** [Kaggle — awsaf49/brats20-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- **Total cases:** 371 patients (368 valid after filtering)
- **Split:** 80% train / 20% validation (random seed 42)
- **MRI modalities:** T1, T1ce (contrast-enhanced), T2, FLAIR — stacked as 4-channel input
- **Segmentation classes:**
  - `0` — Background
  - `1` — NCR/NET (Necrotic and Non-Enhancing Tumor Core)
  - `2` — ED (Peritumoral Edema)
  - `3` — ET (Enhancing Tumor) ← remapped from original label `4`

---

##  Architecture — SwinUNETR

SwinUNETR is a hybrid Vision Transformer + CNN architecture designed for 3D medical image segmentation.

```
Input: (4, H, W, D)  — 4 MRI channels
  │
  ▼
Swin Transformer Encoder (hierarchical, shifted-window self-attention)
  │  └─ 4 stages of patch merging + window attention
  │
  ▼
CNN Decoder (skip connections from each encoder stage)
  │  └─ Transposed convolutions + residual blocks
  │
  ▼
Output: (4, H, W, D)  — 4-class voxel logits
```

**Model config used:**
```python
SwinUNETR(
    in_channels=4,
    out_channels=4,
    feature_size=24
)
```

---

##  Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Loss function | DiceCE Loss (`to_onehot_y=True, softmax=True`) |
| Patch size | 64 × 64 × 64 |
| Voxel spacing | 1.5 × 1.5 × 2.0 mm |
| Epochs | 15 |
| Val every | 2 epochs |
| Mixed precision | AMP (`torch.amp.autocast`) |
| Val inference | Sliding window (roi=96³, overlap=0.5) |
| GPU | NVIDIA Tesla T4 |

### Preprocessing Pipeline

```
LoadImaged → EnsureChannelFirstd → ConvertLabels (4→3)
→ Orientationd (RAS) → Spacingd (1.5×1.5×2.0)
→ ScaleIntensityd → CropForegroundd
→ SpatialPadd (64³) → RandSpatialCropd (64³)
→ EnsureTyped
```

---

##  Training Curve

| Epoch | Train Loss | Val Dice |
|-------|------------|----------|
| 1 | 1.4242 | — |
| 2 | 1.0152 | 0.1666 |
| 4 | 0.7280 | 0.3673 |
| 6 | 0.5991 | 0.3908 |
| 8 | 0.5275 | 0.3992 |
| 10 | 0.4815 | 0.3980 |
| **12** | **0.4469** | **0.4244 ← Best** |
| 14 | 0.4332 | 0.4196 |
| 15 | 0.4263 | — |

---

##  How to Run

### 1. Setup (Google Colab)

```python
!pip install -q monai nibabel einops scikit-learn kaggle

 3. Train

Run all cells in `BraTS_SwinUNETR.ipynb` sequentially. The notebook handles:
- Data loading and preprocessing
- Model setup
- Training loop with AMP
- Validation with sliding window inference
- Saving the best checkpoint as `best_model.pth`

 4. Evaluate & Visualize

Run Phase end cells to generate:
- Per-class Dice scores
- Training loss + validation Dice curves
- Side-by-side GT vs prediction slice plots
- Scrolling axial GIF



## Dependencies
```torch >= 2.0
monai >= 1.3
nibabel
einops
scikit-learn
numpy
matplotlib
imageio
```

## References

- **SwinUNETR:** Hatamizadeh et al., *"Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"*, MICCAI 2022. [arXiv:2201.01266](https://arxiv.org/abs/2201.01266)
- **MONAI Framework:** [monai.io](https://monai.io)
- **BraTS 2020 Challenge:** [med.upenn.edu/cbica/brats2020](https://www.med.upenn.edu/cbica/brats2020/)
- **Dataset (Kaggle):** [awsaf49/brats20-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

## Notes
- Label `4` (Enhancing Tumor) is remapped to `3` before training to ensure contiguous class indices
- `Orientationd` and `Spacingd` produce deprecation warnings in MONAI 1.5.2 — these are harmless
- Sliding window inference uses `roi_size=(96,96,96)` at validation even though training patches are `64³` — this improves val accuracy
- The model was trained for only 15 epochs as a proof of concept; longer training (50–100 epochs) would yield significantly higher Dice scores
