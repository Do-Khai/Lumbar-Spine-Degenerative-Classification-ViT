# Lumbar Spine Degenerative Classification using 3D Vision Transformer (ViT)

This repository contains code for the classification of lumbar spine degenerative conditions using 3D Vision Transformers. The project is based on the [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) competition.

## 📌 Overview
The goal of this project is to develop a deep learning model that can simulate a radiologist's performance in diagnosing degenerative spine conditions from lumbar spine MR images. The model identifies pathologies across five vertebral levels (L1/L2, L2/L3, L3/L4, L4/L5, L5/S1) and classifies them into three severity levels: **Normal/Mild**, **Moderate**, and **Severe**.

### 📄 Research Paper
Detailed methodology and theoretical background can be found in the associated technical report:
**["Lumbar Spine Degeneration Classification in Radiographic Image"](https://drive.google.com/file/d/163KWjJpNe2Fkyhn1bIEzUBoOyAlJhkJV/view?usp=sharing)**

## 📊 Exploratory Data Analysis (EDA)
The dataset consists of approximately 2,000 MRI studies, each containing multiple sequences:
- **Axial T2**: Useful for detecting edema, inflammation, and cross-sectional pathologies.
- **Sagittal T1**: Provides high anatomical detail and contrast between soft tissues.
- **Sagittal T2/STIR**: Sensitive to lesions and edema, featuring fat-suppression techniques (Short Tau Inversion Recovery).

### Conditions Classified:
- Spinal Canal Stenosis
- Neural Foraminal Narrowing (Left/Right)
- Subarticular Stenosis (Left/Right)

## 🛠️ Data Preprocessing & Pipeline
Due to the 3D nature of MRI data, the project implements a sophisticated preprocessing pipeline:
1. **DICOM to Point Cloud**: DICOM slices are converted into 3D point clouds using `open3d`, preserving spatial orientation and slice thickness.
2. **Voxel Grid Generation**: Point clouds are transformed into a $(128, 128, 128)$ voxel grid.
3. **Caching**: Preprocessed voxel grids are cached using `pgzip` to significantly speed up training.
4. **Majority Voting**: A custom dataset class (`studyleveldataset`) implements majority voting across different labels for a given study to handle multi-label classification tasks efficiently.

## 🏗️ Model Architecture: 3D-ViT
The core of the system is a **Vision Transformer (ViT)** adapted for 3D volumetric data.

### Key Components:
- **PatchEmbed3D**: Splits the 3D volume into patches and projects them into an embedding space using 3D convolutions.
- **Attention3D**: Multi-head self-attention mechanism specifically tuned for volumetric spatial relationships.
- **TransformerBlock3D**: Standard transformer encoder blocks with LayerNorm, Attention, and MLP layers.
- **VisionTransformer3D**: The complete architecture including class tokens, positional embeddings, and a classification head.

## 🚀 Training & Evaluation
- **Framework**: PyTorch
- **Optimizer**: Adam (Learning Rate: $1e-4$)
- **Loss Function**: Cross-Entropy Loss
- **Cross-Validation**: 5-Fold Cross-Validation strategy to ensure model robustness.
- **Metrics**: Weighted Log Loss (Weighted as 1 for Normal, 2 for Moderate, 4 for Severe).

## 📈 Visualizations
The project includes tools for:
- 3D Point Cloud visualization of spinal structures.
- Voxel grid slice inspection.
- Training/Validation loss and accuracy plots.

## 📁 Repository Structure
- `spine-vit.ipynb`: Main notebook containing data processing, model architecture, and training logic.
- `visualizing-eda.ipynb`: Comprehensive Exploratory Data Analysis and data visualization.

---
*Created as part of a Final Thesis project on Lumbar Spine Classification.*
