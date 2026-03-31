# BYOL-MRI: Bootstrap Your Own Latent for Accelerated MRI Reconstruction

This repository extends [CL-MRI](https://github.com/mevanekanayake/cl-mri) by replacing SimCLR-based 
contrastive pretraining with **BYOL (Bootstrap Your Own Latent)** for undersampled MRI reconstruction 
using the fastMRI knee dataset.

## Overview

Accelerated MRI reconstruction requires learning good representations from undersampled k-space data.
The original CL-MRI paper uses SimCLR contrastive learning for pretraining. This work investigates
whether BYOL — which requires no negative pairs — can learn better representations for this task.

### Key Contributions
- Implementation of BYOL pretraining for MRI reconstruction
- Adaptation of the CL-MRI pipeline to the fastMRI knee dataset
- Quantitative comparison of SimCLR vs BYOL pretraining strategies
- Data preparation scripts for the fastMRI knee multicoil dataset

## Method

### SimCLR (Original CL-MRI)
- Two views of same MRI slice via different undersampling masks
- Contrastive loss pulling positive pairs together, pushing negatives apart
- Requires large batch sizes for sufficient negative pairs

### BYOL (This Work)
- Two views of same MRI slice via different undersampling masks
- No negative pairs needed — uses EMA target network instead
- Online network: VarNet → ProjectionMLP → PredictionMLP
- Target network: VarNet → ProjectionMLP (updated via EMA only)
- Works better with smaller datasets and batch sizes
```
Online Network:  MRI view 1 → VarNet → ProjectionMLP → PredictionMLP → pred_1
Target Network:  MRI view 2 → VarNet → ProjectionMLP               → proj_2
Loss: MSE(pred_1, proj_2) + MSE(pred_2, proj_1)  [symmetric]
```

## Repository Structure
```
byol-mri/
├── pretrain_clmri.py           # SimCLR pretraining (original CL-MRI)
├── pretrain_byol.py            # BYOL pretraining (this work)
├── train_unet.py               # Baseline U-Net training (no pretraining)
├── train_unet_with_clmri.py    # Downstream U-Net with pretrained VarNet
├── requirements.txt
│
├── losses/
│   ├── supconloss.py           # SimCLR supervised contrastive loss
│   └── byolloss.py             # BYOL loss (cosine similarity MSE)
│
├── models/
│   ├── varnet.py               # VarNet + BYOL heads (ProjectionMLP, PredictionMLP, VarNetBYOL)
│   └── unet.py                 # U-Net downstream model
│
├── utils/
│   ├── data.py                 # Dataset and dataloader utilities
│   ├── transform.py            # k-space transforms and augmentations
│   ├── mask.py                 # k-space undersampling masks
│   ├── manager.py              # Training manager and logging
│   ├── fourier.py              # Fourier transform utilities
│   ├── math.py                 # Complex math utilities
│   ├── metrics.py              # NMSE, PSNR, SSIM metrics
│   └── paths.json              # Dataset and experiment paths (edit this!)
│
├── scripts/
│   └── prepare_data.py         # Convert .h5 files to .pt format
│
├── data/                       # Put your fastMRI .h5 files here
└── experiments/                # Training results saved here automatically
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/byol-mri.git
cd byol-mri
```

### 2. Install dependencies
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 3. Configure paths
Edit `utils/paths.json` with your machine name and data paths:
```json
{
    "YOUR_MACHINE_NAME": {
        "fastmriknee": "/path/to/your/data/",
        "experiments": "/path/to/your/experiments/"
    }
}
```

To find your machine name:
- **Windows**: run `hostname` in PowerShell
- **Linux/Mac**: run `hostname` in Terminal

### 4. Download the dataset
Download the fastMRI knee multicoil dataset from:
- **Official**: https://fastmri.med.nyu.edu/
- Cite as: Zbontar et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI", 2018

Place all `.h5` files in your data directory.

### 5. Prepare the data
```bash
python scripts/prepare_data.py \
    --data_dir /path/to/your/data/ \
    --train_ratio 0.8 \
    --seq_types CORPDFS_FBK CORPD_FBK
```

This converts `.h5` files to `.pt` format and builds `library.pt`.

## Training

### Step 1 — Pretraining

**SimCLR (original):**
```bash
python pretrain_clmri.py \
    --dset fastmriknee \
    --seq_types CORPDFS_FBK \
    --dp 0 --bs 2 --ne 100 \
    --num_cascades 12 --pools 4 --chans 18 \
    --sens_pools 4 --sens_chans 8
```

**BYOL (this work):**
```bash
python pretrain_byol.py \
    --dset fastmriknee \
    --seq_types CORPDFS_FBK \
    --dp 0 --bs 2 --ne 100 \
    --num_cascades 12 --pools 4 --chans 18 \
    --sens_pools 4 --sens_chans 8 \
    --tau 0.996 --proj_dim 256
```

### Step 2 — Downstream Training
```bash
python train_unet_with_clmri.py \
    --dset fastmriknee \
    --seq_types CORPDFS_FBK \
    --pret /path/to/experiments/Experiment_XXXX/Experiment_XXXX_model.pth \
    --dp 0 --bs 2 --ne 50 \
    --num_cascades 12 --pools 4 --chans 18 \
    --sens_pools 4 --sens_chans 8
```

### Step 3 — Baseline (no pretraining)
```bash
python train_unet.py \
    --dset fastmriknee \
    --seq_types CORPDFS_FBK \
    --dp 0 --bs 2 --ne 50
```

## Results

Preliminary results on 6 fastMRI knee volumes (2 epochs, debug settings):

| Method | NMSE ↓ | PSNR ↑ | SSIM ↑ |
|--------|--------|--------|--------|
| Baseline (no pretraining) | 0.0254 | 35.71 | 0.8757 |
| SimCLR (CL-MRI) | 0.0333 | 33.95 | 0.8467 |
| BYOL (this work) | 0.0341 | 33.86 | 0.8441 |

> **Note:** These results are from a small-scale debug run (6 volumes, 2 epochs).
> Full-scale results on 194 volumes with 100 epochs are in progress.
> With sufficient data and epochs, BYOL is expected to outperform SimCLR
> due to its stability with smaller batch sizes and lack of negative pairs.

## Citation

If you use this code, please cite the original CL-MRI paper:
```bibtex
@article{ekanayake2023clmri,
  title={CL-MRI: Self-Supervised Contrastive Learning to Improve the Accuracy
         of Undersampled MRI Reconstruction},
  author={Ekanayake, Mevan and others},
  year={2023}
}
```

And the fastMRI dataset:
```bibtex
@article{zbontar2018fastmri,
  title={fastMRI: An Open Dataset and Benchmarks for Accelerated MRI},
  author={Zbontar, Jure and others},
  year={2018},
  journal={arXiv:1811.08839}
}
```

## Acknowledgements

This work builds directly on the CL-MRI repository by
[mevanekanayake](https://github.com/mevanekanayake/cl-mri).
The BYOL implementation follows Grill et al., NeurIPS 2020.