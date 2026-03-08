# Normalizing Flow for HH4b Analysis

This subdirectory contains implementations of machine learning models for HH→4b (double Higgs production decaying to four b-quarks) physics analysis. The project includes both **Normalizing Flow** models and **Feed-Forward Neural Network** (FFN) benchmarks for anomaly detection and event classification.

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Models](#models)
- [Data](#data)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Training on HTCondor](#training-on-htcondor)
- [Output](#output)

---

## Overview

The goal of this project is to develop and benchmark machine learning techniques for distinguishing signal-like events from background events in the HH→4b analysis. The project explores two main approaches:

1. **Normalizing Flows (NF)**: A generative model that learns the probability density of the data, suitable for anomaly detection tasks.
2. **Feed-Forward Neural Networks (FFN)**: A discriminative classifier serving as a benchmark for comparison.

Both models are trained on 2022 CMS data from control regions (2b and 4b) and utilize high-level physics features reconstructed from jets.

---

## Directory Structure

```
NormalizingFlow/
├── README.md                    # This file
├── lib/                         # Shared utilities
│   ├── __init__.py
│   ├── data_loader.py          # Data loading, preprocessing, scaling
│   └── tester_function.py      # Model evaluation and testing functions
├── NF/                          # Normalizing Flow implementation
│   ├── __init__.py
│   ├── main.py                 # Main training script for standard NF
│   ├── main_phi.py             # Main training script with phi features
│   ├── model.py                # NF model definition and training logic
│   ├── model_phi.py            # NF model variant with phi features
│   ├── test_model.py           # NF model testing and evaluation
│   ├── run_training.sh         # Batch job execution script
│   ├── train_job.sub           # HTCondor submission file
│   └── logs/                   # Training logs and error files
└── FFN_benchmark/               # Feed-Forward Network benchmark
    ├── __init__.py
    ├── main.py                 # Main FFN training script
    ├── main_cathegorical_embedding.py  # FFN with categorical embeddings
    ├── model.py                # FFN model definition
    ├── test_model.py           # FFN model testing
    ├── run_training_FFN.sh     # Batch job execution script for FFN
    ├── train_job.sub           # HTCondor submission file for FFN
    └── logs/                   # Training logs and error files
```

---

## Models

### Normalizing Flow (NF)

The Normalizing Flow model is implemented using the `zuko` library and consists of:
- **Rational-Quadratic Neural Spline Flows** with multiple transformation layers
- **Conditional flow** with context dimension for different event categories
- **Learning rate scheduler** with linear warmup and cosine decay
- **Early stopping** mechanism to prevent overfitting

**Key Features:**
- Input dimension: Based on selected physics features (~34 features)
- Context dimension: 1 (for conditioning on event type e.g., 2b or  4b)
- Number of bins: 20 (for spline transformations)
- Number of transforms: 4
- Hidden layers: [256, 256]

### Feed-Forward Neural Network (FFN)

The FFN benchmark includes:
- **Standard FFN** for continuous features
- **Embedding-based FFN** for handling categorical features (e.g., era or njets)
- Architecture: 512 → 256 → 128 → 64 → 2 (output classes)
- **Regularization**: Batch normalization, dropout (20%), GELU activation
- **Optional embedding noise** for categorical features during training

---

## Data

### Data Sources

The models are trained on 2022 CMS data from two control regions:
- **2b Control Region** (CR2b): Events with 2 b-tagged jets
- **4b Control Region** (CR4b): Events with 4 b-tagged jets

Data files are located on EOS:
```
/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/
├── 2b_control_region_only/
│   ├── data_2022EE.root
│   └── data_2022preEE.root
└── 4b_control_region_only/
    ├── data_2022EE.root
    └── data_2022preEE.root
```

### Physics Features

The models use high-level reconstructed features including:
- **Event-level**: HT (scalar sum of jet pT)
- **Higgs candidates**: Reconstructed mass, pT, eta, phi, uncertainties
- **Di-Higgs system**: Invariant mass, pT, eta, ΔR, Δφ, Δη
- **Individual jets**: pT, eta, phi, mass for each of the 4 jets
- **Angular separations**: ΔR between jets and Higgs candidates

### Data Preprocessing

The `data_loader.py` module provides:
- **Loading**: ROOT file reading using `uproot`
- **Bootstrapping**: Resampling for balancing datasets
- **Scaling**: Min-max scaling (default range: [-5, 5])
- **Train/validation/test splits**: Configurable split ratios
- **Feature reshaping**: Log transformation for specific features (pT, masses, HT)

---

## Installation & Setup

### Requirements

This project runs on **CERN's LXPLUS** infrastructure with:
- HTCondor batch system
- Apptainer (Singularity) containers
- Access to `/afs` and `/eos` filesystems

### Container

The code uses the pre-built CMSML container with all necessary dependencies:
```bash
/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest
```

This container includes:
- Python 3.x
- PyTorch with CUDA support
- Scientific libraries (NumPy, SciPy, scikit-learn, matplotlib)
- `uproot` (for ROOT file I/O)


`zuko` library for normalizing flows is installed via pip in the afs environment.

### Training Normalizing Flow

#### Interactive Training
```bash
cd NormalizingFlow/NF
python main.py
```

#### Batch Submission
```bash
cd NormalizingFlow/NF
condor_submit train_job.sub
```

### Training FFN Benchmark

#### Interactive Training
```bash
cd NormalizingFlow/FFN_benchmark
python main.py
```

#### Batch Submission
```bash
cd NormalizingFlow/FFN_benchmark
condor_submit train_job.sub
```

---

## Output

### Model Checkpoints

Trained models are saved to EOS:
```
/eos/user/a/amorandi/HH4b/
├── NF_HQ/weights/flow_model.pth        # Normalizing Flow weights
└── FFN_HQ/weights/ffn_model.pth        # FFN weights
```

### Training Plots

Training progress is automatically plotted and saved:
- `training_loss.png` - Loss vs. epoch curves
- `validation_metrics.png` - Validation performance

### Evaluation Results

The testing modules (`test_model.py`) generate:
- **ROC curves** and AUC scores
- **Confusion matrices**
- **Probability distributions** for signal vs. background
- **Feature importance** analysis
- **KS test statistics** for distribution comparisons

