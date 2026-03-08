#!/bin/bash

# =============================================================================
# GENERIC TRAINING SCRIPT FOR HTCONDOR BATCH SUBMISSION
# =============================================================================
# This script runs different scenarios using Apptainer container with GPU support
# =============================================================================

# Exit immediately if any command fails
set -e

# Get the main file to run from command line argument
MAIN_FILE=${1:-main.py}

# Print useful debugging information
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Training: ${MAIN_FILE}"
echo "=========================================="

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP WITH APPTAINER
# -----------------------------------------------------------------------------
# Use CMSML container with GPU support (--nv enables NVIDIA GPU passthrough)
CONTAINER=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest
WORKDIR=/afs/cern.ch/user/a/amorandi/private/HH4b/FFN

# -----------------------------------------------------------------------------
# VERIFY GPU IS AVAILABLE ON THE NODE
# -----------------------------------------------------------------------------
echo "GPU Information on the node:"
nvidia-smi
echo ""

# -----------------------------------------------------------------------------
# RUN THE TRAINING INSIDE APPTAINER CONTAINER
# -----------------------------------------------------------------------------
echo "=========================================="
echo "Starting training at: $(date)"
echo "=========================================="

# Run Python inside the container
# -B /afs -B /eos : Bind mount AFS and EOS filesystems
# --nv : Enable NVIDIA GPU support (passes GPU drivers to container)
apptainer exec -B /afs -B /eos --nv ${CONTAINER} bash -c "
    cd ${WORKDIR}
    
    echo 'Inside container:'
    echo 'Python version:' \$(python --version)
    echo 'PyTorch version:' \$(python -c 'import torch; print(torch.__version__)')
    echo 'GPU devices available:' \$(python -c 'import torch; print(torch.cuda.device_count())')
    if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
        echo 'GPU details:' \$(python -c 'import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])')
    fi
    echo ''
    
    # Install required packages not in the container
    echo 'Installing required packages...'
    pip install --quiet --user uproot scikit-learn tqdm matplotlib seaborn scipy h5py
    echo 'Packages installed.'
    echo ''
    
    # Run the training
    python ${MAIN_FILE}
"

echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
