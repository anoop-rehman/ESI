# ESI Model Setup and Evaluation Guide

## Current Setup Status ✓

We have:
- ✓ Python virtual environment (.venv)
- ✓ All dependencies installed
- ✓ PTB-XL dataset (2567 records from 100Hz version)
- ✓ PTB-XL benchmarking code

## Step 1: Download Pre-trained ESI Model Weights

According to the README, the model weights are available at:
https://drive.google.com/drive/folders/1hUpfr0TNe2WoQC3Vi6l7ofDJe5CMjUqu?usp=sharing

**Manual Download Instructions:**
1. Visit the Google Drive link above
2. Download the model checkpoint file (likely named something like `esi_convnext_base.pth` or similar)
3. Place it in `/root/sickkids/ESI/pretrained_models/`

**Alternative - Using gdown:**
```bash
cd /root/sickkids/ESI
source .venv/bin/activate
pip install gdown
mkdir -p pretrained_models
# Use gdown with the file ID from the Google Drive link
```

## Step 2: Data Location

PTB-XL data is located at:
```
/root/sickkids/ESI/ecg_ptbxl_benchmarking/data/ptbxl/
```

Key files:
- `ptbxl_database.csv` - Metadata with labels
- `scp_statements.csv` - SCP code definitions
- `records100/` - ECG signal files (100Hz)

## Step 3: Run Evaluation

We've created an evaluation script that:
1. Loads the pre-trained ESI model
2. Evaluates on PTB-XL test set
3. Reports AUC scores for arrhythmia diagnosis

```bash
cd /root/sickkids/ESI
source .venv/bin/activate
python evaluate_esi.py --model_path pretrained_models/esi_model.pth --data_path ecg_ptbxl_benchmarking/data/ptbxl/
```

## Expected Results

According to the paper (Table 2), ESI should achieve:
- **PTB-XL (Fine-tuned)**: 0.938 ± 0.011 AUC
- **PTB-XL (Linear Probing)**: 0.931 ± 0.008 AUC
- **PTB-XL (Zero-shot)**: 0.812 AUC

## Dataset Information

- **Total records downloaded**: 2,567 ECG recordings
- **Sampling rate**: 100 Hz
- **Duration**: 10 seconds per recording
- **Leads**: 12-lead ECG
- **Task**: Multi-label arrhythmia classification

## Notes

- The evaluation uses the PTB-XL benchmarking framework
- Test set is fold 10 (as specified in the paper)
- Training folds 1-8, validation fold 9, test fold 10

