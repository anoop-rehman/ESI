# Getting Started with ESI Model Evaluation

## ✅ Setup Complete!

Your environment is fully configured and ready to evaluate the ESI model on PTB-XL.

### What's Been Set Up

1. **Python Environment** (.venv)
   - Python 3.12.3
   - PyTorch 2.8.0 with CUDA 12.8
   - GPU: NVIDIA RTX 4000 Ada Generation

2. **PTB-XL Dataset**
   - Location: `ecg_ptbxl_benchmarking/data/ptbxl/`
   - Total records: 21,799
   - Train: 17,418 samples (folds 1-8)
   - Validation: 2,183 samples (fold 9)
   - Test: 2,198 samples (fold 10)
   - Format: 12-lead ECG, 10 seconds, 100 Hz

3. **Model Code**
   - ESI framework
   - ConvNeXtV2 and XResNet1D encoders
   - BioLinkBERT text encoder
   - Evaluation scripts

---

## 📥 Download Pre-trained Model Weights

**Important:** You need to download the pre-trained ESI model weights to reproduce paper results.

### Google Drive Link
https://drive.google.com/drive/folders/1hUpfr0TNe2WoQC3Vi6l7ofDJe5CMjUqu?usp=sharing

### Steps:
```bash
# 1. Create directory for model weights
mkdir -p /root/sickkids/ESI/pretrained_models

# 2. Download the model file from Google Drive
#    (Use browser or gdown command)

# 3. Place the .pth file in the pretrained_models directory
```

### Using gdown (command line):
```bash
cd /root/sickkids/ESI
source .venv/bin/activate
pip install gdown

# Get the file ID from the Google Drive folder and download
# Example: gdown <file_id> -O pretrained_models/esi_model.pth
```

---

## 🚀 Quick Start

### 1. Verify Setup
```bash
cd /root/sickkids/ESI
source .venv/bin/activate
python quick_test.py
```

### 2. Run Evaluation (without pre-trained weights)
This will extract embeddings using a randomly initialized model (for testing the pipeline):
```bash
python evaluate_esi.py \
    --data_path ecg_ptbxl_benchmarking/data/ptbxl/ \
    --split test \
    --batch_size 32
```

### 3. Run Evaluation (with pre-trained weights)
Once you have downloaded the model weights:
```bash
python evaluate_esi.py \
    --model_path pretrained_models/esi_model.pth \
    --data_path ecg_ptbxl_benchmarking/data/ptbxl/ \
    --signal_encoder convnextv2_base \
    --split test \
    --batch_size 32
```

---

## 📊 Expected Results (from paper)

### Table 2: Arrhythmia Diagnosis on PTB-XL

| Method | AUC Score |
|--------|-----------|
| **ESI (Fine-tuned)** | 0.938 ± 0.011 |
| **ESI (Linear Probing)** | 0.931 ± 0.008 |
| **ESI (Zero-shot)** | 0.812 |
| ESI-tiny (Fine-tuned) | 0.935 ± 0.011 |
| ESI-tiny (Linear Probing) | 0.927 ± 0.009 |

### Comparison with Baselines
- SimCLR (Fine-tuned): 0.916 ± 0.015
- BYOL (Fine-tuned): 0.925 ± 0.014
- Supervised ConvNeXt-Base: 0.914 ± 0.014

---

## 📖 Evaluation Modes

The paper evaluates ESI in three settings:

### 1. Zero-Shot Learning
- No training on PTB-XL
- Uses text-image similarity for classification
- Requires text descriptions of diagnostic classes

### 2. Linear Probing
- Freeze the ESI encoder
- Train only a linear classifier head
- Uses pre-extracted embeddings

### 3. Fine-Tuning
- Fine-tune the entire model on PTB-XL
- Updates both encoder and classifier
- Achieves best performance

---

## 🔧 Full Evaluation Pipeline

To fully reproduce paper results, you would need to:

### For Linear Probing:
```bash
# 1. Extract embeddings from train set
python evaluate_esi.py --model_path pretrained_models/esi_model.pth --split train

# 2. Train a linear classifier on embeddings
#    (using PTB-XL benchmarking code or scikit-learn)

# 3. Evaluate on test set
python evaluate_esi.py --model_path pretrained_models/esi_model.pth --split test
```

### For Fine-Tuning:
```bash
# Use the PTB-XL benchmarking framework
cd ecg_ptbxl_benchmarking/code

# Adapt the code to load ESI encoder instead of their models
# Train end-to-end on PTB-XL training data
```

---

## 📁 Project Structure

```
/root/sickkids/ESI/
├── .venv/                          # Python virtual environment
├── model/                          # Model implementations
│   ├── esi.py                     # ESI framework
│   ├── convnextv2.py              # Signal encoder
│   └── xresnet1d.py               # Alternative encoder
├── rag/                            # RAG pipeline (CQA)
│   └── mimic_waveform_dict.json   # Generated waveform descriptions
├── pretrained_models/              # Pre-trained weights (download here)
├── ecg_ptbxl_benchmarking/        # PTB-XL evaluation code
│   └── data/ptbxl/                # PTB-XL dataset
├── evaluate_esi.py                 # Main evaluation script
├── quick_test.py                   # Setup verification
├── requirements.txt                # Python dependencies
├── GETTING_STARTED.md             # This file
└── SETUP_AND_EVALUATION.md        # Detailed setup guide
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python evaluate_esi.py --batch_size 16  # or 8
```

### Missing Dependencies
```bash
cd /root/sickkids/ESI
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Loading Issues
```bash
# Verify data path
ls ecg_ptbxl_benchmarking/data/ptbxl/
ls ecg_ptbxl_benchmarking/data/ptbxl/records100/00000/
```

---

## 📚 References

**Paper:**
- Yu, Han, Peikun Guo, and Akane Sano. "ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text." Transactions on Machine Learning Research (TMLR), 2024.
- OpenReview: https://openreview.net/forum?id=giEbq8Khcf

**Code:**
- GitHub: https://github.com/comp-well-org/ESI
- Model Weights: https://drive.google.com/drive/folders/1hUpfr0TNe2WoQC3Vi6l7ofDJe5CMjUqu

**Dataset:**
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
- Benchmarking: https://github.com/helme/ecg_ptbxl_benchmarking

---

## 🎯 Next Steps

1. **Download model weights** from Google Drive
2. **Run evaluation** with pre-trained weights
3. **Implement linear probing** classifier
4. **Fine-tune** the model (optional)
5. **Compare results** with paper benchmarks

Good luck with your ESI experiments! 🚀

