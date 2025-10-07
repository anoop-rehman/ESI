"""
Quick test script to verify the setup is working
"""

import os
import sys
import torch
import pandas as pd
import wfdb

print("="*60)
print("ESI Setup Verification")
print("="*60)

# 1. Check Python environment
print("\n1. Python Environment:")
print(f"   Python version: {sys.version.split()[0]}")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")

# 2. Check data
print("\n2. PTB-XL Data:")
data_path = "ecg_ptbxl_benchmarking/data/ptbxl/"
if os.path.exists(data_path):
    print(f"   ✓ Data directory found: {data_path}")
    
    # Check database file
    db_file = os.path.join(data_path, "ptbxl_database.csv")
    if os.path.exists(db_file):
        df = pd.read_csv(db_file)
        print(f"   ✓ Database CSV loaded: {len(df)} total records")
        
        # Check splits
        test_df = df[df['strat_fold'] == 10]
        val_df = df[df['strat_fold'] == 9]
        train_df = df[df['strat_fold'] <= 8]
        
        print(f"   ✓ Train samples: {len(train_df)}")
        print(f"   ✓ Val samples: {len(val_df)}")
        print(f"   ✓ Test samples: {len(test_df)}")
    
    # Try loading a sample ECG
    try:
        sample_path = os.path.join(data_path, "records100/00000/00001_lr")
        if os.path.exists(sample_path + ".dat"):
            signal, fields = wfdb.rdsamp(sample_path)
            print(f"   ✓ Sample ECG loaded: shape {signal.shape}")
            print(f"     Sampling rate: {fields['fs']} Hz")
            print(f"     Duration: {len(signal) / fields['fs']} seconds")
            print(f"     Leads: {len(signal[0])}")
        else:
            print(f"   ⚠ Sample ECG file not found")
    except Exception as e:
        print(f"   ⚠ Error loading sample ECG: {e}")
else:
    print(f"   ✗ Data directory not found: {data_path}")

# 3. Check model code
print("\n3. Model Code:")
model_files = [
    "model/esi.py",
    "model/convnextv2.py",
    "model/xresnet1d.py"
]
for f in model_files:
    if os.path.exists(f):
        print(f"   ✓ {f}")
    else:
        print(f"   ✗ {f} (missing)")

# 4. Check for model weights
print("\n4. Pre-trained Model Weights:")
model_dirs = ["pretrained_models", "saved_models", "checkpoints"]
found_weights = False
for d in model_dirs:
    if os.path.exists(d):
        files = os.listdir(d)
        if files:
            print(f"   ✓ Found directory: {d}/")
            for f in files:
                if f.endswith(('.pth', '.pt', '.ckpt')):
                    print(f"     - {f}")
                    found_weights = True

if not found_weights:
    print("   ⚠ No pre-trained model weights found")
    print("   → Download from: https://drive.google.com/drive/folders/1hUpfr0TNe2WoQC3Vi6l7ofDJe5CMjUqu")

# 5. Test model initialization
print("\n5. Model Initialization Test:")
try:
    sys.path.append(os.path.dirname(__file__))
    from model.convnextv2 import convnextv2_base
    from model.esi import ESI
    from transformers import AutoModel, AutoTokenizer
    
    print("   ✓ Imports successful")
    
    # Try creating a small model
    signal_encoder = convnextv2_base(in_chans=12, num_classes=5, return_embedding=True)
    print("   ✓ Signal encoder created")
    
    text_encoder = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base')
    tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
    print("   ✓ Text encoder loaded")
    
    print("   ✓ Model initialization successful!")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("Setup verification complete!")
print("="*60)
print("\nNext steps:")
print("1. Download pre-trained ESI model weights")
print("2. Run: python evaluate_esi.py --model_path <path_to_weights>")
print("="*60)

