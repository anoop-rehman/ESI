"""
ESI Model Evaluation Script for PTB-XL
This script evaluates a pre-trained ESI model on the PTB-XL dataset for arrhythmia diagnosis.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import wfdb

# Add the model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from model.convnextv2 import convnextv2_base, convnextv2_tiny
from model.xresnet1d import xresnet1d101
from model.esi import ESI
from transformers import AutoTokenizer, AutoModel


class PTBXLDataset(Dataset):
    """PTB-XL Dataset for evaluation"""
    
    def __init__(self, data_path, sampling_rate=100, split='test'):
        """
        Args:
            data_path: Path to PTB-XL data directory
            sampling_rate: 100 or 500 Hz
            split: 'train', 'val', or 'test'
        """
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        
        # Load database
        db = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'))
        
        # Filter by split (fold 10 is test, fold 9 is val, folds 1-8 are train)
        if split == 'test':
            self.data = db[db['strat_fold'] == 10].reset_index(drop=True)
        elif split == 'val':
            self.data = db[db['strat_fold'] == 9].reset_index(drop=True)
        else:  # train
            self.data = db[db['strat_fold'] <= 8].reset_index(drop=True)
        
        # Load SCP statements for label mapping
        self.scp_statements = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
        
        # Get diagnostic superclass labels
        self.labels = []
        for _, row in self.data.iterrows():
            scp_codes = eval(row['scp_codes'])
            label = self._get_diagnostic_superclass(scp_codes)
            self.labels.append(label)
        
        self.labels = np.array(self.labels)
        self.n_classes = self.labels.shape[1]
        
        print(f"Loaded {len(self.data)} samples for {split} split with {self.n_classes} classes")
    
    def _get_diagnostic_superclass(self, scp_codes):
        """Convert SCP codes to diagnostic superclass labels"""
        # Diagnostic superclasses: NORM, MI, STTC, CD, HYP
        superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        label = np.zeros(len(superclasses))
        
        for code, confidence in scp_codes.items():
            if code in self.scp_statements.index:
                diagnostic_class = self.scp_statements.loc[code, 'diagnostic_class']
                if diagnostic_class in superclasses:
                    idx = superclasses.index(diagnostic_class)
                    label[idx] = 1
        
        return label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load ECG signal
        if self.sampling_rate == 100:
            ecg_path = os.path.join(self.data_path, row['filename_lr'])
        else:
            ecg_path = os.path.join(self.data_path, row['filename_hr'])
        
        # Read signal using wfdb
        signal, _ = wfdb.rdsamp(ecg_path)
        
        # Convert to tensor (shape: [leads, samples])
        signal = torch.tensor(signal.T, dtype=torch.float32)
        
        # Normalize
        signal = (signal - signal.mean(dim=1, keepdim=True)) / (signal.std(dim=1, keepdim=True) + 1e-8)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return signal, label


def load_esi_model(model_path, device, signal_encoder='convnextv2_base'):
    """Load pre-trained ESI model"""
    
    # Initialize signal encoder
    if signal_encoder == "convnextv2_base":
        signal_enc = convnextv2_base(in_chans=12, num_classes=5, return_embedding=True)
        signal_dim = 1024
    elif signal_encoder == "convnextv2_tiny":
        signal_enc = convnextv2_tiny(in_chans=12, num_classes=5, return_embedding=True)
        signal_dim = 768
    elif signal_encoder == "xresnet1d101":
        signal_enc = xresnet1d101(num_classes=5, input_channels=12, kernel_size=5, ps_head=0.5, lin_ftrs_head=[128])
        signal_dim = 512
    else:
        raise ValueError(f"Unknown signal encoder: {signal_encoder}")
    
    # Initialize text encoder
    text_encoder = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base')
    tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
    total_tokens = len(tokenizer.vocab)
    
    # Initialize ESI model
    esi = ESI(
        dim=768,
        image_dim=signal_dim,
        num_tokens=total_tokens,
        pretrained_text_encoder=text_encoder,
        unimodal_depth=6,
        multimodal_depth=6,
        dim_head=64,
        heads=8,
        ff_mult=4,
        img_encoder=signal_enc,
        caption_loss_weight=1.,
        contrastive_loss_weight=1.,
    )
    
    # Load checkpoint if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            esi.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            esi.load_state_dict(checkpoint['state_dict'])
        else:
            esi.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")
    else:
        print("WARNING: No model checkpoint provided or file not found!")
        print("Evaluating with randomly initialized weights (for testing setup)")
    
    esi = esi.to(device)
    esi.eval()
    
    return esi


class LinearClassifier(nn.Module):
    """Simple linear classifier on top of frozen ESI encoder"""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and ground truth"""
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in tqdm(dataloader, desc="Evaluating"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Get embeddings from ESI encoder
            embeddings = model.img_encoder(signals)
            
            # For now, we'll just use a simple linear layer for classification
            # In practice, you would use a trained classifier head
            # Here we just compute embeddings
            
            all_preds.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    
    # Compute AUC for each class
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:  # Skip if only one class present
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
    
    macro_auc = np.mean(aucs)
    
    # Compute average precision
    aps = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            aps.append(ap)
    
    macro_ap = np.mean(aps)
    
    return {
        'macro_auc': macro_auc,
        'macro_ap': macro_ap,
        'per_class_auc': aucs
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ESI model on PTB-XL')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--data_path', type=str, 
                        default='ecg_ptbxl_benchmarking/data/ptbxl/',
                        help='Path to PTB-XL data directory')
    parser.add_argument('--signal_encoder', type=str, default='convnextv2_base',
                        choices=['convnextv2_base', 'convnextv2_tiny', 'xresnet1d101'],
                        help='Signal encoder architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--sampling_rate', type=int, default=100,
                        choices=[100, 500],
                        help='Sampling rate (100 or 500 Hz)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading PTB-XL {args.split} dataset...")
    dataset = PTBXLDataset(args.data_path, sampling_rate=args.sampling_rate, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    print(f"\nLoading ESI model...")
    model = load_esi_model(args.model_path, device, signal_encoder=args.signal_encoder)
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    embeddings, labels = evaluate_model(model, dataloader, device)
    
    print(f"\nExtracted embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Note: For full evaluation, you would need to:
    # 1. Train a linear classifier on the embeddings (for linear probing)
    # 2. Or fine-tune the whole model (for fine-tuning evaluation)
    # 3. Or use contrastive similarity for zero-shot classification
    
    print("\n" + "="*60)
    print("EVALUATION SETUP COMPLETE!")
    print("="*60)
    print("\nTo reproduce paper results, you need to:")
    print("1. Download the pre-trained ESI model from Google Drive")
    print("2. For linear probing: Train a linear classifier on train embeddings")
    print("3. For fine-tuning: Fine-tune the model on PTB-XL training data")
    print("4. For zero-shot: Use text-image similarity with diagnostic labels")
    print("\nSee the paper and PTB-XL benchmarking code for full details.")
    print("="*60)


if __name__ == '__main__':
    main()

