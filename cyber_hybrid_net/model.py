"""
CyberHybridNet: A Hybrid Transformer with Multi-Scale Attention for Cybersecurity Anomaly Detection
==============================================================================================================
Architecture:
  1. Multi-Scale CNN Feature Extractor (local pattern capture at 3 scales)
  2. Rotary Position Embeddings for temporal awareness
  3. Hybrid Attention Block:
     - Multi-Head Self-Attention (global flow dependencies)
     - Gated Cross-Attention (cross-feature interaction)
     - Feed-Forward with SwiGLU activation
  4. Mixture-of-Experts Classifier with uncertainty estimation
  
Datasets: CICIDS2017 (lacg030175) + UNSW-NB15 (Mouwiya)
"""

import os
import sys
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Try importing optional monitoring
try:
    import trackio
    HAS_TRACKIO = True
except ImportError:
    HAS_TRACKIO = False

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memoryory / 1e9:.1f} GB")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# ARCHITECTURE COMPONENTS
# ============================================================

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for temporal awareness in flow sequences."""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()[None, :, None, :]
        sin_emb = emb.sin()[None, :, None, :]
        return cos_emb, sin_emb


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiScaleCNNExtractor(nn.Module):
    """Multi-scale 1D CNN for local pattern extraction at different granularities."""
    def __init__(self, input_dim, hidden_dim, num_scales=3):
        super().__init__()
        self.scales = nn.ModuleList()
        # Compute per-scale channels so they sum exactly to hidden_dim
        base_ch = hidden_dim // num_scales
        channels = [base_ch] * num_scales
        channels[-1] = hidden_dim - base_ch * (num_scales - 1)  # last scale absorbs remainder
        self.total_channels = sum(channels)
        for i in range(num_scales):
            kernel_size = 2 * i + 1  # 1, 3, 5
            padding = i
            ch = channels[i]
            self.scales.append(nn.Sequential(
                nn.Conv1d(input_dim, ch, kernel_size, padding=padding),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size, padding=padding),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ))
        self.fusion = nn.Sequential(
            nn.Linear(self.total_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> need (batch, features, seq_len) for conv1d
        x_conv = x.transpose(1, 2)
        multi_scale_out = []
        for scale in self.scales:
            out = scale(x_conv)  # (batch, hidden//num_scales, seq_len)
            multi_scale_out.append(out)
        concatenated = torch.cat(multi_scale_out, dim=1)  # (batch, hidden, seq_len)
        concatenated = concatenated.transpose(1, 2)  # (batch, seq_len, hidden)
        return self.fusion(concatenated)


class SwiGLU(nn.Module):
    """SwiGLU activation function from PaLM/LLaMA."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, heads, N, head_dim)
        
        # Apply RoPE
        cos, sin = self.rope(x)
        cos = cos.expand(B, -1, self.num_heads, -1).transpose(1, 2)
        sin = sin.expand(B, -1, self.num_heads, -1).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class GatedCrossAttention(nn.Module):
    """Gated Cross-Attention for cross-feature interaction."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, query, context):
        B, N, C = query.shape
        _, M, _ = context.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        gate_val = self.gate(query)
        return self.out_proj(out * gate_val)


class HybridAttentionBlock(nn.Module):
    """
    Hybrid Attention Block combining:
    1. Multi-Head Self-Attention (global)
    2. Gated Cross-Attention (cross-feature)
    3. SwiGLU FFN
    """
    def __init__(self, dim, num_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = GatedCrossAttention(dim, num_heads, dropout)
        
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, dim * ffn_mult, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context=None):
        # Self-attention with residual
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        
        # Cross-attention with residual (if context provided)
        if context is not None:
            x = x + self.dropout(self.cross_attn(self.norm2(x), context))
        
        # FFN with residual
        x = x + self.dropout(self.ffn(self.norm3(x)))
        
        return x


class MixtureOfExpertsClassifier(nn.Module):
    """Mixture-of-Experts classifier with uncertainty estimation."""
    def __init__(self, dim, num_classes, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_experts),
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, num_classes),
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.einsum('be,bec->bc', gate_probs, expert_outputs)
        
        return output, gate_probs


class CyberHybridNet(nn.Module):
    """
    CyberHybridNet: Hybrid Transformer for Cybersecurity Anomaly Detection
    
    Architecture:
    - Input Feature Projection
    - Multi-Scale CNN Feature Extractor (3 scales)
    - N x Hybrid Attention Blocks (Self-Attention + Cross-Attention + SwiGLU)
    - Mixture-of-Experts Classifier
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        num_experts=4,
        ffn_mult=4,
        dropout=0.1,
        seq_len=1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multi-scale CNN extractor (creates context for cross-attention)
        self.cnn_extractor = MultiScaleCNNExtractor(hidden_dim, hidden_dim, num_scales=3)
        
        # Hybrid attention layers
        self.attention_blocks = nn.ModuleList([
            HybridAttentionBlock(hidden_dim, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Pooling attention
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # MoE Classifier
        self.classifier = MixtureOfExpertsClassifier(hidden_dim, num_classes, num_experts, dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        x: (batch_size, input_dim) for single-step or (batch_size, seq_len, input_dim) for sequence
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim) -> create sequence dim
            
        B, S, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, S, hidden_dim)
        
        # CNN multi-scale features (context for cross-attention)
        cnn_features = self.cnn_extractor(x)  # (B, S, hidden_dim)
        
        # Apply hybrid attention blocks
        for block in self.attention_blocks:
            x = block(x, context=cnn_features)
        
        x = self.final_norm(x)
        
        # Attention pooling
        pool_query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(pool_query, x, x)
        pooled = pooled.squeeze(1)  # (B, hidden_dim)
        
        # Classification
        logits, gate_probs = self.classifier(pooled)
        
        return logits, gate_probs
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

class CyberSecurityDataset(Dataset):
    def __init__(self, features, labels, seq_len=1):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_preprocess_cicids2017(max_samples=None):
    """Load CICIDS2017 from HuggingFace with proper preprocessing."""
    from datasets import load_dataset
    
    print("Loading CICIDS2017 dataset...")
    ds = load_dataset("lacg030175/CICIDS2017", "temporal_3way")
    
    train_df = ds['train'].to_pandas()
    val_df = ds['validation'].to_pandas()
    test_df = ds['test'].to_pandas()
    
    print(f"CICIDS2017 - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Label distribution (train):\n{train_df['label'].value_counts()}")
    
    # Stratified subsample if max_samples is set
    if max_samples and len(train_df) > max_samples:
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(train_df, train_size=max_samples, 
                                        random_state=42, stratify=train_df['label'])
        print(f"Subsampled train to {len(train_df)}")
    if max_samples:
        val_size = min(len(val_df), max_samples // 4)
        test_size = min(len(test_df), max_samples // 4)
        if len(val_df) > val_size:
            val_df, _ = train_test_split(val_df, train_size=val_size,
                                          random_state=42, stratify=val_df['label'])
        if len(test_df) > test_size:
            test_df, _ = train_test_split(test_df, train_size=test_size,
                                           random_state=42, stratify=test_df['label'])
    
    # Get feature columns (exclude labels)
    exclude_cols = ['Label', 'label']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Clean data: replace inf, drop NaN
    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['label'].values
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Clip extreme values
    X_train = np.clip(X_train, -10, 10)
    X_val = np.clip(X_val, -10, 10)
    X_test = np.clip(X_test, -10, 10)
    
    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols, num_classes, 'CICIDS2017'


def load_and_preprocess_unsw_nb15(max_samples=None):
    """Load UNSW-NB15 from HuggingFace with proper preprocessing."""
    from datasets import load_dataset
    
    print("Loading UNSW-NB15 dataset...")
    ds = load_dataset("Mouwiya/UNSW-NB15")
    
    df = ds['train'].to_pandas()
    print(f"UNSW-NB15 total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Subsample if needed
    if max_samples and len(df) > max_samples:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=max_samples, random_state=42, stratify=df['label'])
        print(f"Subsampled to {len(df)}")
    
    # Remove non-numeric/IP columns
    drop_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'service', 
                 'attack_cat', 'label', 'ct_ftp_cmd']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Encode categorical columns that remain
    for col in feature_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Clean data
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values
    
    # Stratified split: 70/15/15
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Clip
    X_train = np.clip(X_train, -10, 10)
    X_val = np.clip(X_val, -10, 10)
    X_test = np.clip(X_test, -10, 10)
    
    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols, num_classes, 'UNSW-NB15'


# ============================================================
# TRAINING ENGINE
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in cybersecurity datasets."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CosineWarmupScheduler:
    """Cosine LR scheduler with linear warmup."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr_mult = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
        
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = max(self.min_lr, self.base_lrs[i] * lr_mult)
        
        return self.optimizer.param_groups[0]['lr']


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, gate_probs = model(features)
        
        # Main loss
        loss = criterion(logits, labels)
        
        # Load balancing loss for MoE (encourage uniform expert usage)
        expert_usage = gate_probs.mean(0)
        lb_loss = (expert_usage * torch.log(expert_usage + 1e-8)).sum() * 0.01
        loss = loss + lb_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr = scheduler.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | LR: {lr:.2e}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        logits, _ = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        probs = F.softmax(logits, dim=-1)
        _, predicted = logits.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    try:
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1]) * 100
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
    except:
        auc = 0.0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'auc': auc,
    }
    
    return metrics, all_preds, all_labels


def train_model(dataset_name='CICIDS2017', config=None, max_samples=None):
    """Main training loop."""
    
    if config is None:
        config = {
            'hidden_dim': 128,
            'num_layers': 4,
            'num_heads': 8,
            'num_experts': 4,
            'ffn_mult': 4,
            'dropout': 0.15,
            'batch_size': 512,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'epochs': 30,
            'patience': 7,
            'focal_gamma': 2.0,
        }
    
    print(f"\n{'='*70}")
    print(f"Training CyberHybridNet on {dataset_name}")
    print(f"{'='*70}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Load data
    if dataset_name == 'CICIDS2017':
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols, num_classes, name = load_and_preprocess_cicids2017(max_samples)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols, num_classes, name = load_and_preprocess_unsw_nb15(max_samples)
    
    input_dim = X_train.shape[1]
    
    # Create datasets
    train_dataset = CyberSecurityDataset(X_train, y_train)
    val_dataset = CyberSecurityDataset(X_val, y_val)
    test_dataset = CyberSecurityDataset(X_test, y_test)
    
    # Compute class weights for balanced sampling
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum()
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, 
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'] * 2, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'] * 2, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    # Create model
    model = CyberHybridNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_experts=config['num_experts'],
        ffn_mult=config['ffn_mult'],
        dropout=config['dropout'],
    ).to(DEVICE)
    
    num_params = model.count_parameters()
    print(f"\nModel Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Input dim: {input_dim}, Classes: {num_classes}")
    
    # Loss function with focal loss
    alpha = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=config['focal_gamma'])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    
    # Initialize tracking
    if HAS_TRACKIO:
        try:
            trackio.init(project="cyberhybridnet", name=f"{dataset_name.lower()}-training")
            print("Trackio monitoring initialized")
        except Exception as e:
            print(f"Trackio init failed: {e}")
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    training_history = []
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, DEVICE, epoch
        )
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch}/{config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}% | "
              f"F1-Macro: {val_metrics['f1_macro']:.2f}% | F1-Wt: {val_metrics['f1_weighted']:.2f}% | "
              f"AUC: {val_metrics['auc']:.2f}%")
        
        # Log to trackio
        if HAS_TRACKIO:
            try:
                trackio.log({
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/f1_macro': val_metrics['f1_macro'],
                    'val/f1_weighted': val_metrics['f1_weighted'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'val/auc': val_metrics['auc'],
                    'lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                })
            except:
                pass
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })
        
        # Early stopping
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  ★ New best F1-Macro: {best_val_f1:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch} (patience={config['patience']})")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(DEVICE)
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print(f"FINAL TEST EVALUATION ({dataset_name})")
    print(f"{'='*70}")
    
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.2f}%")
    print(f"  F1-Macro:    {test_metrics['f1_macro']:.2f}%")
    print(f"  F1-Weighted: {test_metrics['f1_weighted']:.2f}%")
    print(f"  Precision:   {test_metrics['precision']:.2f}%")
    print(f"  Recall:      {test_metrics['recall']:.2f}%")
    print(f"  AUC-ROC:     {test_metrics['auc']:.2f}%")
    
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, zero_division=0))
    
    return model, test_metrics, config, scaler, feature_cols, num_classes, input_dim, training_history


def push_model_to_hub(model, config, metrics_cicids, metrics_unsw, input_dim, num_classes_cicids, 
                       num_classes_unsw, feature_cols_cicids, feature_cols_unsw):
    """Push trained model to Hugging Face Hub."""
    from huggingface_hub import HfApi, create_repo
    import tempfile
    
    repo_id = "ha5eeb001/CyberHybridNet-anomaly-detector"
    
    try:
        api = HfApi()
        try:
            api.create_repo(repo_id, exist_ok=True, private=False)
        except Exception as e:
            print(f"Repo creation note: {e}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model weights
            model_path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), model_path)
            
            # Save config
            model_config = {
                'architecture': 'CyberHybridNet',
                'description': 'Hybrid Transformer with Multi-Scale CNN + Gated Cross-Attention + MoE for Cybersecurity Anomaly Detection',
                'training_config': config,
                'input_dim_cicids': int(feature_cols_cicids) if isinstance(feature_cols_cicids, int) else len(feature_cols_cicids),
                'input_dim_unsw': int(feature_cols_unsw) if isinstance(feature_cols_unsw, int) else len(feature_cols_unsw),
                'num_classes_cicids': num_classes_cicids,
                'num_classes_unsw': num_classes_unsw,
                'metrics_cicids2017': {k: float(v) for k, v in metrics_cicids.items()},
                'metrics_unsw_nb15': {k: float(v) for k, v in metrics_unsw.items()},
                'components': [
                    'Multi-Scale 1D CNN Feature Extractor (3 scales)',
                    'Rotary Position Embeddings',
                    'Multi-Head Self-Attention', 
                    'Gated Cross-Attention',
                    'SwiGLU Feed-Forward Networks',
                    'Mixture-of-Experts Classifier (4 experts)',
                    'Focal Loss for class imbalance',
                    'Cosine LR with warmup',
                ],
                'datasets': [
                    'lacg030175/CICIDS2017 (temporal_3way split)',
                    'Mouwiya/UNSW-NB15',
                ]
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            # Create README
            readme = f"""---
tags:
- cybersecurity
- anomaly-detection
- intrusion-detection
- transformer
- hybrid-attention
- pytorch
license: apache-2.0
datasets:
- lacg030175/CICIDS2017
- Mouwiya/UNSW-NB15
metrics:
- accuracy
- f1
- precision
- recall
---

# 🛡️ CyberHybridNet: Hybrid Transformer for Cybersecurity Anomaly Detection

## Architecture

**CyberHybridNet** is a cutting-edge hybrid transformer architecture designed specifically for network intrusion / anomaly detection in cybersecurity. It combines multiple advanced components:

### Key Components:
1. **Multi-Scale 1D CNN Feature Extractor** - Captures local patterns at 3 different granularities (kernel sizes 1, 3, 5)
2. **Rotary Position Embeddings (RoPE)** - Temporal awareness for network flow sequences
3. **Multi-Head Self-Attention** - Global dependency modeling across flow features
4. **Gated Cross-Attention** - Cross-feature interaction between CNN and transformer pathways with learned gating
5. **SwiGLU Feed-Forward Networks** - Advanced activation function from PaLM/LLaMA
6. **Mixture-of-Experts (MoE) Classifier** - 4-expert ensemble with load balancing for robust classification
7. **Focal Loss** - Handles severe class imbalance common in cybersecurity datasets
8. **Attention Pooling** - Learnable query-based pooling instead of naive mean pooling

### Architecture Diagram:
```
Input Features
      │
  ┌───▼───┐
  │ Input  │
  │Project │
  └───┬───┘
      │
  ┌───▼───────────┐     ┌──────────────────┐
  │ Multi-Scale    │────▶│ CNN Context       │
  │ CNN Extractor  │     │ (3 scales: 1,3,5) │
  └───┬───────────┘     └──────┬───────────┘
      │                        │
      │    ┌───────────────────┘
      │    │
  ┌───▼────▼───────────┐
  │ Hybrid Attention    │ × N layers
  │ ┌─────────────────┐│
  │ │Self-Attn + RoPE ││
  │ ├─────────────────┤│
  │ │Gated Cross-Attn ││
  │ ├─────────────────┤│
  │ │SwiGLU FFN       ││
  │ └─────────────────┘│
  └────────┬───────────┘
           │
  ┌────────▼───────────┐
  │ Attention Pooling   │
  └────────┬───────────┘
           │
  ┌────────▼───────────┐
  │ MoE Classifier     │
  │ (4 experts + gate) │
  └────────┬───────────┘
           │
       Predictions
```

## Performance

### CICIDS2017 (Temporal Split)
| Metric | Score |
|--------|-------|
| Accuracy | {metrics_cicids.get('accuracy', 0):.2f}% |
| F1-Macro | {metrics_cicids.get('f1_macro', 0):.2f}% |
| F1-Weighted | {metrics_cicids.get('f1_weighted', 0):.2f}% |
| Precision | {metrics_cicids.get('precision', 0):.2f}% |
| Recall | {metrics_cicids.get('recall', 0):.2f}% |
| AUC-ROC | {metrics_cicids.get('auc', 0):.2f}% |

### UNSW-NB15
| Metric | Score |
|--------|-------|
| Accuracy | {metrics_unsw.get('accuracy', 0):.2f}% |
| F1-Macro | {metrics_unsw.get('f1_macro', 0):.2f}% |
| F1-Weighted | {metrics_unsw.get('f1_weighted', 0):.2f}% |
| Precision | {metrics_unsw.get('precision', 0):.2f}% |
| Recall | {metrics_unsw.get('recall', 0):.2f}% |
| AUC-ROC | {metrics_unsw.get('auc', 0):.2f}% |

## Training Details

- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: Cosine with linear warmup (2 epochs)
- **Loss**: Focal Loss (γ=2.0) with class-weighted sampling
- **Regularization**: Dropout (0.15), gradient clipping (max_norm=1.0), MoE load balancing
- **Early Stopping**: Patience=7 on validation F1-Macro

## Usage

```python
import torch
from model import CyberHybridNet

# Load model
model = CyberHybridNet(
    input_dim=78,  # CICIDS2017 features
    num_classes=3,  # BENIGN, ATTACK, UNKNOWN
    hidden_dim=128,
    num_layers=4,
    num_heads=8,
    num_experts=4,
)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Predict
with torch.no_grad():
    features = torch.randn(1, 78)  # Your preprocessed features
    logits, gate_probs = model(features)
    prediction = logits.argmax(dim=-1)
```

## Datasets
- [CICIDS2017](https://huggingface.co/datasets/lacg030175/CICIDS2017) - Canadian Institute for Cybersecurity IDS 2017
- [UNSW-NB15](https://huggingface.co/datasets/Mouwiya/UNSW-NB15) - Australian Centre for Cyber Security
"""
            readme_path = os.path.join(tmpdir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(readme)
            
            # Upload all files
            api.upload_file(path_or_fileobj=model_path, path_in_repo="model.pt", repo_id=repo_id)
            api.upload_file(path_or_fileobj=config_path, path_in_repo="config.json", repo_id=repo_id)
            api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=repo_id)
            
            # Upload the model architecture code
            script_path = os.path.abspath(__file__)
            api.upload_file(path_or_fileobj=script_path, path_in_repo="model.py", repo_id=repo_id)
            
            print(f"\n✅ Model pushed to: https://huggingface.co/{repo_id}")
            
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Detect if GPU is available and set config accordingly
    is_gpu = torch.cuda.is_available()
    max_samples = None if is_gpu else 500000  # Use full data on GPU, subsample on CPU
    
    config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 8,
        'num_experts': 4,
        'ffn_mult': 4,
        'dropout': 0.15,
        'batch_size': 1024 if is_gpu else 512,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'epochs': 30 if is_gpu else 20,
        'patience': 7 if is_gpu else 5,
        'focal_gamma': 2.0,
    }
    
    # ---- Train on CICIDS2017 ----
    print("\n" + "="*80)
    print("PHASE 1: Training on CICIDS2017")
    print("="*80)
    model_cicids, metrics_cicids, _, scaler_cicids, fcols_cicids, nclasses_cicids, input_dim_cicids, hist_cicids = \
        train_model('CICIDS2017', config, max_samples)
    
    # ---- Train on UNSW-NB15 ----
    print("\n" + "="*80) 
    print("PHASE 2: Training on UNSW-NB15")
    print("="*80)
    model_unsw, metrics_unsw, _, scaler_unsw, fcols_unsw, nclasses_unsw, input_dim_unsw, hist_unsw = \
        train_model('UNSW-NB15', config, max_samples)
    
    # ---- Push best model to Hub ----
    print("\n" + "="*80)
    print("PHASE 3: Pushing models to Hub")
    print("="*80)
    
    # Push the CICIDS model (typically more challenging dataset)
    push_model_to_hub(
        model_cicids, config, metrics_cicids, metrics_unsw,
        input_dim_cicids, nclasses_cicids, nclasses_unsw,
        fcols_cicids, fcols_unsw
    )
    
    # Also save UNSW model
    torch.save(model_unsw.state_dict(), '/tmp/model_unsw.pt')
    from huggingface_hub import HfApi
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj='/tmp/model_unsw.pt',
            path_in_repo="model_unsw_nb15.pt",
            repo_id="ha5eeb001/CyberHybridNet-anomaly-detector"
        )
        print("UNSW-NB15 model weights uploaded")
    except Exception as e:
        print(f"Upload error: {e}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nCICIDS2017 - Acc: {metrics_cicids['accuracy']:.2f}% | F1: {metrics_cicids['f1_macro']:.2f}% | AUC: {metrics_cicids['auc']:.2f}%")
    print(f"UNSW-NB15  - Acc: {metrics_unsw['accuracy']:.2f}% | F1: {metrics_unsw['f1_macro']:.2f}% | AUC: {metrics_unsw['auc']:.2f}%")
    print(f"\nModel: https://huggingface.co/ha5eeb001/CyberHybridNet-anomaly-detector")
