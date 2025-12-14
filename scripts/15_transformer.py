"""
15 - NOVEL: Transformer for Exoplanet Detection
Vision Transformer (ViT) style architecture adapted for 1D light curves

Transformers are state-of-the-art for sequence modeling.
This adapts the attention mechanism for transit detection.

Created: December 2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import math
import os
import warnings
warnings.filterwarnings('ignore')

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
output_path = os.path.join(project_dir, 'graphs', '15_transformer.png')

print("\n" + "="*60)
print("NOVEL: TRANSFORMER FOR EXOPLANET DETECTION")
print("="*60)

# Load data
print("\nLoading Kepler light curve data...")
train_df = pd.read_csv(os.path.join(data_dir, 'exoTrain.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'exoTest.csv'))

X_train = train_df.iloc[:, 1:].values
y_train = (train_df.iloc[:, 0].values == 2).astype(int)
X_test = test_df.iloc[:, 1:].values
y_test = (test_df.iloc[:, 0].values == 2).astype(int)

print(f"Training: {len(X_train)} | Test: {len(X_test)}")
print(f"Sequence length: {X_train.shape[1]}")

# Normalize
def normalize_lc(X):
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        X_norm[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-8)
    return X_norm

X_train = normalize_lc(X_train)
X_test = normalize_lc(X_test)

# Reduce sequence length for memory (patch the light curve)
patch_size = 32
seq_len = X_train.shape[1]
n_patches = seq_len // patch_size

# Reshape into patches: (batch, n_patches, patch_size)
X_train_patched = X_train[:, :n_patches * patch_size].reshape(-1, n_patches, patch_size)
X_test_patched = X_test[:, :n_patches * patch_size].reshape(-1, n_patches, patch_size)

print(f"Patched shape: {X_train_patched.shape}")
print(f"Number of patches: {n_patches}")

# Tensors
X_train_t = torch.FloatTensor(X_train_patched).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test_patched).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class weight: {pos_weight:.1f}")

#======================================
# NOVEL: Light Curve Transformer (LCT)
#======================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LightCurveTransformer(nn.Module):
    """
    NOVEL: Transformer adapted for light curve classification
    - Patches the light curve like ViT patches images
    - Multi-head self-attention finds transit patterns
    - CLS token for classification
    """
    def __init__(self, patch_size, n_patches, d_model=64, n_heads=4, n_layers=3, dropout=0.3):
        super().__init__()
        
        # Patch embedding (like ViT)
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, n_patches + 1)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.d_model = d_model
        
    def forward(self, x, return_attention=False):
        batch_size = x.size(0)
        
        # Embed patches
        x = self.patch_embed(x)  # (batch, n_patches, d_model)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches+1, d_model)
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        out = self.classifier(cls_output)
        
        return out

# Initialize model
model = LightCurveTransformer(
    patch_size=patch_size,
    n_patches=n_patches,
    d_model=64,
    n_heads=4,
    n_layers=3,
    dropout=0.3
).to(device)

print(f"\nTransformer parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Weighted sampler
weights = np.where(y_train == 1, pos_weight, 1.0)
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

# Training
print("\nTraining Light Curve Transformer...")
epochs = 50
train_losses = []
val_accs = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X).squeeze()
        loss = criterion(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t).squeeze()
        val_pred_class = (val_pred > 0.5).float()
        val_acc = (val_pred_class == y_test_t).float().mean().item()
        val_accs.append(val_acc)
    
    train_losses.append(epoch_loss / len(train_loader))
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f} - Val Acc: {val_acc*100:.1f}%")

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_t).squeeze().cpu().numpy()
    y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
except:
    auc = 0
    fpr, tpr = [0, 1], [0, 1]

print(f"\n{'='*60}")
print("LIGHT CURVE TRANSFORMER RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# Visualization
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#050510')

fig.suptitle('LIGHT CURVE TRANSFORMER', fontsize=28, fontweight='bold', 
             color='#ff8800', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.935, 'NOVEL: Vision Transformer Architecture Adapted for Time-Series Transit Detection', 
         ha='center', fontsize=13, color='#8888aa', fontfamily='monospace')

# Plot 1: Architecture diagram
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Architecture flowchart
elements = [
    (5, 9, 'LIGHT CURVE\n(3168 points)', '#00ffff'),
    (5, 7.5, 'PATCH\nEMBEDDING', '#ff8800'),
    (5, 6, 'POSITIONAL\nENCODING', '#ffff00'),
    (5, 4.5, 'TRANSFORMER\n(3 layers×4 heads)', '#ff00ff'),
    (5, 3, 'CLS TOKEN', '#00ff88'),
    (5, 1.5, 'CLASSIFIER', '#4488ff'),
]

for x, y, text, color in elements:
    ax1.add_patch(plt.Rectangle((x-1.2, y-0.5), 2.4, 1, 
                                  facecolor=color, alpha=0.2, edgecolor=color, linewidth=2))
    ax1.text(x, y, text, ha='center', va='center', color='white', fontsize=9, 
             fontfamily='monospace', fontweight='bold')

# Arrows
for i in range(len(elements)-1):
    ax1.annotate('', xy=(5, elements[i+1][1]+0.5), xytext=(5, elements[i][1]-0.5),
                 arrowprops=dict(arrowstyle='->', color='white', lw=2))

ax1.set_title('TRANSFORMER ARCHITECTURE', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace', y=1.02)

# Plot 2: Training progress
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

ax2.plot(train_losses, color='#ff8800', linewidth=2, label='Train Loss')
ax2_twin = ax2.twinx()
ax2_twin.plot(val_accs, color='#00ff88', linewidth=2, label='Val Accuracy')

ax2.set_xlabel('EPOCH', color='#ccccff', fontfamily='monospace')
ax2.set_ylabel('LOSS', color='#ff8800', fontfamily='monospace')
ax2_twin.set_ylabel('ACCURACY', color='#00ff88', fontfamily='monospace')
ax2.set_title('TRAINING PROGRESS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.tick_params(colors='#888899')
ax2_twin.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa')
for spine in ax2.spines.values(): spine.set_color('#3333aa')
for spine in ax2_twin.spines.values(): spine.set_color('#3333aa')

# Plot 3: ROC Curve
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

ax3.plot(fpr, tpr, color='#ff8800', linewidth=3, label=f'AUC = {auc:.3f}')
ax3.plot([0, 1], [0, 1], 'w--', alpha=0.3)
ax3.fill_between(fpr, tpr, alpha=0.2, color='#ff8800')

ax3.set_xlabel('FALSE POSITIVE RATE', color='#ccccff', fontfamily='monospace')
ax3.set_ylabel('TRUE POSITIVE RATE', color='#ccccff', fontfamily='monospace')
ax3.set_title('ROC CURVE', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.legend(loc='lower right', facecolor='#1a1a2e', labelcolor='#ccccff', fontsize=12)
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa')
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Plot 4: Summary
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

for i in range(40):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.3, 'TRANSFORMER RESULTS', ha='center', fontsize=16, fontweight='bold', 
         color='#ff8800', fontfamily='monospace')

stats = [
    ('ARCHITECTURE', 'Light Curve Transformer', '#ff8800'),
    ('ATTENTION HEADS', '4', '#00ffff'),
    ('TRANSFORMER LAYERS', '3', '#ffff00'),
    ('PATCH SIZE', f'{patch_size}', '#ff00ff'),
    ('ACCURACY', f'{acc*100:.1f}%', '#00ff88'),
    ('PRECISION', f'{prec*100:.1f}%', '#4488ff'),
    ('AUC', f'{auc:.4f}', '#ff4466'),
    ('PARAMETERS', f'{sum(p.numel() for p in model.parameters()):,}', '#88aaff'),
]

for i, (label, value, color) in enumerate(stats):
    y_pos = 7.8 - i * 0.9
    ax4.text(0.3, y_pos, f'{label}:', fontsize=10, color='#8888aa', fontfamily='monospace', va='center')
    ax4.text(9.7, y_pos, str(value), fontsize=10, color=color, fontweight='bold', fontfamily='monospace', ha='right', va='center')
    ax4.axhline(y=y_pos - 0.4, xmin=0.03, xmax=0.97, color='#333366', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(output_path, dpi=150, facecolor='#050510', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
print("\n" + "="*60)
print("NOVEL CONTRIBUTION:")
print("  1. ViT-style patch embedding for light curves")
print("  2. Multi-head self-attention for transit patterns")
print("  3. CLS token classification (BERT-style)")
print("  4. State-of-the-art architecture for sequences")
print("="*60)
