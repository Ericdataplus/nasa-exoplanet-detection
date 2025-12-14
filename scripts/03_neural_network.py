"""
03 - Neural Network Exoplanet Detection
Deep Learning with PyTorch on GPU (RTX 3060)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'data', 'kepler_exoplanets.csv')
output_path = os.path.join(project_dir, 'graphs', '03_neural_network.png')

print("\nLoading NASA Kepler data...")
df = pd.read_csv(data_path)

# Binary classification
df_binary = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
df_binary['is_planet'] = (df_binary['koi_disposition'] == 'CONFIRMED').astype(int)

# Features
feature_cols = [
    'koi_score', 'koi_period', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_depth', 'koi_duration', 'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]

X = df_binary[feature_cols].copy()
y = df_binary['is_planet'].values

# Fill missing values
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define Neural Network
class ExoplanetNet(nn.Module):
    def __init__(self, input_dim):
        super(ExoplanetNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize model
model = ExoplanetNet(len(feature_cols)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"\nTraining Neural Network on GPU...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
epochs = 100
train_losses = []
val_accs = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t).squeeze()
        val_pred_class = (val_pred > 0.5).float()
        val_acc = (val_pred_class == y_test_t).float().mean().item()
        val_accs.append(val_acc)
    
    train_losses.append(epoch_loss / len(train_loader))
    scheduler.step(epoch_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f} - Val Acc: {val_acc*100:.1f}%")

# Final evaluation
model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_t).squeeze().cpu().numpy()
    y_pred = (y_pred_proba > 0.5).astype(int)

final_acc = accuracy_score(y_test, y_pred)
final_f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nFinal Results:")
print(f"  Accuracy: {final_acc*100:.2f}%")
print(f"  F1 Score: {final_f1:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Create SPACE-THEMED visualization
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0a0a1a')  # Deep space black

# Add title
fig.suptitle('EXOPLANET DETECTION: NEURAL NETWORK', fontsize=26, fontweight='bold', 
             color='#e0e0ff', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.93, 'Deep Learning on NASA Kepler Mission Data | PyTorch + RTX 3060', 
         ha='center', fontsize=12, color='#8888aa', fontfamily='monospace')

# Plot 1: Training progress
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')
ax1.plot(train_losses, color='#00ffff', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch', color='#ccccff', fontsize=11)
ax1.set_ylabel('Loss', color='#ccccff', fontsize=11)
ax1.set_title('NEURAL NETWORK TRAINING', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.tick_params(colors='#888899')
ax1.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8)
ax1.grid(True, alpha=0.2, color='#4444aa')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Plot 2: ROC Curve
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')
ax2.plot(fpr, tpr, color='#ff00ff', linewidth=3, label=f'AUC = {roc_auc:.3f}')
ax2.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1)
ax2.fill_between(fpr, tpr, alpha=0.2, color='#ff00ff')
ax2.set_xlabel('False Positive Rate', color='#ccccff', fontsize=11)
ax2.set_ylabel('True Positive Rate', color='#ccccff', fontsize=11)
ax2.set_title('ROC CURVE', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.tick_params(colors='#888899')
ax2.legend(loc='lower right', facecolor='#1a1a2e', labelcolor='#ccccff', fontsize=12)
ax2.grid(True, alpha=0.2, color='#4444aa')
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Plot 3: Confusion Matrix with space theme
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')
im = ax3.imshow(cm, cmap='plasma')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['FALSE POSITIVE', 'EXOPLANET'], color='#ccccff', fontsize=10)
ax3.set_yticklabels(['FALSE POSITIVE', 'EXOPLANET'], color='#ccccff', fontsize=10)
ax3.set_xlabel('PREDICTED', color='#ccccff', fontsize=11, fontfamily='monospace')
ax3.set_ylabel('ACTUAL', color='#ccccff', fontsize=11, fontfamily='monospace')
ax3.set_title('CLASSIFICATION MATRIX', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', 
                color='white', fontsize=18, fontweight='bold', fontfamily='monospace')

# Plot 4: Mission Stats
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Draw decorative elements
for i in range(50):
    x, y = np.random.rand(2) * 10
    size = np.random.rand() * 3 + 0.5
    alpha = np.random.rand() * 0.5 + 0.2
    ax4.scatter(x, y, s=size, c='white', alpha=alpha)

# Title
ax4.text(5, 9.2, 'MISSION STATISTICS', ha='center', fontsize=16, fontweight='bold', 
         color='#00ffff', fontfamily='monospace')

# Stats with space styling
stats = [
    ('MODEL', 'NEURAL NETWORK', '#ff00ff'),
    ('ACCURACY', f'{final_acc*100:.2f}%', '#00ff88'),
    ('F1 SCORE', f'{final_f1:.4f}', '#ffff00'),
    ('AUC', f'{roc_auc:.4f}', '#ff8800'),
    ('PLANETS FOUND', f'{cm[1,1]:,}', '#00ffff'),
    ('FALSE ALARMS', f'{cm[0,1]:,}', '#ff4444'),
    ('GPU', 'RTX 3060', '#aa88ff'),
    ('PARAMETERS', f'{sum(p.numel() for p in model.parameters()):,}', '#88aaff'),
]

for i, (label, value, color) in enumerate(stats):
    y_pos = 7.5 - i * 0.9
    ax4.text(0.5, y_pos, f'{label}:', fontsize=11, color='#8888aa', fontfamily='monospace', va='center')
    ax4.text(9.5, y_pos, value, fontsize=11, color=color, fontweight='bold', fontfamily='monospace', ha='right', va='center')
    # Add separator line
    ax4.axhline(y=y_pos - 0.4, xmin=0.05, xmax=0.95, color='#333366', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(output_path, dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
