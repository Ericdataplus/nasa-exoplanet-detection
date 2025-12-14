"""
13 - NOVEL: End-to-End Habitability Predictor
Direct prediction of habitability metrics from stellar/planetary features

This goes beyond classification to predict:
1. Whether planet is in habitable zone
2. Whether planet is Earth-sized
3. Estimated equilibrium temperature
4. HABITABILITY SCORE (novel composite metric)

Created: December 2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
output_path = os.path.join(project_dir, 'graphs', '13_habitability_predictor.png')

print("\n" + "="*60)
print("NOVEL: END-TO-END HABITABILITY PREDICTOR")
print("="*60)

# Load NASA confirmed planets
print("\nLoading NASA confirmed planets...")
df = pd.read_csv(os.path.join(data_dir, 'nasa_confirmed_planets.csv'), low_memory=False)
print(f"Total planets: {len(df):,}")

# Select features for habitability prediction
feature_cols = [
    'pl_orbper',    # Orbital period
    'pl_orbsmax',   # Semi-major axis
    'pl_rade',      # Planet radius
    'st_teff',      # Stellar temperature
    'st_rad',       # Stellar radius
    'st_mass',      # Stellar mass
    'pl_insol',     # Insolation flux
]

# Target: Create habitability score (novel composite metric)
print("\nCreating habitability targets...")

# Filter to planets with required data
df_clean = df[feature_cols + ['pl_eqt', 'pl_name']].dropna().copy()
print(f"Planets with complete data: {len(df_clean):,}")

# Calculate habitability components
T_sun = 5778
df_clean['st_luminosity'] = (df_clean['st_rad'] ** 2) * ((df_clean['st_teff'] / T_sun) ** 4)

# Habitable zone boundaries
df_clean['hz_inner'] = 0.99 * np.sqrt(df_clean['st_luminosity'])
df_clean['hz_outer'] = 1.70 * np.sqrt(df_clean['st_luminosity'])

# Binary targets
df_clean['in_hz'] = ((df_clean['pl_orbsmax'] >= df_clean['hz_inner']) & 
                      (df_clean['pl_orbsmax'] <= df_clean['hz_outer'])).astype(float)
df_clean['earth_sized'] = ((df_clean['pl_rade'] >= 0.5) & 
                           (df_clean['pl_rade'] <= 2.0)).astype(float)

# Temperature habitability (0-1 scale, peak at Earth temp ~255K)
earth_temp = 255
df_clean['temp_score'] = np.exp(-((df_clean['pl_eqt'] - earth_temp) / 100) ** 2)

# NOVEL: Composite Habitability Score (0-1)
# Weighted combination: HZ proximity + size + temperature
df_clean['habitability_score'] = (
    0.4 * df_clean['in_hz'] + 
    0.3 * df_clean['earth_sized'] + 
    0.3 * df_clean['temp_score']
)

print(f"\nHabitability Score Distribution:")
print(f"  Min: {df_clean['habitability_score'].min():.3f}")
print(f"  Max: {df_clean['habitability_score'].max():.3f}")
print(f"  Mean: {df_clean['habitability_score'].mean():.3f}")
print(f"  Highly habitable (>0.7): {(df_clean['habitability_score'] > 0.7).sum()}")

# Prepare features and targets
X = df_clean[feature_cols].values
y_hab_score = df_clean['habitability_score'].values
y_in_hz = df_clean['in_hz'].values
y_earth_sized = df_clean['earth_sized'].values
y_temp = df_clean['pl_eqt'].values

# Scale features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Scale temperature target (for regression)
scaler_temp = MinMaxScaler()
y_temp_scaled = scaler_temp.fit_transform(y_temp.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train_hab, y_test_hab, y_train_hz, y_test_hz, y_train_es, y_test_es, y_train_temp, y_test_temp = train_test_split(
    X_scaled, y_hab_score, y_in_hz, y_earth_sized, y_temp_scaled, test_size=0.2, random_state=42
)

print(f"\nTraining: {len(X_train)} | Test: {len(X_test)}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

y_train_t = torch.FloatTensor(np.column_stack([y_train_hab, y_train_hz, y_train_es, y_train_temp])).to(device)
y_test_t = torch.FloatTensor(np.column_stack([y_test_hab, y_test_hz, y_test_es, y_test_temp])).to(device)

#======================================
# NOVEL: Multi-Task Habitability Network
#======================================
class HabitabilityNet(nn.Module):
    """
    NOVEL ARCHITECTURE:
    Multi-task learning to predict:
    1. Habitability score (regression)
    2. In habitable zone (binary)
    3. Earth-sized (binary)
    4. Equilibrium temperature (regression)
    """
    def __init__(self, n_features):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Task-specific heads
        self.hab_score_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 score
        )
        
        self.hz_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Binary
        )
        
        self.earth_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Binary
        )
        
        self.temp_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Scaled temp
        )
        
    def forward(self, x):
        features = self.encoder(x)
        
        hab_score = self.hab_score_head(features)
        in_hz = self.hz_head(features)
        earth_sized = self.earth_head(features)
        temp = self.temp_head(features)
        
        return torch.cat([hab_score, in_hz, earth_sized, temp], dim=1)

# Initialize
model = HabitabilityNet(len(feature_cols)).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Multi-task loss
def multi_task_loss(pred, target):
    # Weighted losses for each task
    loss_hab = nn.MSELoss()(pred[:, 0], target[:, 0]) * 2.0  # Main task
    loss_hz = nn.BCELoss()(pred[:, 1], target[:, 1])
    loss_es = nn.BCELoss()(pred[:, 2], target[:, 2])
    loss_temp = nn.MSELoss()(pred[:, 3], target[:, 3])
    
    return loss_hab + loss_hz + loss_es + loss_temp

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training
print("\nTraining Multi-Task Habitability Network...")
epochs = 100
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = multi_task_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    scheduler.step(epoch_loss)
    
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            hab_mae = mean_absolute_error(y_test_t[:, 0].cpu(), test_pred[:, 0].cpu())
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f} - Hab MAE: {hab_mae:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_t).cpu().numpy()

pred_hab = predictions[:, 0]
pred_hz = (predictions[:, 1] > 0.5).astype(int)
pred_es = (predictions[:, 2] > 0.5).astype(int)
pred_temp = scaler_temp.inverse_transform(predictions[:, 3].reshape(-1, 1)).flatten()

actual_temp = scaler_temp.inverse_transform(y_test_temp.reshape(-1, 1)).flatten()

# Metrics
hab_mae = mean_absolute_error(y_test_hab, pred_hab)
hab_r2 = r2_score(y_test_hab, pred_hab)
hz_acc = (pred_hz == y_test_hz).mean()
es_acc = (pred_es == y_test_es).mean()
temp_mae = mean_absolute_error(actual_temp, pred_temp)

print(f"\n{'='*60}")
print("HABITABILITY PREDICTOR RESULTS")
print(f"{'='*60}")
print(f"Habitability Score MAE: {hab_mae:.4f}")
print(f"Habitability Score R²: {hab_r2:.4f}")
print(f"Habitable Zone Accuracy: {hz_acc*100:.1f}%")
print(f"Earth-Sized Accuracy: {es_acc*100:.1f}%")
print(f"Temperature MAE: {temp_mae:.0f}K")

# Create visualization
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#050510')

fig.suptitle('HABITABILITY PREDICTOR', fontsize=28, fontweight='bold', 
             color='#00ff88', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.935, 'NOVEL: Multi-Task Learning for End-to-End Habitability Assessment', 
         ha='center', fontsize=13, color='#8888aa', fontfamily='monospace')

# Plot 1: Predicted vs Actual Habitability Score
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')

scatter = ax1.scatter(y_test_hab, pred_hab, c=y_test_hab, cmap='viridis', alpha=0.6, s=30)
ax1.plot([0, 1], [0, 1], 'w--', linewidth=2, label='Perfect prediction')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

ax1.set_xlabel('ACTUAL HABITABILITY SCORE', color='#ccccff', fontfamily='monospace')
ax1.set_ylabel('PREDICTED HABITABILITY SCORE', color='#ccccff', fontfamily='monospace')
ax1.set_title(f'HABITABILITY PREDICTION (R² = {hab_r2:.3f})', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.legend(loc='lower right', facecolor='#1a1a2e', labelcolor='#ccccff')
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Score', color='#ccccff')
cbar.ax.tick_params(colors='#888899')

# Plot 2: Temperature Prediction
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

ax2.scatter(actual_temp, pred_temp, c='#ff8800', alpha=0.5, s=20)
ax2.plot([0, 3000], [0, 3000], 'w--', linewidth=2)

# Highlight habitable range
ax2.axvspan(200, 350, alpha=0.1, color='#00ff88')
ax2.axhspan(200, 350, alpha=0.1, color='#00ff88')

ax2.set_xlabel('ACTUAL TEMPERATURE (K)', color='#ccccff', fontfamily='monospace')
ax2.set_ylabel('PREDICTED TEMPERATURE (K)', color='#ccccff', fontfamily='monospace')
ax2.set_title(f'TEMPERATURE PREDICTION (MAE = {temp_mae:.0f}K)', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.set_xlim(0, 3000)
ax2.set_ylim(0, 3000)
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa')
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Plot 3: Multi-task performance
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

tasks = ['Habitability\nScore', 'Habitable\nZone', 'Earth-\nSized', 'Temperature']
metrics = [hab_r2, hz_acc, es_acc, max(0, 1 - temp_mae/300)]  # Normalize temp MAE
colors = ['#00ff88', '#00ffff', '#ffff00', '#ff8800']

bars = ax3.bar(tasks, metrics, color=colors)
ax3.set_ylim(0, 1)
ax3.axhline(y=0.9, color='white', linestyle='--', alpha=0.3, label='90% threshold')

for bar, val in zip(bars, metrics):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
             ha='center', color='white', fontsize=11, fontweight='bold')

ax3.set_ylabel('PERFORMANCE (R² or Accuracy)', color='#ccccff', fontfamily='monospace')
ax3.set_title('MULTI-TASK PERFORMANCE', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.tick_params(colors='#888899')
ax3.set_xticklabels(tasks, color='#ccccff', fontsize=10)
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Plot 4: Results Summary
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

for i in range(40):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.3, 'NOVEL APPROACH', ha='center', fontsize=16, fontweight='bold', 
         color='#00ff88', fontfamily='monospace')

stats = [
    ('INNOVATION', 'Multi-Task Habitability', '#00ff88'),
    ('OUTPUTS', '4 simultaneous predictions', '#00ffff'),
    ('HAB SCORE R²', f'{hab_r2:.3f}', '#ffff00'),
    ('HZ ACCURACY', f'{hz_acc*100:.1f}%', '#ff00ff'),
    ('EARTH-SIZED ACC', f'{es_acc*100:.1f}%', '#ff8800'),
    ('TEMP MAE', f'{temp_mae:.0f}K', '#4488ff'),
    ('PARAMETERS', f'{sum(p.numel() for p in model.parameters()):,}', '#88aaff'),
    ('GPU', 'RTX 3060 12GB', '#ff4466'),
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
print("  1. Multi-task learning for 4 outputs simultaneously")
print("  2. Novel composite Habitability Score metric")
print("  3. End-to-end prediction from stellar features")
print("  4. Scientific interpretability maintained")
print("="*60)
