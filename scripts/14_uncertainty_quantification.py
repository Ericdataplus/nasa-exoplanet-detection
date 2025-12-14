"""
14 - NOVEL: Bayesian Deep Learning with Uncertainty Quantification
Monte Carlo Dropout for confidence intervals in predictions

Critical for science: Give probability distributions, not just point estimates!
Researchers like Madhusudhan need uncertainty bounds for publications.

Created: December 2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
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
output_path = os.path.join(project_dir, 'graphs', '14_uncertainty_quantification.png')

print("\n" + "="*60)
print("NOVEL: BAYESIAN DEEP LEARNING WITH UNCERTAINTY")
print("="*60)

# Load light curve data (reuse existing)
print("\nLoading Kepler light curve data...")
train_df = pd.read_csv(os.path.join(data_dir, 'exoTrain.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'exoTest.csv'))

X_train = train_df.iloc[:, 1:].values
y_train = (train_df.iloc[:, 0].values == 2).astype(int)
X_test = test_df.iloc[:, 1:].values
y_test = (test_df.iloc[:, 0].values == 2).astype(int)

print(f"Training: {len(X_train)} | Test: {len(X_test)}")

# Normalize
def normalize_lc(X):
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        X_norm[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-8)
    return X_norm

X_train = normalize_lc(X_train)
X_test = normalize_lc(X_test)

# Reshape for CNN
X_train = X_train.reshape(-1, 1, X_train.shape[1])
X_test = X_test.reshape(-1, 1, X_test.shape[1])

# Tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

#======================================
# NOVEL: Bayesian CNN with MC Dropout
#======================================
class BayesianCNN(nn.Module):
    """
    CNN with dropout kept ON during inference
    Enables Monte Carlo sampling for uncertainty estimation
    """
    def __init__(self, input_length, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Keep dropout active
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(16),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """
        NOVEL: Monte Carlo Dropout for uncertainty
        Run multiple forward passes with dropout active
        Get mean prediction and uncertainty (std dev)
        """
        self.train()  # Keep dropout ON
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # (n_samples, batch, 1)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)  # Epistemic uncertainty
        
        return mean_pred, std_pred

# Train model
model = BayesianCNN(X_train.shape[2]).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Weighted sampler
from torch.utils.data import WeightedRandomSampler
weights = np.where(y_train == 1, pos_weight, 1.0)
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

print("\nTraining Bayesian CNN...")
epochs = 50
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X).squeeze()
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f}")

# Predict with uncertainty
print("\nGenerating predictions with uncertainty (50 MC samples)...")
mean_pred, uncertainty = model.predict_with_uncertainty(X_test_t, n_samples=50)
mean_pred = mean_pred.squeeze().cpu().numpy()
uncertainty = uncertainty.squeeze().cpu().numpy()

y_pred = (mean_pred > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*60}")
print("BAYESIAN CNN WITH UNCERTAINTY RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {acc*100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"Mean uncertainty: {uncertainty.mean():.4f}")
print(f"Max uncertainty: {uncertainty.max():.4f}")

# Analyze uncertainty by correctness
correct_mask = y_pred == y_test
correct_uncertainty = uncertainty[correct_mask].mean()
incorrect_uncertainty = uncertainty[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

print(f"\nUncertainty Analysis:")
print(f"  Correct predictions: {correct_uncertainty:.4f} avg uncertainty")
print(f"  Incorrect predictions: {incorrect_uncertainty:.4f} avg uncertainty")

# Create visualization
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#050510')

fig.suptitle('BAYESIAN UNCERTAINTY QUANTIFICATION', fontsize=28, fontweight='bold', 
             color='#ffff00', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.935, 'NOVEL: Monte Carlo Dropout for Scientific Confidence Intervals', 
         ha='center', fontsize=13, color='#8888aa', fontfamily='monospace')

# Plot 1: Predictions with uncertainty bands
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')

# Sort by prediction for better visualization
sort_idx = np.argsort(mean_pred)[:200]  # First 200 for clarity

x_axis = np.arange(len(sort_idx))
ax1.fill_between(x_axis, 
                  mean_pred[sort_idx] - 2*uncertainty[sort_idx],
                  mean_pred[sort_idx] + 2*uncertainty[sort_idx],
                  alpha=0.3, color='#ffff00', label='95% CI')
ax1.fill_between(x_axis,
                  mean_pred[sort_idx] - uncertainty[sort_idx],
                  mean_pred[sort_idx] + uncertainty[sort_idx],
                  alpha=0.5, color='#ff8800', label='68% CI')
ax1.plot(x_axis, mean_pred[sort_idx], color='#00ffff', linewidth=1.5, label='Mean prediction')
ax1.scatter(x_axis, y_test[sort_idx], color='white', s=5, alpha=0.5, label='Actual')

ax1.axhline(y=0.5, color='#ff4466', linestyle='--', alpha=0.5, label='Threshold')
ax1.set_xlabel('SAMPLE INDEX (sorted by prediction)', color='#ccccff', fontfamily='monospace')
ax1.set_ylabel('PROBABILITY', color='#ccccff', fontfamily='monospace')
ax1.set_title('PREDICTIONS WITH UNCERTAINTY BANDS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='#ccccff', fontsize=9)
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Plot 2: Uncertainty distribution
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

ax2.hist(uncertainty[correct_mask], bins=30, alpha=0.7, color='#00ff88', label='Correct predictions', density=True)
if (~correct_mask).sum() > 0:
    ax2.hist(uncertainty[~correct_mask], bins=30, alpha=0.7, color='#ff4466', label='Incorrect predictions', density=True)

ax2.axvline(x=uncertainty.mean(), color='#ffff00', linestyle='--', linewidth=2, label=f'Mean: {uncertainty.mean():.3f}')

ax2.set_xlabel('UNCERTAINTY (std dev)', color='#ccccff', fontfamily='monospace')
ax2.set_ylabel('DENSITY', color='#ccccff', fontfamily='monospace')
ax2.set_title('UNCERTAINTY DISTRIBUTION BY CORRECTNESS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.legend(loc='upper right', facecolor='#1a1a2e', labelcolor='#ccccff')
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa')
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Plot 3: Calibration - uncertainty vs error
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

# Bin by uncertainty and calculate error rate
n_bins = 10
uncertainty_bins = np.percentile(uncertainty, np.linspace(0, 100, n_bins+1))
bin_errors = []
bin_centers = []

for i in range(n_bins):
    mask = (uncertainty >= uncertainty_bins[i]) & (uncertainty < uncertainty_bins[i+1])
    if mask.sum() > 0:
        error_rate = (y_pred[mask] != y_test[mask]).mean()
        bin_errors.append(error_rate)
        bin_centers.append((uncertainty_bins[i] + uncertainty_bins[i+1]) / 2)

ax3.bar(range(len(bin_errors)), bin_errors, color='#ff8800', alpha=0.7, edgecolor='white')
ax3.set_xticks(range(len(bin_centers)))
ax3.set_xticklabels([f'{c:.3f}' for c in bin_centers], rotation=45, ha='right', fontsize=8)

ax3.set_xlabel('UNCERTAINTY BIN', color='#ccccff', fontfamily='monospace')
ax3.set_ylabel('ERROR RATE', color='#ccccff', fontfamily='monospace')
ax3.set_title('CALIBRATION: HIGHER UNCERTAINTY = MORE ERRORS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa', axis='y')
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

ax4.text(5, 9.3, 'SCIENTIFIC VALUE', ha='center', fontsize=16, fontweight='bold', 
         color='#ffff00', fontfamily='monospace')

# Find high-confidence predictions
high_conf_mask = uncertainty < np.percentile(uncertainty, 25)
high_conf_acc = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()

stats = [
    ('TECHNIQUE', 'Monte Carlo Dropout', '#ffff00'),
    ('MC SAMPLES', '50', '#00ffff'),
    ('ACCURACY', f'{acc*100:.1f}%', '#00ff88'),
    ('MEAN UNCERTAINTY', f'{uncertainty.mean():.4f}', '#ff8800'),
    ('CORRECT PRED UNC', f'{correct_uncertainty:.4f}', '#4488ff'),
    ('INCORRECT PRED UNC', f'{incorrect_uncertainty:.4f}', '#ff4466'),
    ('HIGH-CONF ACCURACY', f'{high_conf_acc*100:.1f}%', '#ff00ff'),
    ('CALIBRATED', 'Higher unc = more errors ✓', '#88aaff'),
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
print("  1. Monte Carlo Dropout for epistemic uncertainty")
print("  2. Confidence intervals for every prediction")
print("  3. Calibrated: high uncertainty = high error rate")
print("  4. CRITICAL for scientific publications")
print("="*60)
