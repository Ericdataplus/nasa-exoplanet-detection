"""
07 - Anomaly Detection: Finding Unusual Exoplanets
Using Isolation Forest to detect outlier planets that don't fit typical patterns
This is cutting-edge - finding the weird planets that might be most interesting!
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'data', 'kepler_exoplanets.csv')
output_path = os.path.join(project_dir, 'graphs', '07_anomaly_detection.png')

print("Loading NASA Kepler data...")
df = pd.read_csv(data_path)

# Focus on confirmed planets
confirmed = df[df['koi_disposition'] == 'CONFIRMED'].copy()

# Features for anomaly detection
feature_cols = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_depth', 'koi_duration']
anomaly_df = confirmed[feature_cols].copy()

# Clean and prepare data
anomaly_df = anomaly_df.dropna()
anomaly_df = anomaly_df.replace([np.inf, -np.inf], np.nan).dropna()

# Remove extreme outliers that would skew the model
anomaly_df = anomaly_df[anomaly_df['koi_period'] < 1000]
anomaly_df = anomaly_df[anomaly_df['koi_prad'] < 30]
anomaly_df = anomaly_df[anomaly_df['koi_teq'] < 3000]

print(f"Planets for anomaly detection: {len(anomaly_df):,}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(anomaly_df)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
anomaly_df['anomaly'] = iso_forest.fit_predict(X_scaled)
anomaly_df['anomaly_score'] = iso_forest.decision_function(X_scaled)

# Anomalies are marked as -1
n_anomalies = (anomaly_df['anomaly'] == -1).sum()
print(f"Anomalous planets detected: {n_anomalies} ({n_anomalies/len(anomaly_df)*100:.1f}%)")

# Get most anomalous planets
anomalies = anomaly_df[anomaly_df['anomaly'] == -1].copy()
normal = anomaly_df[anomaly_df['anomaly'] == 1].copy()

# Find the most extreme anomalies
most_extreme = anomaly_df.nsmallest(10, 'anomaly_score')

print("\nMost Unusual Planets:")
for i, (idx, row) in enumerate(most_extreme.head(5).iterrows()):
    print(f"  {i+1}. Period: {row['koi_period']:.1f}d, Radius: {row['koi_prad']:.1f} R_E, Temp: {row['koi_teq']:.0f}K")

# Create SPACE-THEMED visualization
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#050510')

fig.suptitle('ANOMALY DETECTION: UNUSUAL EXOPLANETS', fontsize=26, fontweight='bold', 
             color='#ff00ff', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.93, 'Isolation Forest Identifies Planets That Break the Rules', 
         ha='center', fontsize=12, color='#8888aa', fontfamily='monospace')

# Plot 1: Period vs Radius (most useful for finding hot Jupiters, etc.)
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')

ax1.scatter(normal['koi_period'], normal['koi_prad'], c='#4466aa', alpha=0.4, s=15, label='Normal')
ax1.scatter(anomalies['koi_period'], anomalies['koi_prad'], c='#ff00ff', alpha=0.8, s=40, 
            edgecolors='white', linewidths=0.5, label=f'Anomalies ({n_anomalies})')

ax1.set_xlabel('ORBITAL PERIOD (DAYS)', color='#ccccff', fontfamily='monospace')
ax1.set_ylabel('PLANET RADIUS (EARTH RADII)', color='#ccccff', fontfamily='monospace')
ax1.set_title('PERIOD vs RADIUS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8)
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Plot 2: Temperature vs Radius
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

ax2.scatter(normal['koi_teq'], normal['koi_prad'], c='#4466aa', alpha=0.4, s=15)
ax2.scatter(anomalies['koi_teq'], anomalies['koi_prad'], c='#ff00ff', alpha=0.8, s=40,
            edgecolors='white', linewidths=0.5)

# Mark categories
ax2.axhline(y=4, color='#ffff00', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(2500, 4.5, 'Gas Giants', color='#ffff00', fontsize=9, fontfamily='monospace')
ax2.axhline(y=1.5, color='#00ff88', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(2500, 1.8, 'Super-Earths', color='#00ff88', fontsize=9, fontfamily='monospace')

ax2.set_xlabel('EQUILIBRIUM TEMPERATURE (K)', color='#ccccff', fontfamily='monospace')
ax2.set_ylabel('PLANET RADIUS (EARTH RADII)', color='#ccccff', fontfamily='monospace')
ax2.set_title('TEMPERATURE vs RADIUS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa')
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Plot 3: Anomaly Score Distribution
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

ax3.hist(normal['anomaly_score'], bins=50, alpha=0.7, color='#4466aa', label='Normal Planets')
ax3.hist(anomalies['anomaly_score'], bins=20, alpha=0.7, color='#ff00ff', label='Anomalies')

ax3.axvline(x=0, color='#ffff00', linestyle='--', linewidth=2, label='Decision Boundary')

ax3.set_xlabel('ANOMALY SCORE (lower = more unusual)', color='#ccccff', fontfamily='monospace')
ax3.set_ylabel('COUNT', color='#ccccff', fontfamily='monospace')
ax3.set_title('ANOMALY SCORE DISTRIBUTION', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8)
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa')
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Plot 4: Unusual Planet Profiles
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Stars
for i in range(30):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.2, 'UNUSUAL PLANET TYPES FOUND', ha='center', fontsize=14, fontweight='bold', 
         color='#ff00ff', fontfamily='monospace')

# Categorize anomalies
hot_jupiters = anomalies[(anomalies['koi_prad'] > 8) & (anomalies['koi_teq'] > 1000)]
ultra_short = anomalies[anomalies['koi_period'] < 1]
cool_giants = anomalies[(anomalies['koi_prad'] > 8) & (anomalies['koi_teq'] < 500)]
hot_earths = anomalies[(anomalies['koi_prad'] < 2) & (anomalies['koi_teq'] > 1500)]

findings = [
    ('TOTAL ANOMALIES', f'{n_anomalies}', '#ff00ff'),
    ('HOT JUPITERS', f'{len(hot_jupiters)} (scorching giants)', '#ff8800'),
    ('ULTRA-SHORT PERIOD', f'{len(ultra_short)} (<1 day orbit!)', '#00ffff'),
    ('COOL GIANTS', f'{len(cool_giants)} (cold gas giants)', '#4488ff'),
    ('HOT EARTHS', f'{len(hot_earths)} (lava worlds)', '#ff4444'),
    ('DETECTION RATE', f'{n_anomalies/len(anomaly_df)*100:.1f}%', '#ffff00'),
    ('ALGORITHM', 'Isolation Forest', '#88aaff'),
]

for i, (label, value, color) in enumerate(findings):
    y_pos = 7.5 - i * 0.95
    ax4.text(0.5, y_pos, f'{label}:', fontsize=10, color='#8888aa', fontfamily='monospace', va='center')
    ax4.text(9.5, y_pos, str(value), fontsize=10, color=color, fontweight='bold', fontfamily='monospace', ha='right', va='center')
    ax4.axhline(y=y_pos - 0.4, xmin=0.05, xmax=0.95, color='#333366', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(output_path, dpi=150, facecolor='#050510', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
