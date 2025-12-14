"""
02 - Exoplanet Detection Model
Random Forest & XGBoost for classifying Kepler objects
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'data', 'kepler_exoplanets.csv')
output_path = os.path.join(project_dir, 'graphs', '02_detection_model.png')

print("Loading NASA Kepler data...")
df = pd.read_csv(data_path)

# Focus on confirmed vs false positive (binary classification for clearer results)
df_binary = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
df_binary['is_planet'] = (df_binary['koi_disposition'] == 'CONFIRMED').astype(int)

print(f"Binary classification: {len(df_binary):,} samples")
print(f"  Confirmed: {df_binary['is_planet'].sum():,}")
print(f"  False Positive: {len(df_binary) - df_binary['is_planet'].sum():,}")

# Select features (key Kepler parameters)
feature_cols = [
    'koi_score',        # Disposition score
    'koi_period',       # Orbital period
    'koi_prad',         # Planet radius
    'koi_teq',          # Equilibrium temperature
    'koi_insol',        # Insolation flux
    'koi_depth',        # Transit depth
    'koi_duration',     # Transit duration
    'koi_steff',        # Stellar effective temperature
    'koi_slogg',        # Stellar surface gravity
    'koi_srad',         # Stellar radius
]

# Prepare features
X = df_binary[feature_cols].copy()
y = df_binary['is_planet']

# Fill missing values with median
X = X.fillna(X.median())

# Remove rows with infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train):,}")
print(f"Test set: {len(X_test):,}")

# Train models
print("\nTraining models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {
        'accuracy': acc,
        'f1': f1,
        'predictions': y_pred,
        'model': model
    }
    print(f"  {name}: Accuracy={acc*100:.1f}%, F1={f1:.3f}")

best_name = max(results, key=lambda x: results[x]['f1'])
best = results[best_name]

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best['model'].feature_importances_
}).sort_values('importance', ascending=False)

# Better feature names
feature_names = {
    'koi_score': 'Disposition Score',
    'koi_period': 'Orbital Period',
    'koi_prad': 'Planet Radius',
    'koi_teq': 'Eq. Temperature',
    'koi_insol': 'Insolation Flux',
    'koi_depth': 'Transit Depth',
    'koi_duration': 'Transit Duration',
    'koi_steff': 'Star Temperature',
    'koi_slogg': 'Star Surface Gravity',
    'koi_srad': 'Star Radius',
}
feature_importance['feature_name'] = feature_importance['feature'].map(feature_names)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Exoplanet Detection Model: ML Classification', fontsize=22, fontweight='bold', color='white', y=0.98)

# Plot 1: Model comparison
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')
names = list(results.keys())
accs = [results[n]['accuracy'] * 100 for n in names]
f1s = [results[n]['f1'] * 100 for n in names]
x = np.arange(len(names))
width = 0.35

bars1 = ax1.bar(x - width/2, accs, width, color='#4ecdc4', label='Accuracy')
bars2 = ax1.bar(x + width/2, f1s, width, color='#ffd93d', label='F1 Score')

ax1.set_xticks(x)
ax1.set_xticklabels(names, color='white')
ax1.set_ylabel('Score (%)', color='white')
ax1.set_title('Model Comparison', color='white', fontsize=14, fontweight='bold')
ax1.legend(facecolor='#161b22', labelcolor='white')
ax1.set_ylim(0, 100)
ax1.tick_params(colors='white')
for spine in ax1.spines.values(): spine.set_color('#30363d')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%',
                 ha='center', color='white', fontsize=9, fontweight='bold')

# Plot 2: Feature importance
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')
colors = plt.cm.viridis(np.linspace(0.9, 0.3, len(feature_importance)))
bars = ax2.barh(feature_importance['feature_name'], feature_importance['importance'] * 100, color=colors)
ax2.set_xlabel('Importance (%)', color='white')
ax2.set_title('What Detects Real Planets?', color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white')
ax2.invert_yaxis()
for spine in ax2.spines.values(): spine.set_color('#30363d')

# Plot 3: Confusion matrix
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')
cm = confusion_matrix(y_test, best['predictions'])
im = ax3.imshow(cm, cmap='Blues')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['False Positive', 'Real Planet'], color='white')
ax3.set_yticklabels(['False Positive', 'Real Planet'], color='white')
ax3.set_xlabel('Predicted', color='white')
ax3.set_ylabel('Actual', color='white')
ax3.set_title(f'{best_name} Confusion Matrix', color='white', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', 
                       color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16, fontweight='bold')

# Plot 4: Results summary
ax4 = axes[1, 1]
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values(): spine.set_color('#30363d')

ax4.text(0.5, 0.92, 'Detection Results', fontsize=16, fontweight='bold', ha='center', color='white', transform=ax4.transAxes)

summary = [
    ('Best Model:', best_name, '#ffd700'),
    ('Accuracy:', f'{best["accuracy"]*100:.1f}%', '#4ecdc4'),
    ('F1 Score:', f'{best["f1"]*100:.1f}%', '#ffd93d'),
    ('Training Size:', f'{len(X_train):,}', '#58a6ff'),
    ('Top Feature:', feature_importance.iloc[0]['feature_name'], '#56d364'),
    ('True Detections:', f'{cm[1,1]:,}', '#a371f7'),
]

for i, (label, value, color) in enumerate(summary):
    y_pos = 0.78 - i * 0.11
    ax4.text(0.08, y_pos, label, fontsize=12, color='#8b949e', transform=ax4.transAxes, va='center')
    ax4.text(0.50, y_pos, value, fontsize=12, color=color, fontweight='bold', transform=ax4.transAxes, va='center')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
print(f"\nTop 3 features for detecting real planets:")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {row['feature_name']}: {row['importance']*100:.1f}%")
