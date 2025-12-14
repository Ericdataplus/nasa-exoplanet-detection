"""
01 - Data Overview & Class Distribution
NASA Kepler Exoplanet Detection Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'data', 'kepler_exoplanets.csv')
output_path = os.path.join(project_dir, 'graphs', '01_data_overview.png')

print("Loading NASA Kepler data...")
df = pd.read_csv(data_path)

print(f"Total Kepler Objects of Interest: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"\nClass Distribution:")
print(df['koi_disposition'].value_counts())

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('NASA Kepler Exoplanet Search: Data Overview', fontsize=22, fontweight='bold', color='white', y=0.98)

# Plot 1: Class distribution
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')
classes = df['koi_disposition'].value_counts()
colors = ['#ff6b6b', '#4ecdc4', '#ffd93d']
bars = ax1.bar(range(len(classes)), classes.values, color=colors)
ax1.set_xticks(range(len(classes)))
ax1.set_xticklabels(['False Positive', 'Confirmed Planet', 'Candidate'], color='white', fontsize=11)
ax1.set_ylabel('Count', color='white')
ax1.set_title('Object Classification', color='white', fontsize=14, fontweight='bold')
ax1.tick_params(colors='white')
for spine in ax1.spines.values(): spine.set_color('#30363d')

# Add labels
for bar, val in zip(bars, classes.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{val:,}',
             ha='center', color='white', fontsize=12, fontweight='bold')

# Plot 2: Planet score distribution
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')
df_clean = df[df['koi_score'].notna()]
confirmed = df_clean[df_clean['koi_disposition'] == 'CONFIRMED']['koi_score']
false_pos = df_clean[df_clean['koi_disposition'] == 'FALSE POSITIVE']['koi_score']
candidate = df_clean[df_clean['koi_disposition'] == 'CANDIDATE']['koi_score']

ax2.hist(confirmed, bins=30, alpha=0.7, label='Confirmed', color='#4ecdc4')
ax2.hist(false_pos, bins=30, alpha=0.7, label='False Positive', color='#ff6b6b')
ax2.hist(candidate, bins=30, alpha=0.7, label='Candidate', color='#ffd93d')
ax2.set_xlabel('Disposition Score (0-1)', color='white')
ax2.set_ylabel('Count', color='white')
ax2.set_title('Planet Score Distribution', color='white', fontsize=14, fontweight='bold')
ax2.legend(facecolor='#161b22', labelcolor='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values(): spine.set_color('#30363d')

# Plot 3: Orbital period vs radius (confirmed planets)
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')
confirmed_df = df[df['koi_disposition'] == 'CONFIRMED']
confirmed_df = confirmed_df[(confirmed_df['koi_period'].notna()) & (confirmed_df['koi_prad'].notna())]
confirmed_df = confirmed_df[(confirmed_df['koi_period'] < 500) & (confirmed_df['koi_prad'] < 30)]

scatter = ax3.scatter(confirmed_df['koi_period'], confirmed_df['koi_prad'], 
                       c=confirmed_df['koi_score'], cmap='viridis', alpha=0.6, s=20)
ax3.set_xlabel('Orbital Period (days)', color='white')
ax3.set_ylabel('Planet Radius (Earth radii)', color='white')
ax3.set_title('Confirmed Exoplanets: Period vs Radius', color='white', fontsize=14, fontweight='bold')
ax3.tick_params(colors='white')
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Score', color='white')
cbar.ax.tick_params(colors='white')
for spine in ax3.spines.values(): spine.set_color('#30363d')

# Plot 4: Key statistics
ax4 = axes[1, 1]
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values(): spine.set_color('#30363d')

ax4.text(0.5, 0.92, 'Dataset Statistics', fontsize=16, fontweight='bold', ha='center', color='white', transform=ax4.transAxes)

stats = [
    ('Total Objects:', f'{len(df):,}', '#ffd700'),
    ('Confirmed Planets:', f'{len(df[df["koi_disposition"]=="CONFIRMED"]):,}', '#4ecdc4'),
    ('False Positives:', f'{len(df[df["koi_disposition"]=="FALSE POSITIVE"]):,}', '#ff6b6b'),
    ('Candidates:', f'{len(df[df["koi_disposition"]=="CANDIDATE"]):,}', '#ffd93d'),
    ('Features:', f'{len(df.columns)}', '#58a6ff'),
    ('Data Source:', 'NASA Kepler', '#a371f7'),
]

for i, (label, value, color) in enumerate(stats):
    y_pos = 0.78 - i * 0.11
    ax4.text(0.08, y_pos, label, fontsize=12, color='#8b949e', transform=ax4.transAxes, va='center')
    ax4.text(0.55, y_pos, value, fontsize=12, color=color, fontweight='bold', transform=ax4.transAxes, va='center')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
