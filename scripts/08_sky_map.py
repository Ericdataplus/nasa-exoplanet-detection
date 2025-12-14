"""
08 - Planet Discovery Patterns by Sky Position
Analyzing where in the Kepler field of view planets are found
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
output_path = os.path.join(project_dir, 'graphs', '08_sky_map.png')

print("Loading NASA Kepler data...")
df = pd.read_csv(data_path)

# Get coordinate data
confirmed = df[df['koi_disposition'] == 'CONFIRMED'].copy()
false_pos = df[df['koi_disposition'] == 'FALSE POSITIVE'].copy()

# Clean RA/Dec data
confirmed = confirmed[(confirmed['ra'].notna()) & (confirmed['dec'].notna())].copy()
false_pos = false_pos[(false_pos['ra'].notna()) & (false_pos['dec'].notna())].copy()

print(f"Confirmed planets with coordinates: {len(confirmed):,}")

# Calculate brightness-weighted density
confirmed['brightness'] = 10 ** (-confirmed['koi_kepmag'] / 2.5)

# Create SPACE-THEMED sky map
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#050510')

fig.suptitle('KEPLER FIELD OF VIEW: SKY MAP', fontsize=26, fontweight='bold', 
             color='#00ffff', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.93, 'Where in the Sky Are Exoplanets Hiding?', 
         ha='center', fontsize=12, color='#8888aa', fontfamily='monospace')

# Plot 1: Sky map of confirmed planets
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#050510')

# Size by planet radius, color by temperature
sizes = np.clip(confirmed['koi_prad'].fillna(2) * 5, 5, 100)
colors = confirmed['koi_teq'].fillna(500)

scatter = ax1.scatter(confirmed['ra'], confirmed['dec'], 
                       c=colors, cmap='plasma', alpha=0.6, s=sizes,
                       vmin=200, vmax=2000)

cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Temperature (K)', color='#ccccff')
cbar.ax.tick_params(colors='#888899')

ax1.set_xlabel('RIGHT ASCENSION (degrees)', color='#ccccff', fontfamily='monospace')
ax1.set_ylabel('DECLINATION (degrees)', color='#ccccff', fontfamily='monospace')
ax1.set_title('CONFIRMED EXOPLANETS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Plot 2: Comparison - Confirmed vs False Positive distribution
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#050510')

ax2.scatter(false_pos['ra'], false_pos['dec'], c='#ff4466', alpha=0.2, s=5, label='False Positives')
ax2.scatter(confirmed['ra'], confirmed['dec'], c='#00ff88', alpha=0.5, s=10, label='Confirmed')

ax2.set_xlabel('RIGHT ASCENSION (degrees)', color='#ccccff', fontfamily='monospace')
ax2.set_ylabel('DECLINATION (degrees)', color='#ccccff', fontfamily='monospace')
ax2.set_title('CONFIRMED vs FALSE POSITIVES', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8, markerscale=3)
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa')
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Plot 3: Declination distribution
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

ax3.hist(confirmed['dec'], bins=40, alpha=0.7, color='#00ff88', label='Confirmed', orientation='horizontal')
ax3.hist(false_pos['dec'], bins=40, alpha=0.5, color='#ff4466', label='False Positives', orientation='horizontal')

ax3.set_xlabel('COUNT', color='#ccccff', fontfamily='monospace')
ax3.set_ylabel('DECLINATION (degrees)', color='#ccccff', fontfamily='monospace')
ax3.set_title('DECLINATION DISTRIBUTION', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8)
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa')
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Plot 4: Statistics
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Stars
for i in range(50):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.2, 'KEPLER FIELD STATISTICS', ha='center', fontsize=16, fontweight='bold', 
         color='#00ffff', fontfamily='monospace')

# Calculate stats
ra_range = confirmed['ra'].max() - confirmed['ra'].min()
dec_range = confirmed['dec'].max() - confirmed['dec'].min()
avg_mag = confirmed['koi_kepmag'].mean()

stats = [
    ('FIELD CENTER RA', f'{confirmed["ra"].median():.1f} deg', '#00ffff'),
    ('FIELD CENTER DEC', f'{confirmed["dec"].median():.1f} deg', '#00ff88'),
    ('RA RANGE', f'{ra_range:.1f} deg', '#ffff00'),
    ('DEC RANGE', f'{dec_range:.1f} deg', '#ff8800'),
    ('CONSTELLATION', 'Cygnus-Lyra', '#ff00ff'),
    ('AVG STAR MAGNITUDE', f'{avg_mag:.1f}', '#88aaff'),
    ('OBSERVATION YEARS', '2009-2018', '#4488ff'),
]

for i, (label, value, color) in enumerate(stats):
    y_pos = 7.5 - i * 0.95
    ax4.text(0.5, y_pos, f'{label}:', fontsize=10, color='#8888aa', fontfamily='monospace', va='center')
    ax4.text(9.5, y_pos, str(value), fontsize=10, color=color, fontweight='bold', fontfamily='monospace', ha='right', va='center')
    ax4.axhline(y=y_pos - 0.4, xmin=0.05, xmax=0.95, color='#333366', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(output_path, dpi=150, facecolor='#050510', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
