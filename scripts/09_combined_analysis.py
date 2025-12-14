"""
09 - Combined Analysis: Kepler + NASA Archive + TESS
Latest data from all sources - December 2025
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
data_dir = os.path.join(project_dir, 'data')
output_path = os.path.join(project_dir, 'graphs', '09_combined_analysis.png')

print("Loading all data sources...")

# Load all datasets
kepler = pd.read_csv(os.path.join(data_dir, 'kepler_exoplanets.csv'))
nasa = pd.read_csv(os.path.join(data_dir, 'nasa_confirmed_planets.csv'), low_memory=False)
tess = pd.read_csv(os.path.join(data_dir, 'tess_toi.csv'), low_memory=False)

print(f"Kepler KOI: {len(kepler):,}")
print(f"NASA Confirmed: {len(nasa):,}")
print(f"TESS TOI: {len(tess):,}")

# NASA data - confirmed planets
confirmed = nasa.copy()

# Discovery method breakdown
disc_methods = confirmed['discoverymethod'].value_counts().head(6)
print(f"\nDiscovery Methods:")
for method, count in disc_methods.items():
    print(f"  {method}: {count:,}")

# Discovery year trends
disc_years = confirmed['disc_year'].dropna().astype(int).value_counts().sort_index()
disc_years = disc_years[disc_years.index >= 1995]

# Mission breakdown (from disc_facility)
missions = confirmed['disc_facility'].value_counts().head(10)

# TESS disposition breakdown
tess_disp = tess['tfopwg_disp'].value_counts() if 'tfopwg_disp' in tess.columns else pd.Series()

# Planet size classification (using pl_rade - planet radius in Earth radii)
if 'pl_rade' in confirmed.columns:
    radii = confirmed['pl_rade'].dropna()
    radii = radii[(radii > 0) & (radii < 30)]
    
    size_bins = [0, 1, 2, 4, 10, 30]
    size_labels = ['Sub-Earth', 'Earth-like', 'Super-Earth', 'Neptune-like', 'Giant']
    size_counts = pd.cut(radii, bins=size_bins, labels=size_labels).value_counts()

# Create SPACE-THEMED visualization
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('#050510')

fig.suptitle('EXOPLANET CENSUS 2025', fontsize=28, fontweight='bold', 
             color='#00ffff', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.935, f'NASA Exoplanet Archive: {len(confirmed):,} Confirmed Planets | TESS: {len(tess):,} Objects of Interest', 
         ha='center', fontsize=14, color='#8888aa', fontfamily='monospace')

# Plot 1: Discovery Timeline
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')

colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(disc_years)))
bars = ax1.bar(disc_years.index, disc_years.values, color=colors)

ax1.set_xlabel('YEAR', color='#ccccff', fontfamily='monospace')
ax1.set_ylabel('PLANETS DISCOVERED', color='#ccccff', fontfamily='monospace')
ax1.set_title('EXOPLANET DISCOVERIES BY YEAR', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa', axis='y')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Annotate key missions
ax1.axvline(x=2009, color='#00ff88', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(2009.5, ax1.get_ylim()[1] * 0.9, 'Kepler', color='#00ff88', fontsize=9, fontfamily='monospace')
ax1.axvline(x=2018, color='#ff8800', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(2018.5, ax1.get_ylim()[1] * 0.9, 'TESS', color='#ff8800', fontsize=9, fontfamily='monospace')

# Plot 2: Discovery Methods
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

method_colors = ['#00ffff', '#ff00ff', '#ffff00', '#00ff88', '#ff8800', '#4488ff']
bars = ax2.barh(range(len(disc_methods)), disc_methods.values, color=method_colors[:len(disc_methods)])
ax2.set_yticks(range(len(disc_methods)))
ax2.set_yticklabels(disc_methods.index, color='#ccccff', fontfamily='monospace', fontsize=10)
ax2.set_xlabel('NUMBER OF PLANETS', color='#ccccff', fontfamily='monospace')
ax2.set_title('DETECTION METHODS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa', axis='x')
ax2.invert_yaxis()
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Add counts
for i, (bar, val) in enumerate(zip(bars, disc_methods.values)):
    ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, f'{val:,}',
             va='center', color='white', fontsize=9, fontweight='bold', fontfamily='monospace')

# Plot 3: Planet Size Distribution
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

if 'size_counts' in dir():
    size_order = ['Sub-Earth', 'Earth-like', 'Super-Earth', 'Neptune-like', 'Giant']
    size_vals = [size_counts.get(s, 0) for s in size_order]
    size_colors = ['#00ffff', '#00ff88', '#ffff00', '#ff8800', '#ff4466']
    
    bars = ax3.bar(range(len(size_order)), size_vals, color=size_colors)
    ax3.set_xticks(range(len(size_order)))
    ax3.set_xticklabels(['Sub-Earth\n(<1 R)', 'Earth-like\n(1-2 R)', 'Super-Earth\n(2-4 R)', 
                          'Neptune\n(4-10 R)', 'Giant\n(>10 R)'], color='#ccccff', fontfamily='monospace', fontsize=9)
    
    for bar, val in zip(bars, size_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val:,}',
                 ha='center', color='white', fontsize=10, fontweight='bold', fontfamily='monospace')
                 
ax3.set_ylabel('NUMBER OF PLANETS', color='#ccccff', fontfamily='monospace')
ax3.set_title('PLANET SIZE CLASSIFICATION', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa', axis='y')
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Plot 4: Key Statistics
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Stars
for i in range(40):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.3, '2025 EXOPLANET CENSUS', ha='center', fontsize=16, fontweight='bold', 
         color='#00ffff', fontfamily='monospace')

# Calculate unique host stars
n_hosts = confirmed['hostname'].nunique() if 'hostname' in confirmed.columns else 0

# TESS confirmed
tess_confirmed = len(tess[tess['tfopwg_disp'] == 'CP']) if 'tfopwg_disp' in tess.columns else 0

stats = [
    ('CONFIRMED EXOPLANETS', f'{len(confirmed):,}', '#00ffff'),
    ('UNIQUE HOST STARS', f'{n_hosts:,}', '#ffff00'),
    ('TESS OBJECTS OF INTEREST', f'{len(tess):,}', '#ff8800'),
    ('TESS CONFIRMED PLANETS', f'{tess_confirmed:,}', '#00ff88'),
    ('KEPLER KOI', f'{len(kepler):,}', '#ff00ff'),
    ('TRANSIT DETECTIONS', f'{disc_methods.get("Transit", 0):,}', '#4488ff'),
    ('RADIAL VELOCITY', f'{disc_methods.get("Radial Velocity", 0):,}', '#ff4466'),
    ('DISCOVERY PEAK YEAR', f'{disc_years.idxmax()}', '#88aaff'),
]

for i, (label, value, color) in enumerate(stats):
    y_pos = 7.8 - i * 0.9
    ax4.text(0.5, y_pos, f'{label}:', fontsize=10, color='#8888aa', fontfamily='monospace', va='center')
    ax4.text(9.5, y_pos, str(value), fontsize=10, color=color, fontweight='bold', fontfamily='monospace', ha='right', va='center')
    ax4.axhline(y=y_pos - 0.4, xmin=0.05, xmax=0.95, color='#333366', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(output_path, dpi=150, facecolor='#050510', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
print(f"\n=== 2025 EXOPLANET CENSUS ===")
print(f"Total Confirmed: {len(confirmed):,}")
print(f"TESS Objects: {len(tess):,}")
print(f"TESS Confirmed: {tess_confirmed:,}")
