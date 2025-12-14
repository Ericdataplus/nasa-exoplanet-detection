"""
05 - Stellar Host Analysis
Analyze what types of stars host exoplanets - stellar classification and patterns
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
output_path = os.path.join(project_dir, 'graphs', '05_stellar_analysis.png')

print("Loading NASA Kepler data...")
df = pd.read_csv(data_path)

# Focus on confirmed planets with stellar data
confirmed = df[df['koi_disposition'] == 'CONFIRMED'].copy()
confirmed = confirmed[confirmed['koi_steff'].notna() & confirmed['koi_srad'].notna()].copy()

print(f"Confirmed exoplanets with stellar data: {len(confirmed):,}")

# Classify stars by spectral type based on temperature
def get_spectral_type(temp):
    if pd.isna(temp): return 'Unknown'
    if temp >= 30000: return 'O'
    if temp >= 10000: return 'B'
    if temp >= 7500: return 'A'
    if temp >= 6000: return 'F'
    if temp >= 5200: return 'G (Sun-like)'
    if temp >= 3700: return 'K'
    if temp >= 2400: return 'M (Red Dwarf)'
    return 'L/T'

confirmed['spectral_type'] = confirmed['koi_steff'].apply(get_spectral_type)

# Calculate multi-planet systems
planet_counts = confirmed.groupby('kepid').size()
multi_planet_ids = planet_counts[planet_counts > 1].index
confirmed['multi_planet_system'] = confirmed['kepid'].isin(multi_planet_ids)

# Stats
spectral_counts = confirmed['spectral_type'].value_counts()
multi_planet_count = confirmed['multi_planet_system'].sum()
n_systems = confirmed['kepid'].nunique()
n_multi_systems = (planet_counts > 1).sum()

print(f"\nStellar Analysis:")
print(f"  Unique stellar systems: {n_systems}")
print(f"  Multi-planet systems: {n_multi_systems}")
print(f"  Most common star type: {spectral_counts.index[0]}")

# Create SPACE-THEMED visualization
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#050510')

fig.suptitle('STELLAR HOST ANALYSIS', fontsize=26, fontweight='bold', 
             color='#ffdd00', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.93, 'What Types of Stars Host Exoplanets?', 
         ha='center', fontsize=12, color='#8888aa', fontfamily='monospace')

# Plot 1: HR Diagram (Temperature vs Luminosity proxy)
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')

# Calculate luminosity
confirmed['luminosity'] = (confirmed['koi_srad'] ** 2) * ((confirmed['koi_steff'] / 5778) ** 4)
plot_df = confirmed[(confirmed['luminosity'] < 100) & (confirmed['luminosity'] > 0.01)]

# Color by spectral type
spectral_colors = {
    'O': '#9bb0ff', 'B': '#aabfff', 'A': '#cad7ff', 'F': '#f8f7ff',
    'G (Sun-like)': '#fff4ea', 'K': '#ffd2a1', 'M (Red Dwarf)': '#ffcc6f'
}
colors = plot_df['spectral_type'].map(spectral_colors).fillna('#888888')

scatter = ax1.scatter(plot_df['koi_steff'], plot_df['luminosity'], 
                       c=colors, alpha=0.6, s=15, edgecolors='none')

# Add Sun reference
ax1.scatter([5778], [1], c='#ffff00', s=200, marker='*', edgecolors='white', linewidths=1, zorder=10)
ax1.text(5778, 1.5, 'SUN', color='#ffff00', fontsize=10, ha='center', fontfamily='monospace')

ax1.set_xlabel('STELLAR TEMPERATURE (K)', color='#ccccff', fontfamily='monospace')
ax1.set_ylabel('LUMINOSITY (SOLAR)', color='#ccccff', fontfamily='monospace')
ax1.set_title('HR DIAGRAM OF HOST STARS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.set_xlim(7500, 3000)  # Reversed (hotter on left)
ax1.set_yscale('log')
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Plot 2: Spectral Type Distribution
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

# Order spectral types
type_order = ['O', 'B', 'A', 'F', 'G (Sun-like)', 'K', 'M (Red Dwarf)']
type_counts = [spectral_counts.get(t, 0) for t in type_order]
colors = [spectral_colors.get(t, '#888888') for t in type_order]

bars = ax2.barh(range(len(type_order)), type_counts, color=colors)
ax2.set_yticks(range(len(type_order)))
ax2.set_yticklabels(type_order, color='#ccccff', fontfamily='monospace')
ax2.set_xlabel('NUMBER OF HOST STARS', color='#ccccff', fontfamily='monospace')
ax2.set_title('SPECTRAL TYPE DISTRIBUTION', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa', axis='x')
ax2.invert_yaxis()
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Add value labels
for bar, val in zip(bars, type_counts):
    if val > 0:
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(val),
                 va='center', color='white', fontsize=10, fontweight='bold', fontfamily='monospace')

# Plot 3: Planets per System Distribution
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

system_sizes = planet_counts.value_counts().sort_index()
colors_sys = plt.cm.plasma(np.linspace(0.2, 0.8, len(system_sizes)))

bars = ax3.bar(system_sizes.index, system_sizes.values, color=colors_sys)
ax3.set_xlabel('PLANETS PER SYSTEM', color='#ccccff', fontfamily='monospace')
ax3.set_ylabel('NUMBER OF SYSTEMS', color='#ccccff', fontfamily='monospace')
ax3.set_title('MULTI-PLANET SYSTEMS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa', axis='y')
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Highlight multi-planet
for i, bar in enumerate(bars):
    if system_sizes.index[i] > 1:
        bar.set_edgecolor('#00ffff')
        bar.set_linewidth(2)

# Plot 4: Key Findings
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Stars
for i in range(30):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.2, 'STELLAR INSIGHTS', ha='center', fontsize=16, fontweight='bold', 
         color='#ffdd00', fontfamily='monospace')

# Find most common type for multi-planet
multi_df = confirmed[confirmed['multi_planet_system']]
multi_spectral = multi_df['spectral_type'].value_counts()

insights = [
    ('TOTAL HOST STARS', f'{n_systems:,}', '#00ffff'),
    ('MOST COMMON TYPE', spectral_counts.index[0], '#fff4ea'),
    ('MULTI-PLANET SYSTEMS', f'{n_multi_systems}', '#00ff88'),
    ('LARGEST SYSTEM', f'{planet_counts.max()} planets', '#ff00ff'),
    ('MULTI-PLANET FAVORITE', multi_spectral.index[0] if len(multi_spectral) > 0 else 'N/A', '#ffff00'),
    ('AVG STAR TEMP', f'{confirmed["koi_steff"].mean():.0f} K', '#ff8800'),
    ('AVG STAR RADIUS', f'{confirmed["koi_srad"].mean():.2f} R_Sun', '#88aaff'),
]

for i, (label, value, color) in enumerate(insights):
    y_pos = 7.5 - i * 0.95
    ax4.text(0.5, y_pos, f'{label}:', fontsize=10, color='#8888aa', fontfamily='monospace', va='center')
    ax4.text(9.5, y_pos, str(value), fontsize=10, color=color, fontweight='bold', fontfamily='monospace', ha='right', va='center')
    ax4.axhline(y=y_pos - 0.4, xmin=0.05, xmax=0.95, color='#333366', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(output_path, dpi=150, facecolor='#050510', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
