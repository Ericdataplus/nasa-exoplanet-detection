"""
10 - Habitable Zone Analysis (Updated with NASA 2025 Data)
Find potentially habitable planets in the full NASA archive
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
data_path = os.path.join(project_dir, 'data', 'nasa_confirmed_planets.csv')
output_path = os.path.join(project_dir, 'graphs', '10_habitable_zone_2025.png')

print("Loading NASA Confirmed Planets (2025)...")
df = pd.read_csv(data_path, low_memory=False)
print(f"Total confirmed planets: {len(df):,}")

# Key columns:
# pl_rade - planet radius (Earth radii)
# pl_eqt - equilibrium temperature (K)
# pl_orbper - orbital period (days)
# pl_orbsmax - semi-major axis (AU)
# st_teff - stellar effective temperature (K)
# st_rad - stellar radius (solar radii)
# st_lum - stellar luminosity (log solar)

# Filter for planets with radius data
has_radius = df['pl_rade'].notna()
print(f"Planets with radius data: {has_radius.sum():,}")

# Calculate habitable zone for each star
# Using Kopparapu et al. (2013) conservative HZ
# Inner edge: 0.99 AU * sqrt(L/L_sun)
# Outer edge: 1.70 AU * sqrt(L/L_sun)

# Get stellar luminosity
if 'st_lum' in df.columns:
    df['luminosity'] = 10 ** df['st_lum'].fillna(0)  # Convert from log
else:
    # Calculate from temperature and radius
    T_sun = 5778
    df['luminosity'] = (df['st_rad'].fillna(1) ** 2) * ((df['st_teff'].fillna(5778) / T_sun) ** 4)

# Calculate HZ boundaries
df['hz_inner'] = 0.99 * np.sqrt(df['luminosity'])
df['hz_outer'] = 1.70 * np.sqrt(df['luminosity'])

# Check if planet is in HZ
df['in_hz'] = (df['pl_orbsmax'] >= df['hz_inner']) & (df['pl_orbsmax'] <= df['hz_outer'])

# Earth-like size: 0.5 to 2 Earth radii
df['earth_sized'] = (df['pl_rade'] >= 0.5) & (df['pl_rade'] <= 2.0)

# Potentially habitable: in HZ AND Earth-sized
df['potentially_habitable'] = df['in_hz'] & df['earth_sized']

# Also check by temperature (200-300K is roughly habitable)
df['temp_habitable'] = (df['pl_eqt'] >= 200) & (df['pl_eqt'] <= 350)

# Combined criteria
df['best_candidates'] = df['potentially_habitable'] | (df['earth_sized'] & df['temp_habitable'])

n_in_hz = df['in_hz'].sum()
n_earth_sized = df['earth_sized'].sum()
n_potentially_habitable = df['potentially_habitable'].sum()
n_temp_habitable = (df['earth_sized'] & df['temp_habitable']).sum()
n_best = df['best_candidates'].sum()

print(f"\nHabitable Zone Analysis (2025):")
print(f"  Planets in Habitable Zone: {n_in_hz}")
print(f"  Earth-sized (0.5-2 R): {n_earth_sized}")
print(f"  POTENTIALLY HABITABLE: {n_potentially_habitable}")
print(f"  Earth-sized + Temp OK: {n_temp_habitable}")
print(f"  BEST CANDIDATES: {n_best}")

# Find the best candidates
candidates = df[df['best_candidates']].copy()
candidates = candidates.sort_values('pl_rade')

print(f"\nTop Potentially Habitable Planets:")
for i, (idx, row) in enumerate(candidates.head(10).iterrows()):
    name = row['pl_name']
    radius = row['pl_rade']
    temp = row['pl_eqt']
    if pd.notna(radius) and pd.notna(temp):
        print(f"  {i+1}. {name}: R={radius:.2f} Earth, T={temp:.0f}K")
    else:
        print(f"  {i+1}. {name}")

# Create visualization
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#050510')

fig.suptitle('HABITABLE PLANET SEARCH 2025', fontsize=26, fontweight='bold', 
             color='#00ff88', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.93, f'{len(df):,} Confirmed Exoplanets Analyzed | {n_best} Best Habitable Candidates', 
         ha='center', fontsize=12, color='#8888aa', fontfamily='monospace')

# Plot 1: Radius vs Temperature
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')

plot_df = df[(df['pl_rade'].notna()) & (df['pl_eqt'].notna()) & 
             (df['pl_rade'] < 20) & (df['pl_eqt'] < 3000) & (df['pl_eqt'] > 100)]

colors = np.where(plot_df['best_candidates'], '#00ff88',
                  np.where(plot_df['in_hz'], '#ffff00', '#4466aa'))
sizes = np.where(plot_df['best_candidates'], 60, 10)

ax1.scatter(plot_df['pl_eqt'], plot_df['pl_rade'], c=colors, s=sizes, alpha=0.6, edgecolors='none')

# Earth reference
ax1.scatter([255], [1], c='#00ffff', s=150, marker='*', edgecolors='white', zorder=10)
ax1.text(270, 1.3, 'EARTH', color='#00ffff', fontsize=9, fontfamily='monospace')

# Habitable zone
ax1.axvspan(200, 350, alpha=0.1, color='#00ff88')
ax1.axhline(y=2, color='#ffff00', linestyle='--', alpha=0.3)
ax1.axhline(y=0.5, color='#ffff00', linestyle='--', alpha=0.3)

ax1.set_xlabel('EQUILIBRIUM TEMPERATURE (K)', color='#ccccff', fontfamily='monospace')
ax1.set_ylabel('PLANET RADIUS (EARTH RADII)', color='#ccccff', fontfamily='monospace')
ax1.set_title('RADIUS vs TEMPERATURE', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.set_xlim(100, 2500)
ax1.set_ylim(0, 15)
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Plot 2: Earth-sized planets by discovery method
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

earth_sized_df = df[df['earth_sized']]
methods = earth_sized_df['discoverymethod'].value_counts().head(5)

colors = ['#00ff88', '#00ffff', '#ffff00', '#ff8800', '#ff00ff']
bars = ax2.barh(range(len(methods)), methods.values, color=colors)
ax2.set_yticks(range(len(methods)))
ax2.set_yticklabels(methods.index, color='#ccccff', fontfamily='monospace')
ax2.set_xlabel('NUMBER OF EARTH-SIZED PLANETS', color='#ccccff', fontfamily='monospace')
ax2.set_title('EARTH-SIZED DETECTION METHODS', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa', axis='x')
ax2.invert_yaxis()
for spine in ax2.spines.values(): spine.set_color('#3333aa')

for bar, val in zip(bars, methods.values):
    ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(val),
             va='center', color='white', fontsize=10, fontweight='bold')

# Plot 3: Semi-major axis distribution
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

sma = df['pl_orbsmax'].dropna()
sma = sma[(sma > 0) & (sma < 10)]

ax3.hist(sma, bins=50, color='#4466aa', alpha=0.7, label='All Planets')
ax3.hist(df[df['in_hz']]['pl_orbsmax'].dropna(), bins=30, color='#00ff88', alpha=0.8, label='In Habitable Zone')

ax3.axvline(x=1, color='#00ffff', linestyle='--', linewidth=2, label='Earth (1 AU)')

ax3.set_xlabel('SEMI-MAJOR AXIS (AU)', color='#ccccff', fontfamily='monospace')
ax3.set_ylabel('NUMBER OF PLANETS', color='#ccccff', fontfamily='monospace')
ax3.set_title('ORBITAL DISTANCE DISTRIBUTION', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8)
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa')
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Plot 4: Key Findings
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

for i in range(30):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.2, 'HABITABLE ZONE FINDINGS', ha='center', fontsize=16, fontweight='bold', 
         color='#00ff88', fontfamily='monospace')

findings = [
    ('TOTAL CONFIRMED', f'{len(df):,}', '#00ffff'),
    ('IN HABITABLE ZONE', f'{n_in_hz}', '#ffff00'),
    ('EARTH-SIZED', f'{n_earth_sized}', '#ff8800'),
    ('BEST CANDIDATES', f'{n_best}', '#00ff88'),
    ('SMALLEST CANDIDATE', f'{candidates["pl_rade"].min():.2f} R_Earth', '#ff00ff'),
    ('CLOSEST MATCH TO EARTH', candidates.iloc[0]['pl_name'] if len(candidates) > 0 else 'N/A', '#88aaff'),
]

for i, (label, value, color) in enumerate(findings):
    y_pos = 7.5 - i * 1.0
    ax4.text(0.5, y_pos, f'{label}:', fontsize=11, color='#8888aa', fontfamily='monospace', va='center')
    ax4.text(9.5, y_pos, str(value), fontsize=11, color=color, fontweight='bold', fontfamily='monospace', ha='right', va='center')
    ax4.axhline(y=y_pos - 0.5, xmin=0.05, xmax=0.95, color='#333366', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(output_path, dpi=150, facecolor='#050510', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
