"""
04 - Habitable Zone Analysis
Identify potentially habitable exoplanets based on stellar and planetary parameters
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
output_path = os.path.join(project_dir, 'graphs', '04_habitable_zone.png')

print("Loading NASA Kepler data...")
df = pd.read_csv(data_path)

# Focus on confirmed planets
confirmed = df[df['koi_disposition'] == 'CONFIRMED'].copy()
print(f"Confirmed exoplanets: {len(confirmed):,}")

# Calculate habitable zone boundaries
# Using conservative habitable zone estimates (Kopparapu et al. 2013)
# Inner edge: 0.99 AU * sqrt(L/L_sun)
# Outer edge: 1.70 AU * sqrt(L/L_sun)

# Estimate stellar luminosity from temperature and radius
# L/L_sun = (R/R_sun)^2 * (T/T_sun)^4
T_sun = 5778  # K
confirmed['luminosity'] = (confirmed['koi_srad'] ** 2) * ((confirmed['koi_steff'] / T_sun) ** 4)

# Calculate habitable zone boundaries in AU
confirmed['hz_inner'] = 0.99 * np.sqrt(confirmed['luminosity'])
confirmed['hz_outer'] = 1.70 * np.sqrt(confirmed['luminosity'])

# Calculate planet's semi-major axis from orbital period (Kepler's 3rd law)
# a^3 = P^2 * M_star (assuming M_star ~ 1 solar mass for simplicity)
confirmed['semi_major_axis'] = (confirmed['koi_period'] / 365.25) ** (2/3)

# Check if planet is in habitable zone
confirmed['in_hz'] = (confirmed['semi_major_axis'] >= confirmed['hz_inner']) & \
                      (confirmed['semi_major_axis'] <= confirmed['hz_outer'])

# Filter for Earth-like size (0.5 to 2 Earth radii)
confirmed['earth_like_size'] = (confirmed['koi_prad'] >= 0.5) & (confirmed['koi_prad'] <= 2.0)

# Potentially habitable: in HZ AND Earth-like size
confirmed['potentially_habitable'] = confirmed['in_hz'] & confirmed['earth_like_size']

# Clean data for plotting
plot_df = confirmed[(confirmed['koi_prad'].notna()) & 
                     (confirmed['koi_teq'].notna()) &
                     (confirmed['koi_prad'] < 25) &
                     (confirmed['koi_teq'] < 3000)].copy()

n_habitable = confirmed['potentially_habitable'].sum()
n_in_hz = confirmed['in_hz'].sum()
n_earth_size = confirmed['earth_like_size'].sum()

print(f"\nHabitable Zone Analysis:")
print(f"  Planets in Habitable Zone: {n_in_hz}")
print(f"  Earth-like Size (0.5-2 R_Earth): {n_earth_size}")
print(f"  POTENTIALLY HABITABLE: {n_habitable}")

# Create SPACE-THEMED visualization
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#050510')

fig.suptitle('HABITABLE ZONE ANALYSIS', fontsize=26, fontweight='bold', 
             color='#00ffff', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.93, 'Searching for Earth-like Planets in the Goldilocks Zone', 
         ha='center', fontsize=12, color='#8888aa', fontfamily='monospace')

# Plot 1: Planet Radius vs Equilibrium Temperature
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')

# Color by habitability
colors = np.where(plot_df['potentially_habitable'], '#00ff88', 
                  np.where(plot_df['in_hz'], '#ffff00', '#4466aa'))

scatter = ax1.scatter(plot_df['koi_teq'], plot_df['koi_prad'], 
                       c=colors, alpha=0.6, s=20, edgecolors='none')

# Add Earth reference
ax1.axhline(y=1, color='#00ffff', linestyle='--', alpha=0.5, linewidth=1)
ax1.axvline(x=255, color='#00ffff', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(260, 1.2, 'EARTH', color='#00ffff', fontsize=9, fontfamily='monospace')

# Habitable temp zone (roughly 200-300K)
ax1.axvspan(200, 300, alpha=0.1, color='#00ff88')

ax1.set_xlabel('EQUILIBRIUM TEMPERATURE (K)', color='#ccccff', fontfamily='monospace')
ax1.set_ylabel('PLANET RADIUS (EARTH RADII)', color='#ccccff', fontfamily='monospace')
ax1.set_title('RADIUS vs TEMPERATURE', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.set_xlim(0, 2500)
ax1.set_ylim(0, 20)
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Plot 2: Orbital Period Distribution
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

# Split by habitability
hz_periods = confirmed[confirmed['in_hz']]['koi_period'].dropna()
non_hz_periods = confirmed[~confirmed['in_hz']]['koi_period'].dropna()

ax2.hist(non_hz_periods[non_hz_periods < 400], bins=50, alpha=0.6, color='#4466aa', label='Outside HZ')
ax2.hist(hz_periods[hz_periods < 400], bins=50, alpha=0.8, color='#00ff88', label='In Habitable Zone')

ax2.axvline(x=365.25, color='#00ffff', linestyle='--', linewidth=2, label='Earth (365 days)')

ax2.set_xlabel('ORBITAL PERIOD (DAYS)', color='#ccccff', fontfamily='monospace')
ax2.set_ylabel('NUMBER OF PLANETS', color='#ccccff', fontfamily='monospace')
ax2.set_title('ORBITAL PERIOD DISTRIBUTION', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8)
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa')
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Plot 3: Size Distribution
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

radii = confirmed['koi_prad'].dropna()
radii = radii[radii < 25]

# Categorize by size
bins = [0, 1, 2, 4, 10, 25]
labels = ['Sub-Earth\n(<1 R)', 'Earth-like\n(1-2 R)', 'Super-Earth\n(2-4 R)', 'Neptune-like\n(4-10 R)', 'Giant\n(>10 R)']
size_counts = pd.cut(radii, bins=bins).value_counts().sort_index()

colors = ['#00ffff', '#00ff88', '#ffff00', '#ff8800', '#ff4466']
bars = ax3.bar(range(len(labels)), size_counts.values, color=colors)

ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, color='#ccccff', fontsize=9, fontfamily='monospace')
ax3.set_ylabel('NUMBER OF PLANETS', color='#ccccff', fontfamily='monospace')
ax3.set_title('PLANET SIZE CLASSIFICATION', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa', axis='y')
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Add value labels
for bar, val in zip(bars, size_counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(val),
             ha='center', color='white', fontsize=10, fontweight='bold', fontfamily='monospace')

# Plot 4: Key Discoveries
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Add decorative stars
for i in range(30):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.2, 'KEY DISCOVERIES', ha='center', fontsize=16, fontweight='bold', 
         color='#00ffff', fontfamily='monospace')

discoveries = [
    ('CONFIRMED EXOPLANETS', f'{len(confirmed):,}', '#00ffff'),
    ('IN HABITABLE ZONE', f'{n_in_hz}', '#00ff88'),
    ('EARTH-LIKE SIZE', f'{n_earth_size}', '#ffff00'),
    ('POTENTIALLY HABITABLE', f'{n_habitable}', '#ff00ff'),
    ('AVG ORBITAL PERIOD', f'{confirmed["koi_period"].median():.1f} days', '#ff8800'),
    ('AVG PLANET RADIUS', f'{confirmed["koi_prad"].median():.1f} R_Earth', '#88aaff'),
]

for i, (label, value, color) in enumerate(discoveries):
    y_pos = 7.8 - i * 1.1
    ax4.text(0.5, y_pos, f'{label}:', fontsize=11, color='#8888aa', fontfamily='monospace', va='center')
    ax4.text(9.5, y_pos, value, fontsize=11, color=color, fontweight='bold', fontfamily='monospace', ha='right', va='center')
    ax4.axhline(y=y_pos - 0.5, xmin=0.05, xmax=0.95, color='#333366', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(output_path, dpi=150, facecolor='#050510', bbox_inches='tight')
plt.close()

print(f"\nSaved: {output_path}")
