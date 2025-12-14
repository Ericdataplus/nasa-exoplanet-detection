"""
06 - False Positive Analysis
Understanding what makes false positives - eclipsing binaries, background stars, etc.
This is cutting-edge analysis that astronomers care about!
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
output_path = os.path.join(project_dir, 'graphs', '06_false_positive_analysis.png')

print("Loading NASA Kepler data...")
df = pd.read_csv(data_path)

# Focus on false positives with flag data
fp = df[df['koi_disposition'] == 'FALSE POSITIVE'].copy()
confirmed = df[df['koi_disposition'] == 'CONFIRMED'].copy()

print(f"False Positives: {len(fp):,}")
print(f"Confirmed: {len(confirmed):,}")

# Analyze false positive flags
# koi_fpflag_nt: Not Transit-Like - centroid offset, V-shaped transit
# koi_fpflag_ss: Stellar Eclipse - secondary eclipse detected  
# koi_fpflag_co: Centroid Offset - transit not on target star
# koi_fpflag_ec: Ephemeris Match - contamination from known eclipsing binary

fp_flags = {
    'Not Transit-Like': fp['koi_fpflag_nt'].sum(),
    'Stellar Eclipse': fp['koi_fpflag_ss'].sum(),
    'Centroid Offset': fp['koi_fpflag_co'].sum(),
    'Eclipsing Binary\nContamination': fp['koi_fpflag_ec'].sum(),
}

# Multiple flags
fp['n_flags'] = fp[['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']].sum(axis=1)
flag_counts = fp['n_flags'].value_counts().sort_index()

# Compare transit properties between confirmed and FP
print("\nFalse Positive Breakdown:")
for flag, count in fp_flags.items():
    pct = count / len(fp) * 100
    print(f"  {flag.replace(chr(10), ' ')}: {count} ({pct:.1f}%)")

# Create SPACE-THEMED visualization
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#050510')

fig.suptitle('FALSE POSITIVE ANALYSIS', fontsize=26, fontweight='bold', 
             color='#ff4466', y=0.97, fontfamily='monospace')
fig.text(0.5, 0.93, 'Understanding What Causes False Exoplanet Detections', 
         ha='center', fontsize=12, color='#8888aa', fontfamily='monospace')

# Plot 1: False Positive Causes
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#0a0a1a')

colors = ['#ff4466', '#ff8844', '#ffcc44', '#44ccff']
bars = ax1.barh(list(fp_flags.keys()), list(fp_flags.values()), color=colors)

ax1.set_xlabel('NUMBER OF FALSE POSITIVES', color='#ccccff', fontfamily='monospace')
ax1.set_title('CAUSES OF FALSE POSITIVES', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax1.tick_params(colors='#888899')
ax1.grid(True, alpha=0.1, color='#4444aa', axis='x')
for spine in ax1.spines.values(): spine.set_color('#3333aa')

# Add percentage labels
for bar, val in zip(bars, fp_flags.values()):
    pct = val / len(fp) * 100
    ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, f'{val:,} ({pct:.0f}%)',
             va='center', color='white', fontsize=9, fontweight='bold', fontfamily='monospace')

# Plot 2: Transit Depth Comparison
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#0a0a1a')

# Clean data
fp_depth = fp['koi_depth'].dropna()
fp_depth = fp_depth[(fp_depth > 0) & (fp_depth < 50000)]
conf_depth = confirmed['koi_depth'].dropna()
conf_depth = conf_depth[(conf_depth > 0) & (conf_depth < 50000)]

ax2.hist(conf_depth, bins=50, alpha=0.7, color='#00ff88', label='Confirmed Planets', density=True)
ax2.hist(fp_depth, bins=50, alpha=0.5, color='#ff4466', label='False Positives', density=True)

ax2.set_xlabel('TRANSIT DEPTH (ppm)', color='#ccccff', fontfamily='monospace')
ax2.set_ylabel('DENSITY', color='#ccccff', fontfamily='monospace')
ax2.set_title('TRANSIT DEPTH: PLANETS vs FALSE POSITIVES', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax2.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8)
ax2.tick_params(colors='#888899')
ax2.grid(True, alpha=0.1, color='#4444aa')
for spine in ax2.spines.values(): spine.set_color('#3333aa')

# Plot 3: Disposition Score Comparison
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#0a0a1a')

fp_score = fp['koi_score'].dropna()
conf_score = confirmed['koi_score'].dropna()

ax3.hist(conf_score, bins=30, alpha=0.7, color='#00ff88', label='Confirmed Planets')
ax3.hist(fp_score, bins=30, alpha=0.5, color='#ff4466', label='False Positives')

ax3.axvline(x=0.5, color='#ffff00', linestyle='--', linewidth=2, label='Decision Boundary')

ax3.set_xlabel('DISPOSITION SCORE', color='#ccccff', fontfamily='monospace')
ax3.set_ylabel('COUNT', color='#ccccff', fontfamily='monospace')
ax3.set_title('DISPOSITION SCORE DISTRIBUTION', color='#e0e0ff', fontsize=14, fontweight='bold', fontfamily='monospace')
ax3.legend(facecolor='#1a1a2e', labelcolor='#ccccff', framealpha=0.8)
ax3.tick_params(colors='#888899')
ax3.grid(True, alpha=0.1, color='#4444aa')
for spine in ax3.spines.values(): spine.set_color('#3333aa')

# Plot 4: Key Insights
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#0a0a1a')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Stars
for i in range(30):
    x, y = np.random.rand(2) * 10
    ax4.scatter(x, y, s=np.random.rand() * 3 + 0.5, c='white', alpha=np.random.rand() * 0.5 + 0.2)

ax4.text(5, 9.2, 'FALSE POSITIVE INSIGHTS', ha='center', fontsize=16, fontweight='bold', 
         color='#ff4466', fontfamily='monospace')

# Key insight: most common cause
most_common = max(fp_flags, key=fp_flags.get).replace('\n', ' ')

insights = [
    ('TOTAL FALSE POSITIVES', f'{len(fp):,}', '#ff4466'),
    ('#1 CAUSE', most_common, '#ff8844'),
    ('FP RATE', f'{len(fp)/(len(fp)+len(confirmed))*100:.1f}%', '#ffcc44'),
    ('AVG FP SCORE', f'{fp_score.mean():.3f}', '#44ccff'),
    ('AVG PLANET SCORE', f'{conf_score.mean():.3f}', '#00ff88'),
    ('SCORE SEPARATION', f'{conf_score.mean() - fp_score.mean():.3f}', '#ff00ff'),
    ('MULTI-FLAG FPs', f'{(fp["n_flags"] > 1).sum():,}', '#88aaff'),
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
