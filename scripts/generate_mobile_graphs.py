"""
Mobile-Optimized Graphs Generator for NASA Exoplanet Detection
"""
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_dir = os.path.join(project_dir, 'graphs_mobile')
os.makedirs(output_dir, exist_ok=True)

MOBILE_CONFIG = {
    'figsize': (6, 8),
    'bg_color': '#0d1117',
    'text_color': '#ffffff',
}

def setup_style():
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.facecolor'] = MOBILE_CONFIG['bg_color']
    plt.rcParams['axes.facecolor'] = MOBILE_CONFIG['bg_color']
    plt.rcParams['text.color'] = MOBILE_CONFIG['text_color']
    plt.rcParams['axes.labelcolor'] = MOBILE_CONFIG['text_color']
    plt.rcParams['xtick.color'] = MOBILE_CONFIG['text_color']
    plt.rcParams['ytick.color'] = MOBILE_CONFIG['text_color']

def style_axes(ax):
    ax.set_facecolor(MOBILE_CONFIG['bg_color'])
    for spine in ax.spines.values():
        spine.set_color('#30363d')

def generate_stats():
    print("📱 Generating: Key Stats")
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    ax.text(0.5, 0.92, 'Exoplanet Detection', fontsize=22, fontweight='bold',
            ha='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.85, 'NASA Kepler Data', fontsize=16,
            ha='center', color='#8b949e', transform=ax.transAxes)
    
    stats = [
        ('5,578', 'Confirmed Planets', '#56d364'),
        ('10,052', 'Stars Analyzed', '#4facfe'),
        ('99.4%', 'Model Accuracy', '#feca57'),
        ('Transit', 'Best Method', '#a371f7'),
    ]
    
    for i, (value, label, color) in enumerate(stats):
        y = 0.68 - i * 0.17
        ax.text(0.5, y, value, fontsize=40, fontweight='bold',
                ha='center', color=color, transform=ax.transAxes)
        ax.text(0.5, y - 0.05, label, fontsize=14,
                ha='center', color='#8b949e', transform=ax.transAxes)
    
    plt.savefig(os.path.join(output_dir, '01_stats.png'), dpi=200,
                facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 01_stats.png")

def generate_methods():
    print("📱 Generating: Detection Methods")
    methods = ['Transit', 'Radial Vel.', 'Imaging', 'Microlensing', 'Other']
    counts = [4472, 1068, 72, 206, 360]
    colors = ['#56d364', '#4facfe', '#ff6b6b', '#feca57', '#a371f7']
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, counts, color=colors, height=0.6)
    
    for bar, count in zip(bars, counts):
        ax.text(count + 50, bar.get_y() + bar.get_height()/2, f'{count:,}',
                va='center', ha='left', color='white', fontsize=14, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel('Planets Found', fontsize=14)
    ax.set_title('Detection Methods', fontsize=20, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(output_dir, '02_methods.png'), dpi=200,
                facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 02_methods.png")

def generate_ml():
    print("📱 Generating: ML Model")
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    ax.text(0.5, 0.85, 'Classification', fontsize=24, fontweight='bold',
            ha='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.58, '99.4%', fontsize=72, fontweight='bold',
            ha='center', color='#56d364', transform=ax.transAxes)
    ax.text(0.5, 0.42, 'Accuracy', fontsize=20,
            ha='center', color='#8b949e', transform=ax.transAxes)
    ax.text(0.5, 0.26, 'Random Forest', fontsize=24, fontweight='bold',
            ha='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.16, 'vs XGBoost, CNN, SVM', fontsize=14,
            ha='center', color='#58a6ff', transform=ax.transAxes)
    
    plt.savefig(os.path.join(output_dir, '03_ml.png'), dpi=200,
                facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 03_ml.png")

def generate_types():
    print("📱 Generating: Planet Types")
    types = ['Hot Jupiter', 'Super-Earth', 'Gas Giant', 'Terrestrial']
    counts = [1521, 1876, 1423, 758]
    colors = ['#ff6b6b', '#56d364', '#4facfe', '#feca57']
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    
    wedges, texts, autotexts = ax.pie(counts, labels=types, autopct='%1.0f%%',
                                       colors=colors, textprops={'color': 'white', 'fontsize': 11})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax.set_title('Planet Types', fontsize=20, fontweight='bold', pad=20, color='white')
    
    plt.savefig(os.path.join(output_dir, '04_types.png'), dpi=200,
                facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 04_types.png")

def generate_habitable():
    print("📱 Generating: Habitable Zone")
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    ax.text(0.5, 0.85, 'Habitable Zone', fontsize=24, fontweight='bold',
            ha='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.58, '59', fontsize=96, fontweight='bold',
            ha='center', color='#56d364', transform=ax.transAxes)
    ax.text(0.5, 0.40, 'Potentially Habitable', fontsize=18,
            ha='center', color='#8b949e', transform=ax.transAxes)
    ax.text(0.5, 0.25, 'TRAPPIST-1', fontsize=24, fontweight='bold',
            ha='center', color='#feca57', transform=ax.transAxes)
    ax.text(0.5, 0.15, '7 planets • 3 in zone', fontsize=14,
            ha='center', color='#8b949e', transform=ax.transAxes)
    
    plt.savefig(os.path.join(output_dir, '05_habitable.png'), dpi=200,
                facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 05_habitable.png")

def generate_insights():
    print("📱 Generating: Insights")
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Key Findings', fontsize=20, fontweight='bold',
            ha='center', color='white', transform=ax.transAxes)
    
    insights = [
        ('1', 'Transit = 74%', 'Most effective method', '#56d364'),
        ('2', '99.4% Accuracy', 'RF classifier', '#4facfe'),
        ('3', '59 Habitable', 'Goldilocks zone', '#feca57'),
        ('4', 'TRAPPIST-1', '7 planet system', '#a371f7'),
    ]
    
    for i, (num, headline, subtext, color) in enumerate(insights):
        y = 0.78 - i * 0.20
        circle = plt.Circle((0.12, y), 0.05, transform=ax.transAxes,
                           facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.12, y, num, fontsize=18, fontweight='bold', color='white',
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.22, y + 0.02, headline, fontsize=18, fontweight='bold',
                color='white', transform=ax.transAxes)
        ax.text(0.22, y - 0.04, subtext, fontsize=14,
                color='#8b949e', transform=ax.transAxes)
    
    plt.savefig(os.path.join(output_dir, '06_insights.png'), dpi=200,
                facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 06_insights.png")

if __name__ == '__main__':
    print("\n📱 Generating Mobile Graphs (Exoplanets)")
    print("=" * 50)
    setup_style()
    generate_stats()
    generate_methods()
    generate_ml()
    generate_types()
    generate_habitable()
    generate_insights()
    print(f"\n✅ All mobile graphs saved to: {output_dir}")
