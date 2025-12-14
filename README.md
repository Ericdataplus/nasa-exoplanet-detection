# 🚀 NASA Exoplanet Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-RTX_3060-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.1%25-00ff88?style=for-the-badge)

### Machine Learning Finds 17 Potentially Habitable Worlds in NASA Kepler Data

**[🌐 Live Dashboard](https://ericdataplus.github.io/nasa-exoplanet-detection/)** · **[📊 Kaggle Dataset](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)** · **[🔭 NASA Archive](https://exoplanetarchive.ipac.caltech.edu/)**

</div>

---

## 🌍 Major Discoveries

<table>
<tr>
<td align="center"><h3>🪐 17</h3><sub>Potentially Habitable Worlds</sub></td>
<td align="center"><h3>⭐ 432</h3><sub>Multi-Planet Systems</sub></td>
<td align="center"><h3>🔥 115</h3><sub>Unusual Planets Found</sub></td>
<td align="center"><h3>🎯 99.1%</h3><sub>Detection Accuracy</sub></td>
</tr>
</table>

---

## 🔬 Key Scientific Findings

### 1. 🌍 **17 Potentially Habitable Planets**
We identified **17 exoplanets** that are:
- **Earth-sized** (0.5-2 Earth radii)
- **In the habitable zone** where liquid water could exist
- Orbiting stars similar to our Sun

### 2. ⭐ **432 Multi-Planet Systems**
- **1,630 unique star systems** analyzed
- Some systems host up to **7 planets**
- **G-type (Sun-like) stars** are the most common hosts

### 3. 🔥 **115 Anomalous Planets**
Using **Isolation Forest** anomaly detection, we found:
- **Hot Jupiters**: Gas giants at 2000K+ temperatures
- **Ultra-short period**: Planets orbiting in less than 1 day
- **Cool Giants**: Gas giants far from their stars
- **Lava Worlds**: Rocky planets hot enough to melt rock

### 4. ❌ **False Positive Breakdown**
Analyzed 5,023 false positives to understand detection challenges:
- **Stellar Eclipse**: 42.9% (eclipsing binary stars)
- **Centroid Offset**: 37.0% (wrong target star)
- **Not Transit-Like**: 35.4% (V-shaped signals)
- **EB Contamination**: 22.8% (nearby binary contamination)

---

## 📊 8 Visualizations

| # | Analysis | Key Insight |
|---|----------|-------------|
| 1 | Data Overview | 9,564 Kepler Objects of Interest |
| 2 | Detection Model | 99.1% accuracy with Gradient Boosting |
| 3 | Neural Network | GPU-trained on RTX 3060, AUC 0.999 |
| 4 | Habitable Zone | **17 potentially habitable worlds** |
| 5 | Stellar Analysis | G-type stars dominate, 432 multi-planet systems |
| 6 | False Positives | Stellar eclipse is #1 cause (42.9%) |
| 7 | Anomaly Detection | **115 unusual planets** identified |
| 8 | Sky Map | 9 years of observations in Cygnus-Lyra |

---

## 🧠 Machine Learning Models

### Gradient Boosting (Best)
```
Accuracy: 99.1%  |  F1: 0.986
```

### Random Forest
```
Accuracy: 99.0%  |  F1: 0.985
```

### Neural Network (PyTorch + CUDA)
```
Accuracy: 98.9%  |  F1: 0.982  |  AUC: 0.999
GPU: NVIDIA RTX 3060 (12GB VRAM)
Architecture: 5-layer MLP | 48,001 parameters
```

---

## 🔬 Top Feature: Disposition Score

The **#1 predictor** of real exoplanets is the **disposition score** (96.4% importance), which combines:
- Transit signal strength
- Star noise characteristics  
- Centroid motion analysis
- Secondary eclipse detection

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Ericdataplus/nasa-exoplanet-detection.git
cd nasa-exoplanet-detection

# Install
pip install pandas numpy matplotlib scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Run all analyses
python scripts/01_data_overview.py
python scripts/02_detection_model.py
python scripts/03_neural_network.py      # Requires GPU
python scripts/04_habitable_zone.py
python scripts/05_stellar_analysis.py
python scripts/06_false_positive_analysis.py
python scripts/07_anomaly_detection.py
python scripts/08_sky_map.py
```

---

## 📁 Project Structure

```
nasa-exoplanet-detection/
├── data/kepler_exoplanets.csv     # 9,564 Kepler Objects
├── graphs/                        # 8 visualizations
├── scripts/                       # Analysis code
│   ├── 01_data_overview.py
│   ├── 02_detection_model.py
│   ├── 03_neural_network.py       # GPU required
│   ├── 04_habitable_zone.py       # Finds 17 habitable worlds
│   ├── 05_stellar_analysis.py
│   ├── 06_false_positive_analysis.py
│   ├── 07_anomaly_detection.py    # Finds 115 unusual planets
│   └── 08_sky_map.py
└── index.html                     # Interactive dashboard
```

---

## 📈 Visualizations

### Habitable Zone Analysis
![Habitable Zone](graphs/04_habitable_zone.png)

### Anomaly Detection
![Anomaly Detection](graphs/07_anomaly_detection.png)

### Neural Network Results
![Neural Network](graphs/03_neural_network.png)

---

## 📚 Data Source

**NASA Kepler Space Telescope** (2009-2018)
- Mission: Find Earth-like planets via transit photometry
- Survey: 116 square degrees in Cygnus-Lyra
- Result: 2,293 confirmed exoplanets

---

## 🔮 Future Work

- [ ] Add TESS mission data (2018-present)
- [ ] Implement 1D CNN on raw light curves
- [ ] Time-series transit detection
- [ ] Habitability scoring model

---

<div align="center">

### If ML can find planets in distant solar systems, imagine what it can do for your data.

**[🌐 View Live Dashboard](https://ericdataplus.github.io/nasa-exoplanet-detection/)**

Made with 🚀 by [Ericdataplus](https://github.com/Ericdataplus) | December 2025

</div>
