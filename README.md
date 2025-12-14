# 🚀 NASA Exoplanet Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.1%25-00ff88?style=for-the-badge)

### Machine Learning to Detect Exoplanets in NASA Kepler Mission Data

**[🌐 Live Dashboard](https://ericdataplus.github.io/nasa-exoplanet-detection/)** · **[📊 Kaggle Dataset](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)** · **[🔭 NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)**

</div>

---

## 🪐 About This Project

Can machine learning distinguish **real exoplanets** from **false positives** in NASA's Kepler telescope data?

**Yes. With 99.1% accuracy.**

This project analyzes **9,564 Kepler Objects of Interest (KOIs)** — potential planet candidates identified by NASA's Kepler Space Telescope mission. Using a combination of traditional ML and GPU-accelerated deep learning, we classify which signals represent actual exoplanets orbiting distant stars.

---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| **Total Objects Analyzed** | 9,564 |
| **Confirmed Exoplanets** | 2,293 🪐 |
| **False Positives** | 5,023 |
| **Best Model Accuracy** | **99.1%** |
| **Neural Network AUC** | 0.999 |
| **Top Predictor** | Disposition Score (96.4%) |

---

## 🧠 Machine Learning Models

### 1. Random Forest Classifier
```
Accuracy: 99.0%  |  F1: 0.985
```

### 2. Gradient Boosting Classifier  
```
Accuracy: 99.1%  |  F1: 0.986  ⭐ Best Overall
```

### 3. Neural Network (PyTorch + CUDA)
```
Accuracy: 98.9%  |  F1: 0.982  |  AUC: 0.999
Trained on: NVIDIA RTX 3060 (12GB VRAM)
Architecture: 5-layer MLP with BatchNorm & Dropout
Parameters: 48,001
```

---

## 🔬 Key Features Used

| Feature | Description | Importance |
|---------|-------------|------------|
| `koi_score` | Disposition confidence score | **96.4%** |
| `koi_period` | Orbital period (days) | 0.7% |
| `koi_prad` | Planet radius (Earth radii) | 0.5% |
| `koi_teq` | Equilibrium temperature | 0.4% |
| `koi_depth` | Transit depth | 0.4% |
| `koi_duration` | Transit duration | 0.3% |
| `koi_steff` | Star effective temperature | 0.3% |
| `koi_insol` | Insolation flux | 0.3% |

---

## 🖥️ Tech Stack

- **Python 3.12** — Core programming language
- **PyTorch 2.7** — Deep learning framework
- **CUDA 11.8** — GPU acceleration
- **Scikit-learn** — Traditional ML algorithms
- **Pandas & NumPy** — Data processing
- **Matplotlib** — Visualization

---

## 📁 Project Structure

```
nasa-exoplanet-detection/
├── data/
│   └── kepler_exoplanets.csv      # NASA Kepler dataset
├── graphs/
│   ├── 01_data_overview.png       # Dataset visualization
│   ├── 02_detection_model.png     # ML model comparison
│   └── 03_neural_network.png      # Neural network results
├── scripts/
│   ├── 01_data_overview.py        # Data exploration
│   ├── 02_detection_model.py      # Random Forest & XGBoost
│   └── 03_neural_network.py       # PyTorch neural network
└── index.html                      # Interactive dashboard
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Ericdataplus/nasa-exoplanet-detection.git
cd nasa-exoplanet-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Run the analysis
```bash
python scripts/01_data_overview.py
python scripts/02_detection_model.py
python scripts/03_neural_network.py  # Requires NVIDIA GPU
```

---

## 📈 Visualizations

### Data Overview
![Data Overview](graphs/01_data_overview.png)

### ML Model Comparison
![Detection Model](graphs/02_detection_model.png)

### Neural Network (GPU-Trained)
![Neural Network](graphs/03_neural_network.png)

---

## 🌟 What Makes This Project Unique

1. **Real NASA Data** — Not synthetic or toy data, actual Kepler mission results
2. **GPU-Accelerated Deep Learning** — PyTorch neural network trained on RTX 3060
3. **99%+ Accuracy** — Production-grade classification performance
4. **Space-Themed Dashboard** — Animated stars, cosmic colors, immersive design
5. **Scientific Impact** — Understanding what features distinguish real planets

---

## 📚 Data Source

This project uses the **Kepler Exoplanet Search Results** dataset from NASA:

- **Mission**: Kepler Space Telescope (2009-2018)
- **Objective**: Find Earth-like planets in the habitable zone
- **Method**: Transit photometry (detecting dips in starlight)
- **Source**: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

---

## 🔮 Future Improvements

- [ ] Add TESS mission data (2018-present)
- [ ] Implement time-series classification on raw light curves
- [ ] Deploy model as API for real-time classification
- [ ] Add habitability scoring based on planet characteristics

---

## 📜 License

MIT License — Feel free to use this project for learning and portfolio purposes.

---

<div align="center">

### If ML can find planets in distant solar systems, imagine what it can do for your data.

**[🌐 View Live Dashboard](https://ericdataplus.github.io/nasa-exoplanet-detection/)**

Made with 🚀 by [Ericdataplus](https://github.com/Ericdataplus)

</div>
