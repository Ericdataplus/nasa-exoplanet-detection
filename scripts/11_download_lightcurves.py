"""
Download Kepler Light Curve Time Series Data
For 1D CNN Transit Detection
"""
import os
import kagglehub
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

print("=" * 60)
print("DOWNLOADING KEPLER LIGHT CURVE DATA")
print("=" * 60)

# Download the "Exoplanet Hunting in Deep Space" dataset
# This has labeled Kepler time series with flux values
print("\nDownloading Kepler light curve flux data...")
try:
    path = kagglehub.dataset_download("keplersmachines/kepler-labelled-time-series-data")
    print(f"Downloaded to: {path}")
    
    # Copy files to data directory
    for f in os.listdir(path):
        src = os.path.join(path, f)
        dst = os.path.join(data_dir, f)
        if os.path.isfile(src):
            shutil.copy(src, dst)
            size = os.path.getsize(dst) / 1024 / 1024
            print(f"  Copied: {f} ({size:.2f} MB)")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Data files in data/ directory:")
for f in os.listdir(data_dir):
    size = os.path.getsize(os.path.join(data_dir, f)) / 1024 / 1024
    print(f"  {f}: {size:.2f} MB")
