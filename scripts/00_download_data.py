"""
Download latest exoplanet data from multiple sources:
1. NASA Exoplanet Archive - Confirmed Planets (latest)
2. TESS Objects of Interest from Kaggle
"""
import requests
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

print("=" * 60)
print("DOWNLOADING LATEST EXOPLANET DATA")
print("=" * 60)

# 1. Download from NASA Exoplanet Archive - Confirmed Planets
print("\n1. NASA Exoplanet Archive - Confirmed Planets...")
nasa_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+*+FROM+ps+WHERE+default_flag=1&format=csv"

try:
    response = requests.get(nasa_url, timeout=60)
    response.raise_for_status()
    
    # Save to file
    nasa_path = os.path.join(data_dir, 'nasa_confirmed_planets.csv')
    with open(nasa_path, 'wb') as f:
        f.write(response.content)
    
    # Load and check
    df_nasa = pd.read_csv(nasa_path)
    print(f"   Downloaded: {len(df_nasa):,} confirmed planets")
    print(f"   Saved to: nasa_confirmed_planets.csv")
    print(f"   Columns: {len(df_nasa.columns)}")
    
except Exception as e:
    print(f"   Error: {e}")
    df_nasa = None

# 2. Download TESS Objects of Interest from Kaggle
print("\n2. TESS Objects of Interest from Kaggle...")
try:
    import kagglehub
    tess_path_raw = kagglehub.dataset_download("martinsf2001/kepler-tess-exoplanet-data")
    print(f"   Downloaded from Kaggle: {tess_path_raw}")
    
    # Find and copy the TESS file
    import shutil
    for root, dirs, files in os.walk(tess_path_raw):
        for file in files:
            if 'TOI' in file or 'tess' in file.lower():
                src = os.path.join(root, file)
                dst = os.path.join(data_dir, file)
                shutil.copy(src, dst)
                print(f"   Copied: {file}")
                
                # Load and check
                if file.endswith('.csv'):
                    df_temp = pd.read_csv(dst)
                    print(f"   Records: {len(df_temp):,}")
                    
except Exception as e:
    print(f"   Error with Kaggle: {e}")
    print("   Trying direct NASA TESS data...")
    
    # Fallback: Download TESS from NASA
    tess_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=SELECT+*+FROM+toi&format=csv"
    try:
        response = requests.get(tess_url, timeout=60)
        response.raise_for_status()
        tess_path = os.path.join(data_dir, 'tess_toi.csv')
        with open(tess_path, 'wb') as f:
            f.write(response.content)
        df_tess = pd.read_csv(tess_path)
        print(f"   Downloaded TESS TOI: {len(df_tess):,} objects")
    except Exception as e2:
        print(f"   TESS fallback error: {e2}")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

# Summary
print("\nData files in data/ directory:")
for f in os.listdir(data_dir):
    size = os.path.getsize(os.path.join(data_dir, f)) / 1024 / 1024
    print(f"  {f}: {size:.2f} MB")
