"""
fix_dataset_final.py - Complete dataset fix
"""

import pandas as pd
import requests
from pathlib import Path
import numpy as np

print("=" * 70)
print("🔧 COMPLETE DATASET FIX - FINAL VERSION")
print("=" * 70)

# Step 1: Delete old files
old_file = Path('data/raw/breast_cancer_gene_expression.csv')
if old_file.exists():
    old_file.unlink()
    print("✅ Deleted old dataset file")

# Step 2: Download the data
print("\n📊 Downloading dataset from UCI...")
try:
    response = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        timeout=30
    )
    response.raise_for_status()
    print("✅ Download successful!")
except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)

# Step 3: Parse the data
print("\n📝 Parsing data...")
lines = response.text.strip().split('\n')
data = []
skipped = 0

for line in lines:
    parts = line.strip().split(',')
    if len(parts) >= 11:
        # Skip rows with '?' (missing values)
        if '?' in parts:
            skipped += 1
            continue
        
        try:
            # Class: 2 = benign (normal), 4 = malignant (cancer)
            class_label = 'normal' if int(parts[10]) == 2 else 'cancer'
            
            sample = {
                'id': int(parts[0]),
                'clump_thickness': float(parts[1]),
                'uniformity_cell_size': float(parts[2]),
                'uniformity_cell_shape': float(parts[3]),
                'marginal_adhesion': float(parts[4]),
                'single_epithelial_cell_size': float(parts[5]),
                'bare_nuclei': float(parts[6]),
                'bland_chromatin': float(parts[7]),
                'normal_nucleoli': float(parts[8]),
                'mitoses': float(parts[9]),
                'class': class_label
            }
            data.append(sample)
        except ValueError as e:
            print(f"⚠️ Skipping malformed row: {parts[:3]}... Error: {e}")
            skipped += 1
            continue

print(f"✅ Parsed {len(data)} samples, skipped {skipped} rows")

# Step 4: Create DataFrame
if len(data) == 0:
    print("❌ ERROR: No data parsed!")
    exit(1)

df = pd.DataFrame(data)
df = df.set_index('id')

# Step 5: Ensure all numeric columns are proper
for col in df.columns:
    if col != 'class':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 6: Remove any rows with NaN values
df = df.dropna()

if len(df) == 0:
    print("❌ ERROR: All data was removed during cleaning!")
    exit(1)

# Step 7: Verify the class column
if 'class' not in df.columns:
    print("❌ ERROR: class column not found!")
    exit(1)

# Step 8: Check class distribution
class_counts = df['class'].value_counts()
print(f"\n📊 Class distribution:")
print(f"   Normal: {class_counts.get('normal', 0)}")
print(f"   Cancer: {class_counts.get('cancer', 0)}")

# Step 9: Save the dataset
raw_dir = Path('data/raw')
raw_dir.mkdir(parents=True, exist_ok=True)
output_file = raw_dir / 'breast_cancer_gene_expression.csv'
df.to_csv(output_file)

print(f"\n✅ Dataset saved to: {output_file}")
print(f"   Samples: {len(df)}")
print(f"   Features: {len(df.columns) - 1}")
print(f"   Classes: {df['class'].unique().tolist()}")
print(f"   Columns: {list(df.columns)}")

# Step 10: Verify the saved file
print("\n📊 Verifying saved file...")
df_verify = pd.read_csv(output_file, index_col=0)
print(f"   Verified: {len(df_verify)} samples")
print(f"   Class column exists: {'class' in df_verify.columns}")

if 'class' in df_verify.columns:
    print(f"   Classes: {df_verify['class'].unique().tolist()}")
    print(f"   Class distribution: {df_verify['class'].value_counts().to_dict()}")

print("\n" + "=" * 70)
print("✅ DATASET FIX COMPLETE!")
print("🚀 Now run: python pipeline.py")
print("=" * 70)
