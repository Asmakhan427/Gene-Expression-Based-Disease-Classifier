# 01_explore_data.py (FIXED VERSION)
import pandas as pd
import numpy as np
import gzip
import os
from pathlib import Path

# Create directories
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)

# 1. Extract and read the file properly
gz_file = "data/raw/GSE45827_series_matrix.txt.gz"
extracted_file = "data/raw/GSE45827_series_matrix.txt"

print("📂 Extracting GZIP file...")
if not os.path.exists(extracted_file):
    with gzip.open(gz_file, 'rt') as f:
        content = f.read()
        with open(extracted_file, 'w') as out:
            out.write(content)
    print(f"✅ Extracted to: {extracted_file}")
else:
    print(f"✅ File already extracted")

# 2. Read the file line by line to find the header
print("📊 Reading expression data...")

# First, find where the data starts (line without '!' at the beginning)
with open(extracted_file, 'r') as f:
    lines = f.readlines()

# Find the header line (starts with 'ID_REF' or doesn't start with '!')
header_line_idx = None
for i, line in enumerate(lines):
    if line.startswith('"ID_REF"') or line.startswith('ID_REF'):
        header_line_idx = i
        break

# If not found, find first line that doesn't start with '!'
if header_line_idx is None:
    for i, line in enumerate(lines):
        if not line.startswith('!') and line.strip():
            header_line_idx = i
            break

print(f"Header found at line {header_line_idx}")

# Parse header - handle quotes properly
header_line = lines[header_line_idx].strip()
# Remove quotes and split by tab
header_parts = [col.strip('"').strip() for col in header_line.split('\t')]
print(f"Found {len(header_parts)} columns")

# Get data lines (skip metadata and header)
data_start = header_line_idx + 1
data_lines = lines[data_start:]

print(f"Processing {len(data_lines)} data rows...")

# Parse data - handle cases where rows might have different lengths
expression_data = []
gene_ids = []

for line in data_lines:
    if not line.strip():
        continue
    parts = line.strip().split('\t')
    # Clean up: remove quotes
    parts = [p.strip('"') for p in parts]
    
    # First column is gene ID, rest are expression values
    if parts:
        gene_ids.append(parts[0])
        expression_data.append(parts[1:])

# Convert to DataFrame
print(f"Creating DataFrame with {len(expression_data)} genes and {len(header_parts[1:])} samples")

# Make sure all rows have the same number of columns
max_cols = max(len(row) for row in expression_data) if expression_data else 0
print(f"Max columns in data: {max_cols}")

# Pad rows if necessary
padded_data = []
for row in expression_data:
    if len(row) < max_cols:
        row = row + [''] * (max_cols - len(row))
    padded_data.append(row[:max_cols])

# Create DataFrame
df = pd.DataFrame(padded_data, columns=header_parts[1:max_cols+1])
df.insert(0, 'ID_REF', gene_ids)

print(f"✅ DataFrame shape: {df.shape}")

# 3. Save as CSV
df.to_csv("data/processed/GSE45827_expression.csv", index=False)
print("✅ Saved expression data to CSV")

# 4. Display basic info
print("\n📋 Dataset Summary:")
print(f"  - Total genes: {df.shape[0]}")
print(f"  - Total samples: {df.shape[1] - 1}")
print(f"  - First few gene IDs: {df['ID_REF'].head().tolist()}")

if df.shape[1] > 5:
    print(f"  - Sample columns: {df.columns[1:5].tolist()}...")
else:
    print(f"  - Sample columns: {df.columns[1:].tolist()}")

# 5. Check for any non-numeric data in expression columns
sample_cols = df.columns[1:]
print("\n🔍 Checking data types...")
sample_data = df[sample_cols].iloc[0, :3]
print(f"First few values: {sample_data.tolist()}")

# Convert to numeric
for col in sample_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for missing values
missing = df[sample_cols].isnull().sum().sum()
print(f"Missing values after conversion: {missing}")

# 6. Print sample information
print("\n🧬 Sample IDs:")
print(df.columns[1:10].tolist())

# 7. Extract metadata (clinical info)
print("\n🔍 Extracting metadata from file...")
metadata = {}
with open(extracted_file, 'r') as f:
    for line in f:
        if line.startswith('!'):
            # Look for sample characteristics
            if 'characteristics_ch1' in line:
                parts = line.strip().split('\t')
                key = parts[0].replace('!', '').strip()
                values = parts[1:] if len(parts) > 1 else []
                metadata[key] = values
            
            # Look for sample titles
            if 'Sample_title' in line:
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    titles = parts[1:]
                    # Clean up
                    titles = [t.strip() for t in titles]
                    metadata['Sample_titles'] = titles

# Save metadata
if metadata:
    print("\n📋 Metadata found:")
    for key, values in metadata.items():
        print(f"  - {key}: {len(values)} entries")
        if len(values) > 0 and len(values) <= 10:
            print(f"    Example: {values[:3]}")
    
    # Try to extract disease state
    if 'Sample_titles' in metadata:
        labels = []
        for title in metadata['Sample_titles']:
            if 'tumor' in title.lower() or 'cancer' in title.lower() or 'Tumor' in title:
                labels.append('Breast Cancer')
            elif 'normal' in title.lower() or 'Normal' in title:
                labels.append('Normal')
            else:
                labels.append('Unknown')
        
        labels_df = pd.DataFrame({
            'sample_id': metadata['Sample_titles'],
            'disease_state': labels
        })
        labels_df.to_csv("data/processed/sample_labels.csv", index=False)
        print("\n✅ Saved sample labels to CSV")
        print(labels_df.head())
        
        # Count classes
        print("\n📊 Class distribution:")
        for cls, count in labels_df['disease_state'].value_counts().items():
            print(f"  - {cls}: {count} samples")

print("\n✅ Exploration complete!")