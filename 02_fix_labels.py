# 02_fix_labels.py
import pandas as pd
import gzip
import re

print("🔍 Extracting accurate disease labels...")

gz_file = "data/raw/GSE45827_series_matrix.txt.gz"

# Extract metadata with disease information
sample_titles = []
sample_characteristics = []

with gzip.open(gz_file, 'rt') as f:
    for line in f:
        if line.startswith('!Sample_title'):
            parts = line.strip().split('\t')
            sample_titles = [p.strip('"') for p in parts[1:]]
        if line.startswith('!Sample_characteristics_ch1'):
            parts = line.strip().split('\t')
            sample_characteristics = [p.strip('"') for p in parts[1:]]

print(f"Found {len(sample_titles)} samples")
print(f"Found {len(sample_characteristics)} characteristic entries")

# Create labels DataFrame
labels_data = []

for i, title in enumerate(sample_titles):
    # Extract disease type from title
    disease = "Unknown"
    
    # Check title
    title_lower = title.lower()
    if 'basal' in title_lower:
        disease = 'Basal-like Breast Cancer'
    elif 'her2' in title_lower:
        disease = 'HER2+ Breast Cancer'
    elif 'luminal' in title_lower:
        disease = 'Luminal Breast Cancer'
    elif 'normal' in title_lower:
        disease = 'Normal'
    elif 'cell line' in title_lower:
        disease = 'Cell Line'
    
    # If still unknown, check characteristics
    if disease == "Unknown" and i < len(sample_characteristics):
        char = sample_characteristics[i].lower()
        if 'basal' in char:
            disease = 'Basal-like Breast Cancer'
        elif 'her2' in char:
            disease = 'HER2+ Breast Cancer'
        elif 'luminal' in char:
            disease = 'Luminal Breast Cancer'
        elif 'normal' in char:
            disease = 'Normal'
    
    labels_data.append({
        'sample_id': title,
        'disease_type': disease,
        'characteristics': sample_characteristics[i] if i < len(sample_characteristics) else ''
    })

labels_df = pd.DataFrame(labels_data)

# Clean up sample IDs
labels_df['sample_id'] = labels_df['sample_id'].str.replace('"', '').str.strip()

# Display distribution
print("\n📊 Disease Type Distribution:")
dist = labels_df['disease_type'].value_counts()
for disease, count in dist.items():
    print(f"  - {disease}: {count} samples ({count/len(labels_df)*100:.1f}%)")

# Save labels
labels_df.to_csv("data/processed/sample_labels_clean.csv", index=False)
print("\n✅ Saved clean labels to data/processed/sample_labels_clean.csv")

# Show sample
print("\n📋 First 10 samples:")
print(labels_df.head(10))

# Create binary classification (Cancer vs Normal)
labels_df['is_cancer'] = labels_df['disease_type'].apply(
    lambda x: 1 if x != 'Normal' and x != 'Cell Line' else 0
)

# Save binary labels
labels_df[['sample_id', 'is_cancer', 'disease_type']].to_csv(
    "data/processed/sample_labels_binary.csv", index=False
)

print("\n✅ Binary labels saved")
print(f"  - Cancer samples: {labels_df['is_cancer'].sum()}")
print(f"  - Normal samples: {len(labels_df) - labels_df['is_cancer'].sum()}")