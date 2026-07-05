#!/usr/bin/env python3
"""
Diagnostic script to find null bytes in all project files.
Run this to identify which file is corrupted.
"""

from pathlib import Path

def check_file_for_null_bytes(filepath):
    """Check a single file for null bytes."""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            if b'\x00' in content:
                # Find positions of null bytes
                null_positions = [i for i, byte in enumerate(content) if byte == 0]
                return True, null_positions
            return False, []
    except Exception as e:
        return None, str(e)

def main():
    """Scan all Python files in project for null bytes."""
    project_root = Path(__file__).parent
    
    # Files to check (in import order)
    files_to_check = [
        'pipeline.py',
        'utils/__init__.py',
        'utils/config_loader.py',
        'utils/logger.py',
        'utils/helpers.py',
        'data/__init__.py',
        'data/acquire_data.py',
        'preprocessing/__init__.py',
        'preprocessing/preprocessor.py',
        'feature_selection/__init__.py',
        'feature_selection/selector.py',
        'ml/__init__.py',
        'ml/models.py',
        'biomarker_analysis/__init__.py',
        'biomarker_analysis/analyzer.py',
        'explainability/__init__.py',
        'explainability/shap_analyzer.py',
        'visualization/visualizer.py',
    ]
    
    print("🔍 Scanning project files for null bytes...\n")
    print("=" * 70)
    
    corrupted_files = []
    
    for filepath in files_to_check:
        full_path = project_root / filepath
        if not full_path.exists():
            print(f"⊘  {filepath:<50} [FILE NOT FOUND]")
            continue
        
        has_nulls, positions = check_file_for_null_bytes(full_path)
        
        if has_nulls is None:
            print(f"❌ {filepath:<50} [ERROR: {positions}]")
        elif has_nulls:
            corrupted_files.append((filepath, positions))
            print(f"🚫 {filepath:<50} [NULL BYTES AT: {positions[:5]}{'...' if len(positions) > 5 else ''}]")
        else:
            print(f"✅ {filepath:<50} [CLEAN]")
    
    print("=" * 70)
    
    if corrupted_files:
        print(f"\n⚠️  FOUND {len(corrupted_files)} CORRUPTED FILE(S):\n")
        for filepath, positions in corrupted_files:
            print(f"  • {filepath}")
            print(f"    Null bytes at positions: {positions[:10]}{'...' if len(positions) > 10 else ''}")
    else:
        print("\n✅ All files are clean! The issue may be elsewhere.")
    
    print("\n📌 Next: Replace the corrupted files with the clean versions provided.\n")

if __name__ == "__main__":
    main()