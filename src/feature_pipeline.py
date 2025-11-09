"""
Feature Pipeline - Backward Compatibility Wrapper
==================================================
This script now serves as a wrapper that calls update_feature_store.py
to maintain backward compatibility with existing automation.

For new usage:
  - Run once: python src/build_feature_store.py
  - Run hourly: python src/update_feature_store.py

This wrapper will attempt to update if features exist, otherwise build from scratch.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main function - calls appropriate script based on feature store existence."""
    FEATURE_STORE_PATH = "data/processed/features.parquet"
    
    if os.path.exists(FEATURE_STORE_PATH):
        print("Feature store exists - running UPDATE mode...")
        print("-" * 60)
        from src.update_feature_store import main as update_main
        update_main()
    else:
        print("Feature store not found - running BUILD mode...")
        print("-" * 60)
        from src.build_feature_store import main as build_main
        build_main()

if __name__ == "__main__":
    main()
