#!/usr/bin/env python3
"""
Data Structure Verification Script

This script verifies the presence of all required seismic data files and directories
in the expected locations. It checks for both Lunar and Mars seismic data files.
"""

from pathlib import Path
import os

class SeismicDataPaths:
    """Manage all seismic dataset paths"""
    
    def __init__(self, base_path):
        """
        Initialize with base path to the data directory
        
        Args:
            base_path (str or Path): Path to the root data directory
        """
        self.base_path = Path(base_path)
        
        # Lunar paths
        self.lunar_train_data = self.base_path / 'lunar' / 'training' / 'data'
        self.lunar_test_data = self.base_path / 'lunar' / 'test' / 'data'
        
        # Mars paths
        self.mars_train_data = self.base_path / 'mars' / 'training' / 'data'
        self.mars_test_data = self.base_path / 'mars' / 'test' / 'data'
    
    def get_lunar_training_files(self):
        """Get all lunar training .mseed files"""
        mseed_files = list(self.lunar_train_data.rglob('*.mseed'))
        print(f"Found {len(mseed_files)} lunar training .mseed files")
        return mseed_files
    
    def get_mars_training_files(self):
        """Get all Mars training .mseed files"""
        mseed_files = list(self.mars_train_data.rglob('*.mseed'))
        print(f"Found {len(mseed_files)} Mars training .mseed files")
        return mseed_files
    
    def get_lunar_test_files(self):
        """Get all lunar test .mseed files"""
        mseed_files = list(self.lunar_test_data.rglob('*.mseed'))
        print(f"Found {len(mseed_files)} lunar test .mseed files")
        return mseed_files
    
    def get_mars_test_files(self):
        """Get all Mars test .mseed files"""
        mseed_files = list(self.mars_test_data.rglob('*.mseed'))
        print(f"Found {len(mseed_files)} Mars test .mseed files")
        return mseed_files

def verify_data_structure(base_path):
    """
    Verify the seismic data directory structure and file presence
    
    Args:
        base_path (str or Path): Path to the root data directory
        
    Returns:
        bool: True if all required directories exist, False otherwise
    """
    print("🔍 Verifying Seismic Data Structure...\n")
    
    # Initialize paths
    paths = SeismicDataPaths(base_path)
    
    # Check directory existence
    checks = [
        ("Lunar training data", paths.lunar_train_data),
        ("Lunar test data", paths.lunar_test_data),
        ("Mars training data", paths.mars_train_data),
        ("Mars test data", paths.mars_test_data)
    ]
    
    all_dirs_exist = True
    for name, path in checks:
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_dirs_exist = False
    
    if not all_dirs_exist:
        print("\n❌ Some required directories are missing.")
        return False
    
    # Check file counts
    print("\n📊 File Counts:")
    lunar_train_count = len(paths.get_lunar_training_files())
    lunar_test_count = len(paths.get_lunar_test_files())
    mars_train_count = len(paths.get_mars_training_files())
    mars_test_count = len(paths.get_mars_test_files())
    
    # Check if any directory is empty
    if lunar_train_count == 0 or lunar_test_count == 0 or \
       mars_train_count == 0 or mars_test_count == 0:
        print("\n⚠  Warning: Some directories are empty!")
        return False
    
    print("\n✅ Data structure verification complete!")
    return True

if __name__ == "__main__":
    # Look for data in common locations
    possible_data_dirs = [
        "space_apps_2024_seismic_detection/data",
        "data",
        "space_apps_2024_seismic_detection"
    ]
    
    # Find the first existing data directory
    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break
    
    if data_dir is None:
        print("❌ Error: Could not find data directory. Please check the following paths:")
        for dir_path in possible_data_dirs:
            print(f"- {os.path.abspath(dir_path)}")
        exit(1)
    
    # Verify the data structure
    success = verify_data_structure(data_dir)
    
    if not success:
        print("\n❌ Data structure verification failed. Please check the paths and try again.")
        exit(1)
    
    print("\n✨ All checks passed! You're ready to analyze the seismic data.")
