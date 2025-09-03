#!/usr/bin/env python3
"""
Lunar Seismic Data Processing Script

This script processes lunar seismic data by applying a low-pass filter and visualizing
the results. It processes all .mseed files in the lunar data directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Stream
from obspy.signal.filter import lowpass
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='obspy')

# Set plot style
try:
    plt.style.use('seaborn-v0_8')  # Try seaborn style if available
    print("Using seaborn style")
except:
    plt.style.use('ggplot')  # Fallback to ggplot style
    print("Using ggplot style")

plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['font.size'] = 12

class SeismicProcessor:
    """Process and visualize seismic data with filtering"""
    
    def __init__(self, data_dir):
        """
        Initialize with the lunar data directory
        
        Args:
            data_dir (str or Path): Path to the lunar data directory
        """
        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.rglob('*.mseed'))
        
        if not self.files:
            raise FileNotFoundError(f"No .mseed files found in {self.data_dir}")
    
    def apply_lowpass_filter(self, trace, freq=1.0, corners=4):
        """
        Apply a low-pass filter to a seismic trace
        
        Args:
            trace: ObsPy Trace object
            freq (float): Filter corner frequency in Hz
            corners (int): Filter corners/order
            
        Returns:
            Filtered trace
        """
        try:
            # Make a copy to avoid modifying the original
            filtered = trace.copy()
            
            # Detrend and taper before filtering
            filtered.detrend('linear')
            filtered.taper(0.05)
            
            # Apply low-pass filter
            filtered.filter('lowpass', 
                          freq=freq, 
                          corners=corners, 
                          zerophase=True)
            return filtered
            
        except Exception as e:
            print(f"Error filtering {trace.id}: {str(e)}")
            return None
    
    def plot_comparison(self, original, filtered, title_suffix=""):
        """
        Plot original vs filtered data
        
        Args:
            original: Original trace
            filtered: Filtered trace
            title_suffix: Additional text for the plot title
        """
        if original is None or filtered is None:
            return
            
        # Create time axis
        times = np.linspace(0, len(original.data) / original.stats.sampling_rate, 
                           len(original.data))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot original data
        ax1.plot(times, original.data, 'b-', linewidth=0.5, label='Original')
        ax1.set_title(f'Original Seismic Data {title_suffix}')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True)
        
        # Plot filtered data
        ax2.plot(times, filtered.data, 'r-', linewidth=0.5, label='Filtered')
        ax2.set_title(f'Low-Pass Filtered (1 Hz) {title_suffix}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def process_files(self, output_dir='filtered_plots', max_files=5):
        """
        Process all .mseed files in the directory
        
        Args:
            output_dir (str): Directory to save output plots
            max_files (int): Maximum number of files to process (for testing)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Found {len(self.files)} .mseed files. Processing up to {max_files} files...")
        
        processed_count = 0
        
        for file_path in tqdm(self.files[:max_files], desc="Processing files"):
            try:
                # Read the seismic data
                st = read(str(file_path))
                
                for trace in st:
                    # Apply low-pass filter
                    filtered = self.apply_lowpass_filter(trace, freq=1.0)
                    
                    if filtered is None:
                        continue
                    
                    # Create plot
                    fig = self.plot_comparison(
                        trace, 
                        filtered,
                        title_suffix=f"\n{file_path.name}"
                    )
                    
                    if fig is not None:
                        # Save the figure
                        output_file = output_dir / f"filtered_{file_path.stem}.png"
                        fig.savefig(output_file, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        
                        processed_count += 1
                        
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
                continue
        
        print(f"\n✅ Processing complete! Processed {processed_count} traces.")
        print(f"Plots saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    # Path to the lunar training data
    LUNAR_DATA_DIR = Path("space_apps_2024_seismic_detection/data/lunar/training/data")
    
    if not LUNAR_DATA_DIR.exists():
        print(f"Error: Lunar data directory not found at {LUNAR_DATA_DIR}")
        exit(1)
    
    try:
        # Initialize the processor
        processor = SeismicProcessor(LUNAR_DATA_DIR)
        
        # Process files (set max_files=None to process all)
        processor.process_files(max_files=5)  # Process first 5 files as example
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        exit(1)
