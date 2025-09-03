import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from pathlib import Path

def process_mseed_file(file_path, output_dir):
    """Process a single mseed file and create FFT plot"""
    try:
        # Read the mseed file
        st = read(file_path)
        
        if not st:
            print(f"No data in {file_path}")
            return
            
        # Get the trace
        tr = st[0]
        
        # Get sampling rate and number of points
        fs = tr.stats.sampling_rate
        n = len(tr.data)
        
        # Perform FFT
        yf = np.fft.fft(tr.data)
        xf = np.fft.fftfreq(n, 1/fs)
        
        # Take only positive frequencies
        idx = np.where(xf > 0)
        xf = xf[idx]
        yf = np.abs(yf[idx])
        
        # Create output directory if it doesn't exist
        rel_path = os.path.relpath(os.path.dirname(file_path), 'data')
        save_dir = os.path.join(output_dir, rel_path, 'fft_plots')
        os.makedirs(save_dir, exist_ok=True)
        
        # Create and save the plot
        plt.figure(figsize=(12, 6))
        plt.plot(xf, 20 * np.log10(yf))  # Convert to dB
        plt.title(f'FFT of {os.path.basename(file_path)}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True)
        
        # Save the plot
        output_file = os.path.join(save_dir, f"fft_{os.path.basename(file_path)}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Define directories
    data_dir = os.path.join('space_apps_2024_seismic_detection', 'data')
    output_dir = 'analysis_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all mseed files
    mseed_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mseed'):
                mseed_files.append(os.path.join(root, file))
    
    print(f"Found {len(mseed_files)} mseed files to process")
    
    # Process each file
    for i, file_path in enumerate(mseed_files, 1):
        print(f"\nProcessing file {i}/{len(mseed_files)}")
        process_mseed_file(file_path, output_dir)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()