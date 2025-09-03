import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read, Stream
from scipy import signal
from scipy.signal import coherence, detrend, welch, butter, filtfilt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def preprocess_signal(data, fs, lowcut=0.1, highcut=5.0, order=4):
    """Preprocess the signal by demeaning, detrending, and bandpass filtering."""
    # Convert to numpy array if it's not already
    data = np.asarray(data, dtype=np.float64)
    
    # 1. Demean the signal
    data = data - np.mean(data)
    
    # 2. Detrend the signal
    data = detrend(data)
    
    # 3. Bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    data = filtfilt(b, a, data)
    
    return data

def window_data(data, window_length, fs, overlap=0.5):
    """Split data into overlapping windows."""
    window_size = int(window_length * fs)
    step = int(window_size * (1 - overlap))
    
    # Ensure data is 1D
    data = np.asarray(data).flatten()
    
    # Pad the data if needed
    if len(data) < window_size:
        pad_size = window_size - len(data)
        data = np.pad(data, (0, pad_size), 'constant')
    
    # Create windows
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)

def load_seismic_data(directory, window_length=20, overlap=0.5, lowcut=0.1, highcut=5.0):
    """Load and preprocess seismic data from the specified directory."""
    data = {}
    for file in os.listdir(directory):
        if file.endswith('.mseed'):
            file_path = os.path.join(directory, file)
            try:
                st = read(file_path)
                if len(st) > 0:
                    # Use the first trace for simplicity
                    tr = st[0]
                    fs = tr.stats.sampling_rate
                    
                    # Preprocess the signal
                    processed_data = preprocess_signal(tr.data, fs, lowcut, highcut)
                    
                    # Window the data
                    windows = window_data(processed_data, window_length, fs, overlap)
                    
                    data[file] = {
                        'data': processed_data,
                        'windows': windows,
                        'stats': tr.stats,
                        'sampling_rate': fs,
                        'npts': tr.stats.npts,
                        'starttime': tr.stats.starttime.datetime,
                        'window_length': window_length,
                        'overlap': overlap,
                        'filter_band': (lowcut, highcut)
                    }
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    return data

def calculate_coherence(data_dict, file1, file2, nperseg=1024):
    """Calculate coherence between two seismic traces across all windows."""
    if file1 not in data_dict or file2 not in data_dict:
        print("One or both files not found in the data dictionary.")
        return None, None, None
    
    windows1 = data_dict[file1]['windows']
    windows2 = data_dict[file2]['windows']
    fs = data_dict[file1]['sampling_rate']
    
    # Use minimum number of windows between the two signals
    num_windows = min(len(windows1), len(windows2))
    if num_windows == 0:
        print("No valid windows found for coherence calculation.")
        return None, None, None
    
    # Calculate coherence for each window and average
    all_f = []
    all_Cxy = []
    
    for i in range(num_windows):
        # Make sure both windows have the same length
        min_len = min(len(windows1[i]), len(windows2[i]))
        if min_len < 2:  # Need at least 2 points for coherence
            continue
            
        data1 = windows1[i][:min_len]
        data2 = windows2[i][:min_len]
        
        # Calculate coherence for this window
        f, Cxy = coherence(data1, data2, fs=fs, nperseg=min(nperseg, min_len//2))
        
        all_f.append(f)
        all_Cxy.append(Cxy)
    
    if not all_f:
        print("No valid windows for coherence calculation.")
        return None, None, None
    
    # Average coherence across all windows
    f_avg = np.mean(np.array(all_f), axis=0)
    Cxy_avg = np.mean(np.array(all_Cxy), axis=0)
    
    # Calculate total duration used for analysis
    duration = (num_windows * data_dict[file1]['window_length'] * 
               (1 - data_dict[file1]['overlap']))
    
    return f_avg, Cxy_avg, duration

def plot_coherence(f, Cxy, file1, file2, duration, data_dict):
    """Plot the coherence spectrum and time series."""
    plt.figure(figsize=(14, 10))
    
    # Plot coherence spectrum
    plt.subplot(3, 1, 1)
    plt.semilogx(f, Cxy, 'b-', linewidth=1.5)
    plt.title(f'Coherence Spectrum between\n{os.path.basename(file1)}\nand\n{os.path.basename(file2)}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence (0-1)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Plot time series of first window
    plt.subplot(3, 1, 2)
    window_length = data_dict[file1]['window_length']
    time = np.linspace(0, window_length, len(data_dict[file1]['windows'][0]))
    plt.plot(time, data_dict[file1]['windows'][0], 'b-', label=os.path.basename(file1), alpha=0.8, linewidth=0.8)
    plt.plot(time, data_dict[file2]['windows'][0], 'r-', label=os.path.basename(file2), alpha=0.8, linewidth=0.8)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'First {window_length}s Window (Preprocessed)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot spectrogram of first trace for reference
    plt.subplot(3, 1, 3)
    fs = data_dict[file1]['sampling_rate']
    f, t, Sxx = signal.spectrogram(data_dict[file1]['windows'][0], fs=fs, 
                                  nperseg=min(256, len(data_dict[file1]['windows'][0])//4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.ylim([0, 10])  # Limit to 10Hz for better visibility of seismic band
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram of First Trace (First Window)')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    config = {
        'data_dir': r"C:\\Users\\91995\\OneDrive\\Desktop\\Nasa Space App hackathon\\space_apps_2024_seismic_detection\\data\\mars\\training\\data",
        'window_length': 20.0,  # seconds
        'overlap': 0.5,  # 50% overlap between windows
        'lowcut': 0.1,  # Hz - lower frequency bound
        'highcut': 5.0,  # Hz - upper frequency bound
    }
    
    print("Loading and preprocessing seismic data...")
    print(f"Processing parameters:")
    print(f"- Window length: {config['window_length']}s")
    print(f"- Window overlap: {config['overlap']*100}%")
    print(f"- Bandpass filter: {config['lowcut']}-{config['highcut']} Hz")
    
    # Load and preprocess the data
    global data_dict
    data_dict = load_seismic_data(
        config['data_dir'],
        window_length=config['window_length'],
        overlap=config['overlap'],
        lowcut=config['lowcut'],
        highcut=config['highcut']
    )
    
    if len(data_dict) < 2:
        print("Need at least two seismic traces for coherence analysis.")
        return
    
    # Get the list of loaded files
    files = list(data_dict.keys())
    print(f"\nLoaded {len(files)} seismic traces.")
    
    # Perform coherence analysis between the first two traces
    file1, file2 = files[0], files[1]
    print(f"\nAnalyzing coherence between:")
    print(f"1. {file1}")
    print(f"2. {file2}")
    
    # Calculate coherence
    print("\nCalculating coherence across windows...")
    f, Cxy, duration = calculate_coherence(data_dict, file1, file2)
    
    if f is not None and Cxy is not None:
        # Plot the results
        plot_coherence(f, Cxy, file1, file2, duration, data_dict)
        
        # Print some statistics
        print("\nCoherence Analysis Results:")
        print(f"- Total analysis duration: {duration:.2f} seconds")
        print(f"- Sampling rate: {data_dict[file1]['sampling_rate']} Hz")
        print(f"- Window length: {config['window_length']}s")
        print(f"- Window overlap: {config['overlap']*100}%")
        print(f"- Frequency band: {config['lowcut']}-{config['highcut']} Hz")
        print(f"- Number of windows used: {len(data_dict[file1]['windows'])}")
        print(f"- Average coherence: {np.mean(Cxy):.4f}")
        print(f"- Maximum coherence: {np.max(Cxy):.4f} at {f[np.argmax(Cxy)]:.2f} Hz")
        
        # Save the coherence results
        output_file = os.path.join(os.path.dirname(__file__), 'coherence_results.npz')
        np.savez(output_file, frequencies=f, coherence=Cxy, 
                file1=file1, file2=file2, duration=duration)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
