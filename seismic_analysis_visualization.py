import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Stream
from scipy import signal, fft
from scipy.signal import spectrogram, welch
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

class SeismicVisualizer:
    def __init__(self, data_dir):
        """Initialize the seismic visualizer."""
        self.data_dir = data_dir
        self.st = None
        self.fs = None
        self.npts = None
        self.starttime = None
        
    def load_data(self):
        """Load and preprocess seismic data."""
        print("Loading seismic data...")
        mseed_files = [f for f in os.listdir(self.data_dir) if f.endswith('.mseed')]
        if not mseed_files:
            raise FileNotFoundError(f"No .mseed files found in {self.data_dir}")
            
        # Load first file
        file_path = os.path.join(self.data_dir, mseed_files[0])
        self.st = read(file_path)
        self.fs = self.st[0].stats.sampling_rate
        self.npts = self.st[0].stats.npts
        self.starttime = self.st[0].stats.starttime.datetime
        
        # Preprocess
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the seismic data."""
        print("Preprocessing data...")
        # Detrend and demean
        self.st.detrend('linear')
        self.st.detrend('demean')
        
        # Bandpass filter (0.1-10 Hz)
        self.st.filter('bandpass', freqmin=0.1, freqmax=10.0, corners=4, zerophase=True)
    
    def plot_time_series(self):
        """Plot the time series of the seismic data."""
        plt.figure(figsize=(15, 6))
        
        # Time vector
        time = np.arange(self.npts) / self.fs
        
        # Plot each component
        for i, tr in enumerate(self.st):
            plt.plot(time, tr.data, label=f'Component {i+1}', alpha=0.8, linewidth=0.8)
        
        plt.title('Seismic Time Series')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('seismic_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_spectrogram(self, nperseg=256, noverlap=None):
        """Plot spectrogram of the seismic data."""
        if noverlap is None:
            noverlap = nperseg // 2
            
        plt.figure(figsize=(15, 8))
        
        for i, tr in enumerate(self.st):
            f, t, Sxx = spectrogram(tr.data, fs=self.fs, nperseg=nperseg, 
                                  noverlap=noverlap, scaling='density')
            
            # Limit frequency range to 0-10 Hz for better visibility
            f_mask = f <= 10.0
            f = f[f_mask]
            Sxx = Sxx[f_mask, :]
            
            plt.subplot(len(self.st), 1, i+1)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.title(f'Spectrogram - Component {i+1}')
            plt.ylabel('Frequency [Hz]')
            plt.ylim(0, 10)  # Focus on 0-10 Hz range
        
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig('seismic_spectrogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_psd(self, nperseg=1024):
        """Plot power spectral density of the seismic data."""
        plt.figure(figsize=(12, 6))
        
        for i, tr in enumerate(self.st):
            f, Pxx = welch(tr.data, fs=self.fs, nperseg=nperseg)
            
            # Convert to dB
            Pxx_dB = 10 * np.log10(Pxx)
            
            # Limit frequency range
            f_mask = f <= 10.0
            f = f[f_mask]
            Pxx_dB = Pxx_dB[f_mask]
            
            plt.semilogx(f, Pxx_dB, label=f'Component {i+1}', alpha=0.8, linewidth=1.5)
        
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.grid(True, which='both', alpha=0.3)
        plt.xlim(0.1, 10)  # Focus on 0.1-10 Hz range
        plt.legend()
        plt.tight_layout()
        plt.savefig('seismic_psd.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_envelope(self, window_size=100):
        """Plot the envelope of the seismic signal."""
        plt.figure(figsize=(15, 6))
        
        time = np.arange(self.npts) / self.fs
        
        for i, tr in enumerate(self.st):
            # Calculate envelope using Hilbert transform
            analytic_signal = signal.hilbert(tr.data)
            envelope = np.abs(analytic_signal)
            
            # Smooth the envelope
            window = np.ones(window_size) / window_size
            smooth_envelope = np.convolve(envelope, window, mode='same')
            
            # Plot original and envelope
            plt.plot(time, tr.data, 'b-', alpha=0.3, label=f'Original {i+1}')
            plt.plot(time, smooth_envelope, 'r-', label=f'Envelope {i+1}', linewidth=1.5)
        
        plt.title('Seismic Signal Envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('seismic_envelope.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_histogram(self, bins=100):
        """Plot amplitude distribution histogram."""
        plt.figure(figsize=(12, 6))
        
        for i, tr in enumerate(self.st):
            sns.histplot(tr.data, bins=bins, kde=True, 
                        label=f'Component {i+1}', alpha=0.5)
        
        plt.title('Amplitude Distribution')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('seismic_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_noise_vs_signal(self, signal_window=(10, 30), noise_window=(0, 10)):
        """Compare noise and signal characteristics."""
        plt.figure(figsize=(15, 10))
        
        for i, tr in enumerate(self.st):
            # Extract noise and signal segments
            noise_start = int(noise_window[0] * self.fs)
            noise_end = int(noise_window[1] * self.fs)
            signal_start = int(signal_window[0] * self.fs)
            signal_end = int(signal_window[1] * self.fs)
            
            noise_segment = tr.data[noise_start:noise_end]
            signal_segment = tr.data[signal_start:signal_end]
            
            # Time vectors
            noise_time = np.linspace(noise_window[0], noise_window[1], len(noise_segment))
            signal_time = np.linspace(signal_window[0], signal_window[1], len(signal_segment))
            
            # Plot time series comparison
            plt.subplot(2, 1, 1)
            plt.plot(noise_time, noise_segment, 'b-', alpha=0.7, 
                    label=f'Noise (t={noise_window[0]}-{noise_window[1]}s)' if i == 0 else '')
            plt.plot(signal_time, signal_segment, 'r-', alpha=0.7,
                    label=f'Signal (t={signal_window[0]}-{signal_window[1]}s)' if i == 0 else '')
            
            # Plot PSD comparison
            plt.subplot(2, 1, 2)
            f_noise, Pxx_noise = welch(noise_segment, fs=self.fs, nperseg=min(256, len(noise_segment)//4))
            f_signal, Pxx_signal = welch(signal_segment, fs=self.fs, nperseg=min(256, len(signal_segment)//4))
            
            plt.semilogy(f_noise, Pxx_noise, 'b-', alpha=0.7, 
                        label=f'Noise PSD' if i == 0 else '')
            plt.semilogy(f_signal, Pxx_signal, 'r-', alpha=0.7,
                        label=f'Signal PSD' if i == 0 else '')
        
        # Format plots
        plt.subplot(2, 1, 1)
        plt.title('Noise vs Signal Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.title('Power Spectral Density Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency')
        plt.xlim(0, 10)  # Focus on 0-10 Hz range
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('noise_vs_signal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run all analysis and generate plots."""
        self.load_data()
        
        print("Generating visualizations...")
        self.plot_time_series()
        self.plot_spectrogram()
        self.plot_psd()
        self.plot_envelope()
        self.plot_histogram()
        self.analyze_noise_vs_signal()
        
        print("Analysis complete. Check the generated plots in the current directory.")

if __name__ == "__main__":
    # Set up paths
    data_dir = r"C:\Users\91995\OneDrive\Desktop\Nasa Space App hackathon\space_apps_2024_seismic_detection\data\mars\training\data"
    
    # Create and run the visualizer
    visualizer = SeismicVisualizer(data_dir)
    visualizer.run_analysis()
