import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Stream, UTCDateTime
from scipy import signal, fft
from scipy.signal import correlate, correlation_lags, spectrogram
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

class LunarSeismicAnalysis:
    def __init__(self, data_dir):
        """Initialize the lunar seismic analyzer."""
        self.data_dir = data_dir
        self.st = None
        self.fs = None
        self.npts = None
        self.starttime = None
        self.dt = None
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')  # Fallback to older seaborn style
        sns.set_context("notebook", font_scale=1.1)
        
        # Create output directory
        self.output_dir = 'lunar_analysis_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess lunar seismic data."""
        print("Loading lunar seismic data...")
        mseed_files = [f for f in os.listdir(self.data_dir) if f.endswith('.mseed')]
        if not mseed_files:
            raise FileNotFoundError(f"No .mseed files found in {self.data_dir}")
            
        # Sort files chronologically
        mseed_files.sort()
        
        # Load all traces
        self.st = Stream()
        for f in mseed_files[:2]:  # Use first two traces for analysis
            try:
                file_path = os.path.join(self.data_dir, f)
                print(f"Loading {file_path}")
                st_temp = read(file_path)
                if len(st_temp) > 0:
                    self.st += st_temp
                    print(f"  Loaded {len(st_temp)} traces from {f}")
                else:
                    print(f"  Warning: No traces found in {f}")
            except Exception as e:
                print(f"  Error loading {f}: {str(e)}")
        
        # Get common parameters
        self.fs = self.st[0].stats.sampling_rate
        self.dt = 1.0 / self.fs
        self.npts = min(len(tr.data) for tr in self.st)
        self.starttime = min(tr.stats.starttime for tr in self.st)
        
        # Preprocess
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the seismic data."""
        print("Preprocessing data...")
        # Trim to common length
        min_npts = min(len(tr.data) for tr in self.st)
        for tr in self.st:
            tr.data = tr.data[:min_npts]
            
        # Detrend and demean
        self.st.detrend('linear')
        self.st.detrend('demean')
        
        # Bandpass filter (0.1-10 Hz)
        self.st.filter('bandpass', freqmin=0.1, freqmax=10.0, corners=4, zerophase=True)
    
    def noise_correlation_analysis(self, window_length=100, overlap=0.5):
        """Perform noise correlation analysis between traces."""
        print("Performing noise correlation analysis...")
        
        if len(self.st) < 2:
            print("Need at least two traces for correlation analysis.")
            return None, None, None
        
        # Get the two traces
        tr1, tr2 = self.st[0].data, self.st[1].data
        
        # Calculate cross-correlation
        corr = correlate(tr1, tr2, mode='full', method='auto')
        lags = correlation_lags(len(tr1), len(tr2), mode='full') * self.dt
        
        # Normalize correlation
        corr = corr / np.sqrt(np.sum(tr1**2) * np.sum(tr2**2))
        
        # Plot correlation
        plt.figure(figsize=(12, 6))
        plt.plot(lags, corr, 'b-', linewidth=1.5)
        plt.axvline(0, color='r', linestyle='--', alpha=0.5)
        plt.title('Noise Cross-Correlation Between Sensors', fontsize=14)
        plt.xlabel('Time Lag (s)', fontsize=12)
        plt.ylabel('Normalized Correlation', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'noise_correlation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved noise correlation plot to {output_path}")
        plt.close()
        
        return lags, corr, (tr1, tr2)
    
    def dispersion_analysis(self, signal_window=(10, 30), nperseg=256):
        """Perform dispersion analysis on the seismic data."""
        print("Performing dispersion analysis...")
        
        # Select a trace for analysis
        tr = self.st[0]
        
        # Extract signal window
        start_sample = int(signal_window[0] * self.fs)
        end_sample = int(signal_window[1] * self.fs)
        signal_data = tr.data[start_sample:end_sample]
        
        # Calculate spectrogram
        f, t, Sxx = spectrogram(signal_data, fs=self.fs, nperseg=nperseg, 
                              noverlap=nperseg//2, scaling='density')
        
        # Plot dispersion image
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.title('Dispersion Analysis (Frequency vs. Time)', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.ylim(0, 10)  # Focus on 0-10 Hz range
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'dispersion_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved dispersion analysis plot to {output_path}")
        plt.close()
        
        return f, t, Sxx
    
    def passive_image_interferometry(self, reference_window=(0, 10), 
                                  comparison_window=(10, 30), freq_band=(1.0, 5.0)):  # Adjusted default frequency band
        """Perform passive image interferometry to detect velocity changes."""
        print("Performing passive image interferometry...")
        
        tr = self.st[0]
        
        # Extract reference and comparison windows
        ref_start, ref_end = [int(t * self.fs) for t in reference_window]
        comp_start, comp_end = [int(t * self.fs) for t in comparison_window]
        
        ref_signal = tr.data[ref_start:ref_end]
        comp_signal = tr.data[comp_start:comp_end]
        
        # Bandpass filter to frequency band of interest
        def bandpass_filter(sig, freq_band, fs):
            from scipy.signal import butter, filtfilt
            # Ensure frequencies are within valid range (0 < f < fs/2)
            nyq = 0.5 * fs
            low = max(0.01, min(freq_band[0] / nyq, 0.99))  # Keep within (0, 1)
            high = max(0.01, min(freq_band[1] / nyq, 0.99))  # Keep within (0, 1)
            
            # Ensure low < high
            if low >= high:
                low = max(0.01, high - 0.1)  # Keep a minimum bandwidth
                
            try:
                b, a = butter(4, [low, high], btype='band')
                return filtfilt(b, a, sig)
            except ValueError as e:
                print(f"Warning: Filter error - {e}")
                print(f"fs: {fs}, nyq: {nyq}, low: {low}, high: {high}")
                return sig  # Return original signal if filter fails
        
        ref_filtered = bandpass_filter(ref_signal, freq_band, self.fs)
        comp_filtered = bandpass_filter(comp_signal, freq_band, self.fs)
        
        # Calculate cross-correlation between reference and comparison
        corr = correlate(ref_filtered, comp_filtered, mode='same', method='auto')
        lags = correlation_lags(len(ref_filtered), len(comp_filtered), mode='same') * self.dt
        
        # Find time shift at maximum correlation
        max_idx = np.argmax(np.abs(corr))
        time_shift = lags[max_idx]
        
        # Plot results
        time_ref = np.linspace(0, len(ref_filtered)/self.fs, len(ref_filtered))
        time_comp = np.linspace(0, len(comp_filtered)/self.fs, len(comp_filtered))
        
        plt.figure(figsize=(15, 10))
        
        # Plot reference and comparison signals
        plt.subplot(3, 1, 1)
        plt.plot(time_ref, ref_filtered, 'b-', label='Reference', alpha=0.8)
        plt.plot(time_comp, comp_filtered, 'r-', label='Comparison', alpha=0.8)
        plt.title('Reference vs. Comparison Signals', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot cross-correlation
        plt.subplot(3, 1, 2)
        plt.plot(lags, corr, 'g-', linewidth=1.5)
        plt.axvline(time_shift, color='r', linestyle='--', 
                   label=f'Time shift: {time_shift:.4f} s')
        plt.title('Cross-Correlation Between Reference and Comparison', fontsize=14)
        plt.xlabel('Time Lag (s)', fontsize=12)
        plt.ylabel('Correlation', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot spectrograms
        plt.subplot(3, 1, 3)
        f_ref, t_ref, Sxx_ref = spectrogram(ref_filtered, fs=self.fs, nperseg=128)
        f_comp, t_comp, Sxx_comp = spectrogram(comp_filtered, fs=self.fs, nperseg=128)
        
        plt.pcolormesh(t_ref, f_ref, 10 * np.log10(Sxx_ref), shading='gouraud', 
                      cmap='viridis', alpha=0.6, label='Reference')
        plt.pcolormesh(t_comp + t_comp[-1] + 0.5, f_comp, 10 * np.log10(Sxx_comp), 
                      shading='gouraud', cmap='plasma', alpha=0.6, label='Comparison')
        
        plt.title('Spectrograms of Reference and Comparison Windows', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.legend()
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'passive_interferometry.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved passive interferometry plot to {output_path}")
        plt.close()
        
        return time_shift, corr, lags
    
    def plot_layman_summary(self):
        """Create a summary plot for non-technical audience."""
        print("Creating summary visualization...")
        
        plt.figure(figsize=(15, 12))
        
        # Time series
        plt.subplot(3, 1, 1)
        time = np.arange(self.npts) * self.dt
        plt.plot(time, self.st[0].data, 'b-', alpha=0.7, linewidth=0.8)
        plt.title('Lunar Seismic Activity', fontsize=16, pad=20)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Ground Motion', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotation for non-technical explanation
        plt.annotate(
            'This shows the ground motion recorded by the lunar seismometer.\n' +
            'Spikes indicate seismic events or instrument noise.',
            xy=(0.5, 0.9), xycoords='axes fraction',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            ha='center', fontsize=10
        )
        
        # Frequency content
        plt.subplot(3, 1, 2)
        f, Pxx = signal.welch(self.st[0].data, fs=self.fs, nperseg=1024)
        plt.semilogx(f, 10 * np.log10(Pxx), 'g-', linewidth=1.5)
        plt.title('Frequency Content', fontsize=16, pad=20)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Power (dB)', fontsize=12)
        plt.grid(True, which='both', alpha=0.3)
        
        # Add annotation
        plt.annotate(
            'This shows the frequency content of the seismic data.\n' +
            'Different frequencies can indicate different types of moonquakes.',
            xy=(0.5, 0.9), xycoords='axes fraction',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            ha='center', fontsize=10
        )
        
        # Simple dispersion diagram (simplified for layman)
        plt.subplot(3, 1, 3)
        t = np.linspace(0, 100, 1000)
        f = np.linspace(0.1, 10, 1000)
        v = 1.0 + 0.5 * np.sin(2 * np.pi * t/50)  # Simplified velocity model
        
        # Create dispersion-like curves
        for v_phase in [1.0, 2.0, 3.0]:
            plt.plot(t, v_phase * f, '--', alpha=0.7, 
                    label=f'Phase Velocity ~{v_phase} km/s')
        
        plt.title('Seismic Wave Dispersion (Simplified)', fontsize=16, pad=20)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.ylim(0, 10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotation
        plt.annotate(
            'This simplified diagram shows how different frequency waves\n' +
            'travel at different speeds through the lunar subsurface.',
            xy=(0.5, 0.9), xycoords='axes fraction',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            ha='center', fontsize=10
        )
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'lunar_seismic_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to {output_path}")
        plt.close()
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        try:
            # Load and preprocess data
            self.load_data()
            
            # Perform analyses
            print("\n=== Starting Lunar Seismic Analysis ===\n")
            
            print("1. Noise Correlation Analysis")
            self.noise_correlation_analysis()
            
            print("\n2. Dispersion Analysis")
            self.dispersion_analysis()
            
            print("\n3. Passive Image Interferometry")
            self.passive_image_interferometry()
            
            print("\n4. Creating Summary Visualizations")
            self.plot_layman_summary()
            
            print("\n=== Analysis Complete ===")
            print("\nGenerated output files:")
            print("- noise_correlation.png: Shows correlation between sensors")
            print("- dispersion_analysis.png: Shows wave dispersion characteristics")
            print("- passive_interferometry.png: Shows velocity change analysis")
            print("- lunar_seismic_summary.png: Simplified summary for non-experts")
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            raise

if __name__ == "__main__":
    # Set up paths
    data_dir = r"C:\Users\91995\OneDrive\Desktop\Nasa Space App hackathon\space_apps_2024_seismic_detection\data\lunar\training\data\S12_GradeA"
    
    # Create and run the analyzer
    analyzer = LunarSeismicAnalysis(data_dir)
    analyzer.run_analysis()
