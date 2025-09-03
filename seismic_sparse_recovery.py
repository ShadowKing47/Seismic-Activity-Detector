import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, Stream
from scipy import signal, fft, interpolate
from scipy.optimize import minimize_scalar, brentq
from scipy.sparse.linalg import LinearOperator, lsqr
from tqdm import tqdm
import h5py
from datetime import datetime

class SeismicSparseRecovery:
    def __init__(self, data_dir, output_dir='output'):
        """Initialize the sparse recovery processor."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default parameters (adjustable)
        self.c0 = 45.0  # Mars surface wave velocity (m/s)
        self.pmax = 1.0 / self.c0  # Maximum slowness (s/m)
        self.dt = 0.01  # Time sampling (s) - will be updated from data
        self.dx = 100.0  # Receiver spacing (m) - adjust based on your array geometry
        
        # Wavelet parameters
        self.wavelet_freq = 2.0  # Center frequency for Ricker wavelet (Hz)
        self.wavelet_length = 100  # Samples
        
        # Processing parameters
        self.noise_window = [0, 10]  # Time window for noise estimation (s)
        self.signal_window = [10, 30]  # Time window for signal (s)
        self.padding = 10.0  # Padding around signal window (s)
        
        # Radon transform parameters
        self.np = 101  # Number of slowness values
        self.nt = None  # Will be set based on data
        self.p_vals = np.linspace(-self.pmax, self.pmax, self.np)
        
        # Data containers
        self.st = None  # Raw data (ObsPy Stream)
        self.noise_std = None
        self.wavelet = None
        self.m = None  # Sparse model (tau-p domain)
        
    def load_data(self):
        """Load and preprocess seismic data."""
        print("Loading seismic data...")
        mseed_files = [f for f in os.listdir(self.data_dir) if f.endswith('.mseed')]
        if not mseed_files:
            raise FileNotFoundError(f"No .mseed files found in {self.data_dir}")
            
        # Load first file for now (can be extended to multiple files)
        file_path = os.path.join(self.data_dir, mseed_files[0])
        self.st = read(file_path)
        print(f"Loaded {len(self.st)} traces from {file_path}")
        
        # Update parameters from data
        self.dt = self.st[0].stats.delta
        self.nt = len(self.st[0].data)
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the loaded seismic data."""
        print("Preprocessing data...")
        
        # Detrend and demean
        for tr in self.st:
            tr.detrend('linear')
            tr.detrend('demean')
            
        # Bandpass filter (0.1-10 Hz)
        self.st.filter('bandpass', freqmin=0.1, freqmax=10.0, corners=4, zerophase=True)
        
        # Estimate noise level
        self._estimate_noise()
        
        # Estimate or create source wavelet
        self._estimate_wavelet()
        
    def _estimate_noise(self):
        """Estimate noise standard deviation from quiet intervals."""
        print("Estimating noise level...")
        noise_data = []
        
        for tr in self.st:
            # Convert time window to samples
            start_sample = int(self.noise_window[0] / self.dt)
            end_sample = int(self.noise_window[1] / self.dt)
            noise_segment = tr.data[start_sample:end_sample]
            noise_data.append(noise_segment)
            
        # Calculate RMS noise
        noise_rms = np.sqrt(np.mean(np.square(noise_data), axis=1))
        self.noise_std = np.sqrt(np.sum(noise_rms**2))
        print(f"Estimated noise standard deviation: {self.noise_std:.4e}")
        
    def _estimate_wavelet(self):
        """Estimate source wavelet from the data or create a Ricker wavelet."""
        print("Estimating source wavelet...")
        
        # For now, create a Ricker wavelet
        t = np.linspace(-self.wavelet_length//2, self.wavelet_length//2, self.wavelet_length) * self.dt
        self.wavelet = (1.0 - 2.0 * (np.pi * self.wavelet_freq * t)**2) * np.exp(-(np.pi * self.wavelet_freq * t)**2)
        self.wavelet = self.wavelet / np.max(np.abs(self.wavelet))  # Normalize
        
        # TODO: Add wavelet estimation from data using spectral division
        
    def apply_cone_constraint(self, m):
        """Apply cone constraint to the model."""
        # Simple cone constraint: zero out coefficients outside |p| <= pmax
        # This is a placeholder - implement your specific constraint
        return m
        
    def forward_operator(self, m):
        """Forward operator: m (tau-p) -> d (time-space)"""
        # Reshape model vector to 2D (tau, p)
        m_2d = m.reshape((self.nt, self.np))
        
        # Apply cone constraint
        m_2d = self.apply_cone_constraint(m_2d)
        
        # Slant stack (inverse Radon transform)
        d = np.zeros((self.nt, len(self.st)))
        
        for i, tr in enumerate(self.st):
            x = i * self.dx  # Receiver position
            for j, p in enumerate(self.p_vals):
                # Time shift: tau = t - p*x
                t_shifted = np.arange(self.nt) * self.dt - p * x
                # Interpolate to get m(tau, p)
                interp_func = interpolate.interp1d(
                    np.arange(self.nt) * self.dt,
                    m_2d[:, j],
                    kind='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                d[:, i] += interp_func(t_shifted)
        
        # Convolve with wavelet
        d_convolved = np.zeros_like(d)
        for i in range(d.shape[1]):
            d_convolved[:, i] = np.convolve(d[:, i], self.wavelet, mode='same')
            
        return d_convolved.ravel()
    
    def adjoint_operator(self, d):
        """Adjoint operator: d (time-space) -> m (tau-p)"""
        d = d.reshape((self.nt, len(self.st)))
        m = np.zeros((self.nt, self.np))
        
        # Cross-correlate with wavelet
        d_corr = np.zeros_like(d)
        for i in range(d.shape[1]):
            d_corr[:, i] = signal.correlate(d[:, i], self.wavelet, mode='same')
        
        # Slant stack (forward Radon transform)
        for i, tr in enumerate(self.st):
            x = i * self.dx  # Receiver position
            for j, p in enumerate(self.p_vals):
                # Time shift: t = tau + p*x
                t_shifted = np.arange(self.nt) * self.dt + p * x
                # Interpolate to get d(t, x)
                interp_func = interpolate.interp1d(
                    np.arange(self.nt) * self.dt,
                    d_corr[:, i],
                    kind='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                m[:, j] += interp_func(t_shifted)
                
        return m.ravel()
    
    def l1_minimization(self, A, b, sigma, maxiter=100, tol=1e-4):
        """Solve min ||m||_1 s.t. ||A(m) - b||_2 <= sigma using FISTA."""
        print("Solving L1 minimization problem...")
        
        # Initialize
        m = np.zeros(A.shape[1])
        y = m.copy()
        t = 1.0
        
        # FISTA parameters
        L = 1.0  # Lipschitz constant (will be estimated)
        
        for it in tqdm(range(maxiter), desc="FISTA iterations"):
            # Store previous iteration
            m_prev = m.copy()
            
            # Gradient step
            residual = A.dot(y) - b
            grad = A.T.dot(residual)
            y = y - (1.0/L) * grad
            
            # Soft thresholding (proximal operator for L1 norm)
            m = np.sign(y) * np.maximum(np.abs(y) - 1.0/L, 0.0)
            
            # Update step size
            t_prev = t
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
            
            # Update y for next iteration
            y = m + ((t_prev - 1.0) / t) * (m - m_prev)
            
            # Check convergence
            residual_norm = np.linalg.norm(residual)
            if residual_norm <= sigma * 1.01:  # Allow small tolerance
                print(f"Converged after {it+1} iterations")
                break
                
        return m
    
    def process(self):
        """Main processing pipeline."""
        # Load and preprocess data
        self.load_data()
        
        # Extract data vector b
        b = np.vstack([tr.data for tr in self.st]).T.ravel()
        
        # Create linear operator
        A = LinearOperator(
            (len(b), self.nt * self.np),
            matvec=self.forward_operator,
            rmatvec=self.adjoint_operator
        )
        
        # Solve the sparse recovery problem
        self.m = self.l1_minimization(A, b, self.noise_std)
        
        # Reconstruct the signal
        d_recon = self.forward_operator(self.m)
        
        # Save results
        self.save_results(d_recon.reshape((self.nt, len(self.st))), b.reshape((self.nt, len(self.st))))
        
        return d_recon
    
    def save_results(self, d_recon, d_obs):
        """Save the processing results."""
        # Save reconstructed data
        np.savez_compressed(
            os.path.join(self.output_dir, 'reconstruction.npz'),
            d_recon=d_recon,
            d_obs=d_obs,
            m=self.m.reshape((self.nt, self.np)),
            p_vals=self.p_vals,
            noise_std=self.noise_std,
            dt=self.dt,
            dx=self.dx
        )
        
        # Save plots
        self.plot_results(d_recon, d_obs)
    
    def plot_results(self, d_recon, d_obs):
        """Generate and save result plots."""
        plt.figure(figsize=(15, 10))
        
        # Plot observed data
        plt.subplot(3, 1, 1)
        plt.imshow(d_obs.T, aspect='auto', cmap='seismic',
                  extent=[0, self.nt*self.dt, len(self.st), 0])
        plt.colorbar(label='Amplitude')
        plt.title('Observed Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Receiver')
        
        # Plot reconstructed data
        plt.subplot(3, 1, 2)
        plt.imshow(d_recon.T, aspect='auto', cmap='seismic',
                  extent=[0, self.nt*self.dt, len(self.st), 0])
        plt.colorbar(label='Amplitude')
        plt.title('Reconstructed Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Receiver')
        
        # Plot difference
        plt.subplot(3, 1, 3)
        plt.imshow((d_obs - d_recon).T, aspect='auto', cmap='seismic',
                  extent=[0, self.nt*self.dt, len(self.st), 0])
        plt.colorbar(label='Residual')
        plt.title('Residual (Observed - Reconstructed)')
        plt.xlabel('Time (s)')
        plt.ylabel('Receiver')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reconstruction_comparison.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    # Set up paths
    data_dir = r"C:\Users\91995\OneDrive\Desktop\Nasa Space App hackathon\space_apps_2024_seismic_detection\data\mars\training\data"
    output_dir = os.path.join(os.path.dirname(__file__), 'sparse_recovery_output')
    
    # Create and run the processor
    processor = SeismicSparseRecovery(data_dir, output_dir)
    d_recon = processor.process()
    
    print(f"Processing complete. Results saved to: {output_dir}")
