#!/usr/bin/env python3
"""
Tiny U-Net ML Pipeline for Embedded Sensor Processing
Step 1: High-Level Development in Python
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os
import struct

# Configuration
class Config:
    # Model parameters
    INPUT_SIZE = 12  # sqrt(128) rounded up for 2D patches
    INPUT_CHANNELS = 1
    MODEL_SIZE_LIMIT_KB = 20
    QUANTIZE_INT8 = True
    
    # Data directories
    DATA_DIR = "space_apps_2024_seismic_detection/data"
    MARS_TRAIN_DIR = os.path.join(DATA_DIR, "mars/training/data")
    MARS_TEST_DIR = os.path.join(DATA_DIR, "mars/test/data")
    
    # Signal processing
    TARGET_SAMPLE_RATE = 100  # Hz
    WINDOW_SIZE = 128  # samples
    HOP_LENGTH = 64  # samples
    
    # Model training
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Signal processing
    FIR_TAPS = 15  # 10-20 taps as specified
    SAMPLE_RATE = 1000  # Hz
    ISR_FREQUENCY = 50  # 10-100 Hz as specified
    ML_INFERENCE_RATE = 1  # 1 Hz
    NOISE_LEVEL = 0.1
    SIGNAL_FREQ = 10  # Hz

class TinyUNet:
    """Tiny U-Net model for embedded deployment (<20KB)"""
    
    def __init__(self, input_size=128, input_channels=1):
        self.input_size = input_size
        self.input_channels = input_channels
        self.model = None
        
    def depthwise_separable_conv1d(self, x, filters, kernel_size=3, stride=1, name=""):
        """Depthwise separable 1D convolution block"""
        # Depthwise convolution
        x = layers.DepthwiseConv1D(
            kernel_size=kernel_size, 
            strides=stride, 
            padding='same',
            use_bias=False,
            name=f"{name}_dw"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"{name}_dw_act")(x)
        
        # Pointwise convolution
        x = layers.Conv1D(
            filters, 
            1, 
            padding='same', 
            use_bias=False,
            name=f"{name}_pw"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"{name}_pw_act")(x)
        
        return x
    
    def build_model(self):
        """Build 1D U-Net architecture for seismic data"""
        inputs = keras.Input(shape=(self.input_size, self.input_channels))
        
        # Input preprocessing - simulate quantization gain/scale
        x = layers.Lambda(lambda x: x * 127.0, name="quant_scale")(inputs)
        
        # Encoder
        # E1: Conv1D, 1→16
        e1 = layers.Conv1D(16, 3, strides=1, padding='same', use_bias=False, name="e1_conv")(x)
        e1 = layers.BatchNormalization(name="e1_bn")(e1)
        e1 = layers.LeakyReLU(negative_slope=0.1, name="e1_act")(e1)
        e1_pool = layers.MaxPooling1D(pool_size=2, name="e1_pool")(e1)  # 128 -> 64
        
        # E2: DWConv1D, 16→32
        e2 = self.depthwise_separable_conv1d(e1_pool, 32, kernel_size=3, name="e2_conv")
        e2_pool = layers.MaxPooling1D(pool_size=2, name="e2_pool")(e2)  # 64 -> 32
        
        # E3: DWConv1D, 32→64
        e3 = self.depthwise_separable_conv1d(e2_pool, 64, kernel_size=3, name="e3_conv")
        e3_pool = layers.MaxPooling1D(pool_size=2, name="e3_pool")(e3)  # 32 -> 16
        
        # Bottleneck: DWConv1D, 64→64
        bottleneck = self.depthwise_separable_conv1d(e3_pool, 64, kernel_size=3, name="bottleneck")
        
        # Decoder
        # D3: Upsample + concat with E3
        d3 = layers.UpSampling1D(size=2, name="d3_up")(bottleneck)  # 16 -> 32
        d3 = layers.Concatenate(name="d3_concat")([d3, e3])
        d3 = self.depthwise_separable_conv1d(d3, 32, name="d3")
        
        # D2: Upsample + concat with E2
        d2 = layers.UpSampling1D(size=2, name="d2_up")(d3)  # 32 -> 64
        d2 = layers.Concatenate(name="d2_concat")([d2, e2])
        d2 = self.depthwise_separable_conv1d(d2, 16, name="d2")
        
        # D1: Upsample + concat with E1
        d1 = layers.UpSampling1D(size=2, name="d1_up")(d2)  # 64 -> 128
        d1 = layers.Concatenate(name="d1_concat")([d1, e1])
        d1 = self.depthwise_separable_conv1d(d1, 16, name="d1")
        
        # Output layer
        outputs = layers.Conv1D(1, 1, activation='linear', name="output")(d1)
        
        # Output denormalization
        outputs = layers.Lambda(lambda x: x / 127.0, name="dequant_scale")(outputs)
        
        self.model = keras.Model(inputs, outputs, name="TinyUNet1D")
        return self.model
    
    def get_model_size(self):
        """Calculate model size in KB"""
        if self.model is None:
            return 0
        
        param_count = self.model.count_params()
        # Assume INT8 quantization (1 byte per parameter)
        size_kb = param_count / 1024.0
        return size_kb

class FIRFilter:
    """FIR filter implementation for sensor preprocessing"""
    
    def __init__(self, taps=15, cutoff=0.1, filter_type='lowpass'):
        self.taps = taps
        self.cutoff = cutoff
        self.filter_type = filter_type
        self.coefficients = self._design_filter()
        self.buffer = np.zeros(taps)
        
    def _design_filter(self):
        """Design FIR filter using scipy"""
        if self.filter_type == 'lowpass':
            coeffs = scipy.signal.firwin(self.taps, self.cutoff, window='hamming')
        elif self.filter_type == 'bandpass':
            coeffs = scipy.signal.firwin(self.taps, [0.02, 0.3], pass_zero=False)
        else:
            coeffs = scipy.signal.firwin(self.taps, self.cutoff)
        
        return coeffs.astype(np.float32)
    
    def filter_sample(self, sample):
        """Process single sample through FIR filter"""
        # Shift buffer and add new sample
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = sample
        
        # Convolution
        output = np.dot(self.buffer, self.coefficients)
        return output
    
    def filter_batch(self, data):
        """Process batch of data"""
        return scipy.signal.lfilter(self.coefficients, 1.0, data)
    
    def export_coefficients_c(self, filename="fir_coeffs.h"):
        """Export filter coefficients for C++ implementation"""
        with open(filename, 'w') as f:
            f.write("#ifndef FIR_COEFFS_H\n")
            f.write("#define FIR_COEFFS_H\n\n")
            f.write(f"#define FIR_TAPS {self.taps}\n\n")
            f.write("const float fir_coefficients[FIR_TAPS] = {\n")
            for i, coeff in enumerate(self.coefficients):
                f.write(f"    {coeff:.8f}f")
                if i < len(self.coefficients) - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")
            f.write("#endif // FIR_COEFFS_H\n")
        print(f"FIR coefficients exported to {filename}")

class SensorSimulator:
    """Simulate sensor data for testing"""
    
    def __init__(self, sample_rate=1000, noise_level=0.1):
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        
    def generate_synthetic_patch(self, size=128, signal_freq=10):
        """Generate synthetic 2D sensor patch"""
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Create synthetic signal with noise
        signal = np.sin(2 * np.pi * signal_freq * X) * np.cos(2 * np.pi * signal_freq * Y)
        noise = np.random.normal(0, self.noise_level, (size, size))
        
        return signal + noise
    
    def generate_time_series(self, duration=10.0):
        """Generate 1D time series data"""
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        signal = np.sin(2 * np.pi * Config.SIGNAL_FREQ * t)
        noise = np.random.normal(0, self.noise_level, len(t))
        
        return t, signal + noise

class MLPipeline:
    """Complete ML pipeline for embedded deployment"""
    
    def __init__(self):
        self.unet = TinyUNet()
        self.fir_filter = FIRFilter()
        self.simulator = SensorSimulator()
        self.trained_model = None
        
    def load_mars_data(self, data_dir=None, max_files=100):
        """
        Load Mars seismic data from .mseed files
        
        Args:
            data_dir: Directory containing .mseed files (default: Config.MARS_TRAIN_DIR)
            max_files: Maximum number of files to process
            
        Returns:
            X: Input data with shape (n_samples, sequence_length, channels)
            y: Target data (same as input for autoencoder)
        """
        import numpy as np
        from tqdm import tqdm
        from obspy import read
        from pathlib import Path
        
        if data_dir is None:
            data_dir = Config.MARS_TRAIN_DIR
            
        print(f"Loading Mars seismic data from: {os.path.abspath(data_dir)}")
        
        # Find all .mseed files
        mseed_files = list(Path(data_dir).rglob('*.mseed'))
        if not mseed_files:
            raise FileNotFoundError(f"No .mseed files found in {os.path.abspath(data_dir)}")
            
        print(f"Found {len(mseed_files)} .mseed files")
        if max_files:
            mseed_files = mseed_files[:max_files]
            print(f"Processing {len(mseed_files)} files (limited by max_files)")
        
        all_windows = []
        
        for file_path in tqdm(mseed_files, desc="Processing files"):
            try:
                # Load the data
                st = read(str(file_path))
                
                # Process each trace in the stream
                for tr in st:
                    data = tr.data.astype(np.float32)
                    
                    # Skip empty or too short traces
                    if len(data) < Config.INPUT_SIZE:
                        continue
                    
                    # Take the first INPUT_SIZE samples
                    window = data[:Config.INPUT_SIZE]
                    # Normalize the window
                    window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                    all_windows.append(window)
                    
                    # Early stop if we have enough windows
                    if len(all_windows) >= 1000:  # Limit windows per file
                        break
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
        if not all_windows:
            raise ValueError("No valid data windows were created")
            
        # Convert to numpy array and reshape for 1D conv
        all_windows = np.array(all_windows)
        all_windows = all_windows[..., np.newaxis]  # Add channel dimension
        
        # For autoencoder, input = target
        return all_windows.astype(np.float32), all_windows.astype(np.float32)
    
    def prepare_training_data(self, test_split=0.2, max_files=100):
        """
        Prepare training and validation data from Mars seismic data
        
        Args:
            test_split: Fraction of data to use for testing
            max_files: Maximum number of files to process
            
        Returns:
            X_train, X_val, y_train, y_val: Training and validation data
        """
        # Load data
        X, y = self.load_mars_data(max_files=max_files)
        
        # Split into train/validation
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Input shape: {X_train[0].shape}")
        
        # Add noise to create noisy-clean pairs for denoising
        def add_noise(clean_data, noise_level=0.1):
            noise = np.random.normal(0, noise_level, clean_data.shape)
            return clean_data + noise
        
        # Only add noise to input, keep targets clean
        X_train_noisy = add_noise(X_train)
        X_val_noisy = add_noise(X_val)
        
        return X_train_noisy, X_val_noisy, y_train, y_val
    
    def train_model(self, epochs=50, batch_size=8):
        """Train the Tiny U-Net model"""
        print("Training Tiny U-Net model...")
        
        # Build model
        model = self.unet.build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Generate training data
        X_train, X_val, y_train, y_val = self.prepare_training_data(test_split=0.2, max_files=1000)
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        self.trained_model = model
        
        # Check model size
        size_kb = self.unet.get_model_size()
        print(f"Model size: {size_kb:.2f} KB")
        
        if size_kb > Config.MODEL_SIZE_LIMIT_KB:
            print(f"WARNING: Model size ({size_kb:.2f} KB) exceeds limit ({Config.MODEL_SIZE_LIMIT_KB} KB)")
        
        return history
    
    def quantize_model(self):
        """Convert model to INT8 TensorFlow Lite"""
        if self.trained_model is None:
            print("No trained model available for quantization")
            return None
        
        print("Quantizing model to INT8...")
        
        # Representative dataset for quantization
        X_repr, _ = self.prepare_training_data(100)
        
        def representative_data_gen():
            for i in range(100):
                yield [X_repr[i:i+1]]
        
        # Convert to TensorFlow Lite with INT8 quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.trained_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_quant_model = converter.convert()
        
        # Save model
        with open('tiny_unet_int8.tflite', 'wb') as f:
            f.write(tflite_quant_model)
        
        print(f"Quantized model size: {len(tflite_quant_model) / 1024:.2f} KB")
        return tflite_quant_model
    
    def export_for_cpp(self):
        """Export model and filter coefficients for C++ implementation"""
        print("Exporting for C++ implementation...")
        
        # Export FIR filter coefficients
        self.fir_filter.export_coefficients_c("fir_coeffs.h")
        
        # Quantize and save model
        self.quantize_model()
        
        # Generate C++ header with model data
        self.generate_model_header()
    
    def generate_model_header(self):
        """Generate C++ header file with model weights"""
        if not os.path.exists('tiny_unet_int8.tflite'):
            print("No quantized model found. Run quantize_model() first.")
            return
        
        with open('tiny_unet_int8.tflite', 'rb') as f:
            model_data = f.read()
        
        with open('model_data.h', 'w') as f:
            f.write("#ifndef MODEL_DATA_H\n")
            f.write("#define MODEL_DATA_H\n\n")
            f.write(f"#define MODEL_SIZE {len(model_data)}\n\n")
            f.write("const unsigned char model_data[MODEL_SIZE] = {\n")
            
            for i, byte in enumerate(model_data):
                if i % 12 == 0:
                    f.write("    ")
                f.write(f"0x{byte:02x}")
                if i < len(model_data) - 1:
                    f.write(",")
                if i % 12 == 11:
                    f.write("\n")
            
            f.write("\n};\n\n")
            f.write("#endif // MODEL_DATA_H\n")
        
        print("Model data exported to model_data.h")
    
    def test_pipeline(self):
        """Test complete pipeline with synthetic data"""
        print("Testing complete pipeline...")
        
        # Generate test time series
        t, raw_signal = self.simulator.generate_time_series(duration=5.0)
        
        # Apply FIR filtering
        filtered_signal = self.fir_filter.filter_batch(raw_signal)
        
        # Generate test patches for ML inference
        test_patches = []
        for i in range(5):
            patch = self.simulator.generate_synthetic_patch()
            test_patches.append(patch)
        
        test_patches = np.array(test_patches)[..., np.newaxis]
        
        if self.trained_model:
            ml_output = self.trained_model.predict(test_patches)
        
        # Visualization
        self.visualize_results(t, raw_signal, filtered_signal, test_patches[0:1], 
                             ml_output[0:1] if self.trained_model else None)
    
    def visualize_results(self, t, raw_signal, filtered_signal, input_patch, ml_output):
        """Visualize pipeline results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Time series plots
        axes[0, 0].plot(t, raw_signal, alpha=0.7, label='Raw Signal')
        axes[0, 0].plot(t, filtered_signal, label='FIR Filtered')
        axes[0, 0].set_title('Signal Processing Pipeline')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].legend()
        
        # FIR filter frequency response
        w, h = scipy.signal.freqz(self.fir_filter.coefficients)
        axes[0, 1].plot(w / np.pi, 20 * np.log10(abs(h)))
        axes[0, 1].set_title('FIR Filter Frequency Response')
        axes[0, 1].set_xlabel('Normalized Frequency')
        axes[0, 1].set_ylabel('Amplitude (dB)')
        
        # FIR coefficients
        axes[0, 2].stem(range(len(self.fir_filter.coefficients)), 
                       self.fir_filter.coefficients)
        axes[0, 2].set_title('FIR Filter Coefficients')
        
        # ML model input/output
        if input_patch is not None:
            im1 = axes[1, 0].imshow(input_patch[0, :, :, 0], cmap='viridis')
            axes[1, 0].set_title('ML Input Patch')
            plt.colorbar(im1, ax=axes[1, 0])
        
        if ml_output is not None:
            im2 = axes[1, 1].imshow(ml_output[0, :, :, 0], cmap='viridis')
            axes[1, 1].set_title('ML Output Patch')
            plt.colorbar(im2, ax=axes[1, 1])
            
            # Difference
            diff = input_patch[0, :, :, 0] - ml_output[0, :, :, 0]
            im3 = axes[1, 2].imshow(diff, cmap='RdBu_r')
            axes[1, 2].set_title('Input - Output Difference')
            plt.colorbar(im3, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('pipeline_results.png', dpi=150, bbox_inches='tight')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Step 1: Train the model
    print("=== Step 1: Model Training ===")
    history = pipeline.train_model(epochs=20, batch_size=4)
    
    # Step 2: Test the complete pipeline
    print("\n=== Step 2: Pipeline Testing ===")
    pipeline.test_pipeline()
    
    # Step 3: Export for C++ implementation
    print("\n=== Step 3: Export for C++ ===")
    pipeline.export_for_cpp()
    
    print("\n=== Pipeline Setup Complete ===")
    print("Generated files:")
    print("- fir_coeffs.h: FIR filter coefficients for C++")
    print("- model_data.h: Quantized model data for C++")
    print("- tiny_unet_int8.tflite: TensorFlow Lite model")
    print("- pipeline_results.png: Test results visualization")