"""
Seismic Denoising U-Net
Implements a U-Net architecture for denoising lunar seismic data.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from obspy import read
from pathlib import Path
import h5py

class SeismicUNet:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _conv_block(self, x, filters, use_bn=True):
        x = Conv2D(filters, (4, 4), strides=2, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        return LeakyReLU(alpha=0.2)(x)
    
    def _deconv_block(self, x, skip, filters, use_dropout=False):
        x = Conv2DTranspose(filters, (4, 4), strides=2, padding='same')(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        if use_dropout:
            x = Dropout(0.5)(x)
        if skip is not None:
            x = Concatenate()([x, skip])
        return x
    
    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        
        # Encoder
        e1 = self._conv_block(inputs, 64, use_bn=False)
        e2 = self._conv_block(e1, 128)
        e3 = self._conv_block(e2, 256)
        e4 = self._conv_block(e3, 512)
        
        # Bottleneck
        b = self._conv_block(e4, 512, use_bn=False)
        
        # Decoder
        d1 = self._deconv_block(b, e4, 512, use_dropout=True)
        d2 = self._deconv_block(d1, e3, 256, use_dropout=True)
        d3 = self._deconv_block(d2, e2, 128, use_dropout=False)
        d4 = self._deconv_block(d3, e1, 64, use_dropout=False)
        
        # Output
        outputs = Conv2D(1, (1, 1), activation='tanh', padding='same')(d4)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(2e-4), loss='mse')
        return model
    
    def train(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=16):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'seismic_denoiser.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint]
        )

def load_seismic_data(file_paths, patch_size=128):
    """Load and preprocess seismic data"""
    patches = []
    for path in file_paths[:100]:  # Limit to 100 files for demo
        try:
            st = read(str(path))
            data = st[0].data.astype('float32')
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            # Create patches
            for i in range(0, len(data) - patch_size, patch_size):
                patch = data[i:i+patch_size]
                if len(patch) == patch_size:
                    patch_2d = np.tile(patch, (patch_size, 1))
                    patches.append(patch_2d)
        except:
            continue
    
    X = np.array(patches)[..., np.newaxis]  # Add channel dim
    return X

if __name__ == "__main__":
    # Example usage
    data_dir = Path("space_apps_2024_seismic_detection/data/lunar/training/data")
    mseed_files = list(data_dir.rglob('*.mseed'))
    
    print(f"Found {len(mseed_files)} .mseed files")
    
    # Load and prepare data
    X = load_seismic_data(mseed_files)
    
    # Generate noisy version
    noise = np.random.normal(0, 0.1, X.shape).astype('float32')
    X_noisy = X + noise
    
    # Split into train/val
    split = int(0.8 * len(X))
    X_train, X_val = X_noisy[:split], X_noisy[split:]
    y_train, y_val = X[:split], X[split:]
    
    # Train model
    unet = SeismicUNet()
    history = unet.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=16)
    
    print("Training complete. Model saved to 'seismic_denoiser.h5'")
