import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from seismic_utils.data_loader import SeismicDataLoader
from seismic_utils.models import create_moonnet, create_data_augmentation

def parse_args():
    parser = argparse.ArgumentParser(description='Train a seismic event classifier')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs (default: 50)')
    parser.add_argument('--use_augmentation', action='store_true',
                      help='Use data augmentation (default: False)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Hardcoded paths
    DATA_DIR = r"C:\Users\91995\OneDrive\Desktop\Nasa Space App hackathon\space_apps_2024_seismic_detection\data\lunar\training\data\S12_GradeA"
    OUTPUT_DIR = r"C:\Users\91995\OneDrive\Desktop\Nasa Space App hackathon\output"
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(OUTPUT_DIR, f'model_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    print(f"Using data directory: {DATA_DIR}")
    print(f"Saving output to: {model_dir}")
    
    # Initialize data loader
    print("Initializing data loader...")
    data_loader = SeismicDataLoader(
        data_dir=DATA_DIR,
        sampling_rate=20.0,  # 20 Hz sampling rate
        window_size=20       # 20-second windows
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    st = data_loader.load_mseed_files()
    st = data_loader.preprocess_stream(st)
    
    # Generate training data
    print("Generating training data...")
    X, y = data_loader.generate_training_data(st)
    
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    
    # Split into train/validation sets (80/20 split)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create data augmentation
    data_augmentation = None
    if args.use_augmentation:
        data_augmentation = create_data_augmentation()
    
    # Create model
    print("Creating model...")
    model = create_moonnet(input_shape=(224, 224, 3), num_classes=2)
    
    # Add data augmentation to the model
    if data_augmentation:
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        outputs = model(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_dir, 'final_model.h5'))
    print(f"Model saved to {model_dir}")
    
    # Save training history
    np.save(os.path.join(model_dir, 'training_history.npy'), history.history)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
