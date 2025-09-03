import os
import numpy as np
import tensorflow as tf
import argparse
from obspy import read, Stream
from seismic_utils.data_loader import SeismicDataLoader

def load_model(model_path):
    """Load a trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return tf.keras.models.load_model(model_path)

def predict_quake(model, data_loader, data):
    """Predict if the input data contains a quake."""
    # Process the input data window
    spectrogram = data_loader.process_window(data, data_loader.sampling_rate)
    
    # Add batch dimension
    spectrogram = np.expand_dims(spectrogram, axis=0)
    
    # Make prediction
    prediction = model.predict(spectrogram, verbose=0)
    
    # Get class probabilities
    prob_quake = prediction[0][1]  # Assuming class 1 is 'quake'
    
    return {
        'is_quake': prob_quake > 0.5,  # Threshold can be adjusted
        'confidence': float(prob_quake),
        'class_probs': {
            'noise': float(prediction[0][0]),
            'quake': float(prediction[0][1])
        }
    }

def process_stream(stream, model, data_loader):
    """Process a stream of seismic data in real-time."""
    results = []
    
    for tr in stream:
        data = tr.data
        fs = tr.stats.sampling_rate
        
        # Ensure data is at the correct sampling rate
        if fs != data_loader.sampling_rate:
            tr.resample(data_loader.sampling_rate)
            data = tr.data
        
        # Process in windows
        num_windows = len(data) // data_loader.samples_per_window
        
        for i in range(num_windows):
            start = i * data_loader.samples_per_window
            end = start + data_loader.samples_per_window
            window = data[start:end]
            
            # Skip empty or invalid windows
            if np.all(np.isnan(window)) or np.all(window == 0):
                continue
                
            # Make prediction
            result = predict_quake(model, data_loader, window)
            
            # Add timestamp and other metadata
            window_time = tr.stats.starttime + (start / fs)
            result.update({
                'start_time': window_time.isoformat(),
                'end_time': (window_time + data_loader.window_size).isoformat(),
                'station': tr.stats.station,
                'channel': tr.stats.channel
            })
            
            results.append(result)
            
            # Print prediction
            status = "QUAKE DETECTED!" if result['is_quake'] else "No quake"
            print(f"{window_time} - {status} (Confidence: {result['confidence']:.2f})")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Real-time quake detection')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing .mseed files for inference')
    parser.add_argument('--output_file', type=str, default='predictions.json',
                      help='File to save predictions')
    args = parser.parse_args()
    
    # Initialize data loader
    data_loader = SeismicDataLoader(
        data_dir=args.data_dir,
        sampling_rate=20.0,  # Must match training
        window_size=20       # Must match training
    )
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    st = data_loader.load_mseed_files()
    st = data_loader.preprocess_stream(st)
    
    # Process stream and detect quakes
    print("\nStarting quake detection...")
    print("--------------------------------------------------")
    results = process_stream(st, model, data_loader)
    
    # Save results
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nPredictions saved to {args.output_file}")

if __name__ == '__main__':
    main()
