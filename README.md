# Seismic Data Analysis for Space Missions

This project provides tools for analyzing and processing seismic data from both lunar and Martian missions. It includes modules for sparse recovery, FFT analysis, and seismic event detection.

## Project Structure

```
.
├── seismic_utils/                # Core utilities
│   ├── data_loader.py           # Data loading and preprocessing
│   └── models.py                # Neural network architectures
├── analysis/                    # Analysis scripts
│   ├── analyze_seismic_fft.py   # Frequency domain analysis
│   ├── coherence_analysis.py    # Seismic coherence analysis
│   ├── lunar_seismic_analysis.py # Lunar-specific analysis
│   └── seismic_analysis.py      # General seismic analysis
├── processing/                  # Data processing
│   ├── process_lunar_seismic.py # Lunar data processing
│   └── seismic_sparse_recovery.py # Sparse recovery methods
├── models/                      # Model definitions
│   ├── Tiny_UNet.py             # Lightweight U-Net model
│   └── optimized_seismic_unet.py # Optimized U-Net for seismic data
├── train.py                     # Model training
├── infer.py                     # Model inference
└── requirements.txt             # Dependencies
```

## Key Features

1. **Seismic Data Processing**
   - Loading and preprocessing of .mseed files
   - Signal filtering and noise reduction
   - Time-frequency analysis

2. **Sparse Recovery**
   - Implementation of sparse recovery algorithms
   - Wavelet-based signal processing
   - Noise estimation and removal

3. **Frequency Domain Analysis**
   - FFT-based spectral analysis
   - Coherence analysis between stations
   - Dispersion curve estimation

4. **Deep Learning Models**
   - U-Net based architectures for seismic data
   - Pre-trained models for event detection
   - Transfer learning capabilities

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Unix/macOS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

### 1. Sparse Recovery
```python
from processing.seismic_sparse_recovery import SeismicSparseRecovery

processor = SeismicSparseRecovery(data_dir='path/to/seismic/data')
processor.load_data()
processor.process()
```

### 2. Frequency Analysis
```bash
python analysis/analyze_seismic_fft.py --input_dir path/to/mseed/files --output_dir results/fft_analysis
```

### 3. Lunar Seismic Analysis
```python
from analysis.lunar_seismic_analysis import LunarSeismicAnalysis

analyzer = LunarSeismicAnalysis(data_dir='path/to/lunar/data')
analyzer.load_data()
analyzer.analyze()
```

## Data Format

The system works with standard .mseed files containing seismic waveform data. The data processing pipeline includes:
1. Data loading and merging of traces
2. Preprocessing (detrending, filtering, resampling)
3. Time-frequency analysis
4. Feature extraction

## Model Training

To train a new model:
```bash
python train.py --data_dir path/to/training/data --model_output models/new_model.h5 --epochs 50 --batch_size 32
```

## Real-time Processing

For real-time seismic data processing:
```bash
python infer.py --model models/pretrained.h5 --input_stream stream_url --output_file predictions.json
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
