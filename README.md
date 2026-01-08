# EEG Dashboard

A high-performance, real-time EEG visualization and analysis dashboard for 25-pin connector EEG caps via USB adapters.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

## Features

### Core Visualization
- **Real-time waveform display** - Scrolling EEG traces for all 20 channels with adjustable time window and amplitude scale
- **2D Brain topographic map** - Scalp heatmap with multiple styles (PET scan, hot metal, clinical)
- **3D Brain visualization** - Interactive 3D head model with electrode activity mapping
- **Frequency band analysis** - Real-time Delta, Theta, Alpha, Beta, Gamma power bars

### Signal Processing
- **High-performance filtering** - Optimized bandpass (0.5-50 Hz) and notch (50/60 Hz) filters using cascaded biquad sections
- **Adaptive artifact detection** - Automatic threshold adjustment based on running statistics
- **Multiple reference schemes** - Common average (CAR), linked ears, bipolar montage
- **Zero-phase offline filtering** - Forward-backward filtering for batch processing

### Advanced Analysis
- **Spectral analysis** - Welch PSD, FFT, band power computation
- **Connectivity analysis** - Coherence, Phase Locking Value (PLV) between channels
- **Real-time spectrogram** - Time-frequency visualization
- **Basic ICA** - Independent Component Analysis for artifact removal
- **Cognitive indices** - Engagement, relaxation, focus, drowsiness metrics

### Data Management
- **Recording** - Capture EEG data during sessions
- **Export formats** - CSV, EDF (European Data Format), NumPy
- **Session logging** - Automatic metadata, event markers, performance metrics
- **10-20 electrode system** - Standard international EEG placement

## Installation

### Prerequisites
- Python 3.10 or higher
- USB-to-Serial adapter for your EEG cap (optional - simulator included)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/eegDashboard.git
cd eegDashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computing |
| scipy | ≥1.10.0 | Signal processing |
| mne | ≥1.6.0 | EEG-specific processing |
| PyQt6 | ≥6.5.0 | GUI framework |
| pyqtgraph | ≥0.13.0 | Real-time plotting |
| pyserial | ≥3.5 | Hardware communication |
| pyedflib | ≥0.1.30 | EDF file format |
| pyyaml | ≥6.0 | Configuration |

## Quick Start

### Launch Dashboard
```bash
python main.py
```
The dashboard starts in **Simulator mode** by default - no hardware needed!

### Run Tests
```bash
# Full test suite
pytest tests/ -v

# Quick component tests (no GUI)
python main.py --test
```

### Run Performance Benchmarks
```bash
pytest tests/test_suite.py -v -k "performance"
```

## Usage Guide

### Dashboard Controls

1. **Source Selection** - Choose between Simulator or Serial port
2. **Connect/Disconnect** - Start/stop data acquisition
3. **Record** - Begin recording EEG data
4. **Save** - Export recording to CSV or EDF format

### Tabs

| Tab | Description |
|-----|-------------|
| 📊 Waveforms | Real-time scrolling EEG traces for all channels |
| 🧠 2D Brain Map | Topographic map with PET/hot metal/simple styles |
| 🎮 3D Brain | Interactive 3D visualization with rotation |
| 📐 Multi-View | Multiple simultaneous brain views |

### With Your EEG Hardware

1. Connect your EEG cap to the USB adapter
2. Launch the dashboard: `python main.py`
3. Select your serial port from the dropdown
4. Click **Connect** to start acquisition

### Configuration

Edit `config/default_config.yaml`:

```yaml
hardware:
  port: null              # Serial port or null for auto-detect
  baud_rate: 115200
  n_channels: 21
  sample_rate: 256
  use_simulator: true     # Set false for real hardware

processing:
  bandpass:
    low: 0.5              # High-pass cutoff (Hz)
    high: 50.0            # Low-pass cutoff (Hz)
  notch_freq: 60.0        # 60 Hz (US) or 50 Hz (EU)
  artifact_rejection: true

display:
  time_window: 10.0       # Seconds of data to show
  amplitude_scale: 100.0  # µV per division
  refresh_rate: 30        # FPS
  theme: dark
```

## Architecture

```
eegDashboard/
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Test configuration
├── config/
│   └── default_config.yaml          # Default settings
├── src/
│   ├── core/
│   │   ├── channel_map.py           # 10-20 electrode definitions
│   │   ├── constants.py             # Shared constants (NEW)
│   │   ├── data_export.py           # CSV/EDF export
│   │   └── session_logging.py       # Session management (NEW)
│   ├── hardware/
│   │   ├── serial_interface.py      # USB-serial communication
│   │   └── simulator.py             # Synthetic EEG generator
│   ├── processing/
│   │   ├── filters.py               # Basic IIR filters
│   │   ├── optimized_filters.py     # High-performance filters (NEW)
│   │   ├── spectral.py              # Basic spectral analysis
│   │   ├── optimized_spectral.py    # Fast spectral analysis (NEW)
│   │   └── advanced_analysis.py     # ICA, connectivity (NEW)
│   ├── ui/
│   │   └── dashboard.py             # Main PyQt6 interface
│   └── visualization/
│       ├── waveform_plot.py         # Real-time EEG traces
│       ├── topomap.py               # Basic topographic map
│       ├── topomap_enhanced.py      # Enhanced 2D visualization
│       ├── brain_3d.py              # 3D brain rendering
│       ├── brain_anatomical.py      # Anatomical overlays
│       └── fast_renderer.py         # Optimized rendering (NEW)
└── tests/
    ├── test_suite.py                # Comprehensive tests (NEW)
    └── test_waveform.py             # Waveform component test
```

## API Reference

### Filtering

```python
from src.processing.optimized_filters import OptimizedEEGFilter, FilterSettings

# Configure filter
settings = FilterSettings(
    sample_rate=256,
    bandpass_low=0.5,
    bandpass_high=50.0,
    notch_freq=60.0,
)

# Create filter
eeg_filter = OptimizedEEGFilter(settings)

# Real-time filtering (maintains state)
filtered_sample = eeg_filter.filter_sample(sample)  # (n_channels,)
filtered_chunk = eeg_filter.filter_chunk(data)      # (n_samples, n_channels)

# Offline filtering (zero-phase)
clean_data = eeg_filter.filter_batch(data)
```

### Spectral Analysis

```python
from src.processing.optimized_spectral import OptimizedSpectralAnalyzer, BandPower

analyzer = OptimizedSpectralAnalyzer(sample_rate=256, window_size=2.0)

# Compute band power
bp = analyzer.compute_band_power(signal)
print(f"Alpha: {bp.alpha:.2f}, Beta: {bp.beta:.2f}")
print(f"Relative: {bp.relative()}")

# Multi-channel analysis
bp_all = analyzer.compute_band_power_multichannel(data)  # Dict[str, np.ndarray]

# Cognitive indices
from src.processing.optimized_spectral import compute_engagement_index, compute_relaxation_index
engagement = compute_engagement_index(bp)  # beta / (alpha + theta)
relaxation = compute_relaxation_index(bp)  # alpha / beta
```

### Connectivity Analysis

```python
from src.processing.advanced_analysis import ConnectivityAnalyzer

conn = ConnectivityAnalyzer(sample_rate=256)

# Coherence between two channels
result = conn.coherence(signal1, signal2)
print(f"Mean coherence: {result.mean_coherence:.3f}")
print(f"Peak frequency: {result.peak_frequency:.1f} Hz")

# Coherence matrix (all pairs)
coh_matrix = conn.coherence_matrix(data, freq_band=(8, 13))  # Alpha band

# Phase Locking Value
plv = conn.phase_locking_value(signal1, signal2, freq_band=(8, 13))
```

### Session Logging

```python
from src.core.session_logging import SessionLogger

logger = SessionLogger(output_dir="./recordings")

# Start session
session_id = logger.start_session(
    source="simulator",
    sample_rate=256,
    n_channels=21,
)

# Add events during recording
logger.add_event("stimulus", "Visual flash presented")
logger.add_event("artifact", "Eye blink detected", {"channels": [0, 1]})

# Record performance
logger.record_latency("filter_latency_ms", 2.3)

# End session (saves JSON with all metadata)
json_file = logger.end_session()
```

## 10-20 Electrode System

The dashboard maps the 25-pin connector to standard 10-20 positions:

| Region | Electrodes |
|--------|------------|
| Frontal | Fp1, Fp2, F7, F3, Fz, F4, F8 |
| Temporal | T3, T4, T5, T6 |
| Central | C3, Cz, C4 |
| Parietal | P3, Pz, P4 |
| Occipital | O1, Oz, O2 |
| Reference | A1, A2 |

## Frequency Bands

| Band | Frequency | Associated State |
|------|-----------|------------------|
| Delta | 0.5-4 Hz | Deep sleep |
| Theta | 4-8 Hz | Drowsiness, meditation |
| Alpha | 8-13 Hz | Relaxed, eyes closed |
| Beta | 13-30 Hz | Active thinking, focus |
| Gamma | 30-100 Hz | High-level processing |

## Data Formats

### CSV Export
Human-readable format with timestamps and channel names:
```csv
# EEG Recording
# Sample Rate: 256 Hz
timestamp,Fp1,Fp2,F7,F3,...
0.000,12.5,-8.3,5.2,10.1,...
0.004,11.8,-7.9,4.8,9.6,...
```

### EDF Export  
European Data Format - standard for EEG, compatible with:
- EEGLAB
- MNE-Python
- BrainVision Analyzer
- EDFbrowser

## Performance

### Benchmarks (tested on Intel i7, 16GB RAM)

| Operation | Data Size | Time | Throughput |
|-----------|-----------|------|------------|
| Filter (batch) | 10s × 21ch | <100ms | >100× real-time |
| Filter (streaming) | 1 sample | <0.5ms | 256 Hz capable |
| Spectral analysis | 10s × 21ch | <200ms | >50× real-time |
| Topomap render | 150×150 | <30ms | 30+ FPS |
| 3D brain render | 40 res | <50ms | 20+ FPS |

### Optimization Techniques

1. **Vectorized operations** - NumPy broadcasting for multi-channel processing
2. **Pre-allocated buffers** - Ring buffers avoid memory allocation in hot paths
3. **Cached coefficients** - Filter coefficients computed once, reused
4. **SOS filters** - Second-order sections for numerical stability
5. **Frame rate limiting** - Visualization skips redundant renders
6. **FFT-optimal sizes** - Uses `next_fast_len()` for efficient FFTs

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_suite.py::TestFilters -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip performance)
pytest tests/ -v -m "not slow"
```

### Test Categories

- **Unit tests** - Individual component testing
- **Integration tests** - Full pipeline testing
- **Performance tests** - Benchmark assertions

## Troubleshooting

### Common Issues

**No serial ports detected**
```bash
# List available ports
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
```

**GUI doesn't launch (Linux)**
```bash
# Install Qt dependencies
sudo apt-get install libxcb-xinerama0 libegl1
```

**High CPU usage**
- Reduce `display.refresh_rate` in config
- Use "Simple" topomap style instead of "PET Scan"
- Disable 3D auto-rotation

**Filter instability**
- Ensure sample rate matches your hardware
- Use `OptimizedEEGFilter` which uses SOS format

## Development

### Code Style
```bash
pip install black flake8
black src/
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

### Code Style
- Follow PEP 8
- Use type hints
- Document public functions with docstrings
- Add tests for new features

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- [MNE-Python](https://mne.tools/) - EEG processing inspiration
- [PyQtGraph](https://www.pyqtgraph.org/) - Real-time plotting
- 10-20 International System - Electrode placement standard

## Changelog

### v0.2.0 (Current)
- ✨ Optimized signal processing (2-5× faster)
- ✨ Adaptive artifact detection
- ✨ Connectivity analysis (coherence, PLV)
- ✨ Real-time spectrogram
- ✨ Basic ICA for artifact removal
- ✨ Session logging with event markers
- ✨ Comprehensive test suite
- 🐛 Fixed filter numerical stability
- 📚 Updated documentation

### v0.1.0
- Initial release
- Basic waveform display
- 2D/3D brain visualization
- Simulator mode
- CSV/EDF export
