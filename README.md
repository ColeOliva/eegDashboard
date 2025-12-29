# EEG Dashboard

A real-time EEG visualization and analysis dashboard for 25-pin connector EEG caps via USB adapters.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Real-time waveform display** - Scrolling EEG traces for all channels
- **Brain topographic map** - 2D scalp heatmap showing activity distribution  
- **Frequency band analysis** - Delta, Theta, Alpha, Beta, Gamma power
- **10-20 electrode system** - Standard international EEG placement
- **Signal processing** - Bandpass filter, notch filter, artifact rejection
- **Data recording** - Save to CSV, EDF, or NumPy formats
- **Simulation mode** - Test without hardware using realistic synthetic EEG

## Screenshots

*Coming soon - Dashboard with waveforms, topomap, and band power display*

## Installation

### Prerequisites

- Python 3.10 or higher
- USB-to-Serial adapter for your EEG cap

### Setup

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

## Quick Start

### Run the Dashboard

```bash
python main.py
```

The dashboard starts in **Simulator mode** by default - no hardware needed!

### Test Components (No GUI)

```bash
python main.py --test
```

This runs all component tests without launching the graphical interface.

## Usage

### Dashboard Controls

1. **Source Selection** - Choose between Simulator or Serial port
2. **Connect/Disconnect** - Start/stop data acquisition  
3. **Record** - Begin recording EEG data
4. **Save** - Export recording to CSV or EDF format

### With Your EEG Hardware

1. Connect your EEG cap to the USB adapter
2. Launch the dashboard: `python main.py`
3. Select your serial port from the dropdown
4. Click **Connect** to start acquisition

### Configuration

Edit `config/default_config.yaml` to customize:

```yaml
hardware:
  port: "auto"           # Serial port or "auto" for detection
  baud_rate: 115200      # Adjust for your device

processing:
  sample_rate: 256       # Hz
  notch_freq: 60         # 60 Hz (US) or 50 Hz (EU)
  bandpass: [0.5, 50]    # Filter range in Hz
```

## Project Structure

```
eegDashboard/
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
├── config/
│   └── default_config.yaml    # Configuration file
├── src/
│   ├── hardware/
│   │   ├── serial_interface.py  # USB/serial communication
│   │   └── simulator.py         # Synthetic EEG generator
│   ├── processing/
│   │   ├── filters.py           # Signal filtering
│   │   └── spectral.py          # FFT & band power
│   ├── core/
│   │   ├── channel_map.py       # 10-20 electrode positions
│   │   └── data_export.py       # CSV/EDF export
│   ├── visualization/
│   │   ├── waveform_plot.py     # Scrolling traces
│   │   └── topomap.py           # Brain heatmap
│   └── ui/
│       └── dashboard.py         # Main GUI window
└── tests/                       # Unit tests
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

## Dependencies

- **PyQt6** - GUI framework
- **pyqtgraph** - Fast plotting
- **numpy** - Numerical computing
- **scipy** - Signal processing
- **matplotlib** - Topographic maps
- **pyserial** - Serial communication
- **pyedflib** - EDF file format
- **mne** - EEG analysis toolkit

## Troubleshooting

### GUI doesn't launch (Linux)
```bash
# Install Qt dependencies
sudo apt-get install libxcb-xinerama0 libegl1
```

### Serial port not detected
- Check your USB adapter is connected
- Try running with elevated privileges
- Verify the correct driver is installed

### High CPU usage
- Reduce display refresh rate in settings
- Decrease the number of visible channels

## Development

### Run Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
pip install black flake8
black src/
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- [MNE-Python](https://mne.tools/) - EEG analysis inspiration
- [10-20 System](https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)) - Electrode placement standard
- [PyQtGraph](https://pyqtgraph.readthedocs.io/) - Real-time plotting
