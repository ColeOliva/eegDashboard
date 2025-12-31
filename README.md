# EEG Dashboard

A real-time EEG visualization and analysis dashboard for 25-pin connector EEG caps via USB adapters.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

### Visualization
- **Real-time waveform display** - Scrolling EEG traces for all channels
- **2D Brain topographic map** - Scalp heatmap with anatomical brain overlay
- **3D Brain visualization** - Rotating 3D brain surface with activity mapping
- **Anatomical brain rendering** - MRI-style brain structure (white matter, gray matter, ventricles)
- **Custom MRI support** - Load your own MRI/CT scan as the brain overlay
- **Multiple colormaps** - PET scan (rainbow), Hot Metal, or simple styles

### Analysis
- **Frequency band analysis** - Delta, Theta, Alpha, Beta, Gamma power bars
- **Cognitive indices** - Real-time engagement and relaxation metrics
- **10-20 electrode system** - Standard international EEG placement
- **Signal processing** - Bandpass filter, notch filter, artifact rejection

### Data & Recording
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

### Dashboard Tabs

1. **Waveforms** - Real-time scrolling EEG traces for all channels
2. **2D Brain Map** - Topographic brain activity with anatomical overlay
3. **3D Brain** - Rotating 3D brain surface visualization
4. **Multi-View** - Combined view of all frequency bands

### Brain Map Controls

- **Style selector** - Choose visualization style:
  - *PET Scan* - Rainbow colormap with anatomical brain structure
  - *Hot Metal* - Medical thermal imaging style
  - *Simple* - Clean, minimal heatmap
- **Band selector** - Display Alpha, Beta, Theta, Delta, or Gamma activity
- **Load MRI** - Import your own MRI/CT scan image as the brain overlay

### 3D Brain Controls

- **View selector** - Isometric, Top, Front, Left, Right, Back views
- **Auto-Rotate** - Enable continuous rotation
- **Band selector** - Choose which frequency band to visualize

### Recording & Export

1. **Connect** - Start data acquisition from simulator or hardware
2. **Record** - Begin recording EEG data  
3. **Save** - Export recording to CSV or EDF format

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
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── config/
│   └── default_config.yaml      # Configuration file
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
│   │   ├── topomap.py           # Basic brain heatmap
│   │   ├── topomap_enhanced.py  # Enhanced 2D topomap
│   │   ├── brain_3d.py          # 3D brain visualization
│   │   └── brain_anatomical.py  # Anatomical brain + MRI support
│   └── ui/
│       └── dashboard.py         # Main GUI window (tabbed interface)
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
- **pyqtgraph** - Fast real-time plotting
- **numpy** - Numerical computing
- **scipy** - Signal processing & interpolation
- **matplotlib** - Topographic maps & brain visualization
- **Pillow** - Custom MRI image loading
- **pyserial** - Serial communication
- **pyedflib** - EDF file format
- **mne** - EEG analysis toolkit

## Custom MRI Support

You can load your own MRI or CT scan image as the brain overlay:

1. Click the **"Load MRI"** button in the 2D Brain Map tab
2. Select your image file (PNG, JPEG, TIFF, or BMP)
3. The activity heatmap will overlay on your custom brain image

**Best results with:**
- Axial (top-down) MRI T1/T2 slices
- CT scans showing brain structure
- Images with good contrast between brain and background

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
