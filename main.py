#!/usr/bin/env python3
"""
EEG Dashboard - Main Entry Point

A real-time EEG visualization and analysis dashboard.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_dashboard():
    """Launch the PyQt6 dashboard GUI."""
    from PyQt6.QtWidgets import QApplication
    from ui.dashboard import EEGDashboard
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Consistent cross-platform look
    
    dashboard = EEGDashboard()
    dashboard.show()
    
    sys.exit(app.exec())


def run_test():
    """Run quick test of components without GUI."""
    print("=" * 50)
    print("EEG Dashboard v0.1.0 - Component Test")
    print("=" * 50)
    print()
    
    # Test simulator
    print("1. Testing EEG simulator...")
    from hardware.simulator import generate_test_data
    
    data = generate_test_data(duration=1.0, sample_rate=256, n_channels=21)
    print(f"   ✓ Generated {data.shape[0]} samples x {data.shape[1]} channels")
    print(f"   ✓ Signal range: {data.min():.1f} to {data.max():.1f} µV")
    print()
    
    # Test channel map
    print("2. Testing channel mapping...")
    from core.channel_map import ChannelMap
    
    channel_map = ChannelMap()
    print(f"   ✓ Loaded {len(channel_map.electrodes)} electrodes")
    regions = channel_map.get_regions()
    for region, channels in regions.items():
        print(f"     {region}: {', '.join(channels)}")
    print()
    
    # Test filters
    print("3. Testing signal processing...")
    from processing.filters import EEGFilter, FilterSettings
    import numpy as np
    
    settings = FilterSettings(sample_rate=256)
    eeg_filter = EEGFilter(settings)
    test_signal = np.random.randn(256, 21) * 50
    filtered = eeg_filter.filter_chunk(test_signal)
    print(f"   ✓ Filter processed {len(filtered)} samples")
    print()
    
    # Test spectral analysis  
    print("4. Testing spectral analysis...")
    from processing.spectral import SpectralAnalyzer
    
    # Need more data for spectral analysis (at least 2 seconds)
    test_data = generate_test_data(duration=3.0, sample_rate=256, n_channels=21)
    analyzer = SpectralAnalyzer(sample_rate=256, window_size=1.0)
    band_powers = analyzer.compute_band_power(test_data[:, 0])
    print(f"   ✓ Band powers computed:")
    print(f"     Delta: {band_powers.delta:.1f}%")
    print(f"     Theta: {band_powers.theta:.1f}%")
    print(f"     Alpha: {band_powers.alpha:.1f}%")
    print(f"     Beta:  {band_powers.beta:.1f}%")
    print(f"     Gamma: {band_powers.gamma:.1f}%")
    print()
    
    # Test data export
    print("5. Testing data export...")
    from core.data_export import EEGRecorder
    
    recorder = EEGRecorder(channel_map.get_channel_names(), sample_rate=256)
    recorder.start()
    for sample in data:
        recorder.add_sample(sample)
    recorder.stop()
    print(f"   ✓ Recorded {recorder.get_duration():.1f} seconds of data")
    print()
    
    print("=" * 50)
    print("All tests passed! Ready to run dashboard.")
    print("Run: python main.py")
    print("=" * 50)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='EEG Dashboard - Real-time EEG visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py           Launch the dashboard GUI
  python main.py --test    Run component tests (no GUI)
  python main.py --help    Show this help message
        """
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run component tests without launching GUI'
    )
    parser.add_argument(
        '--simulator', '-s',
        action='store_true',
        help='Start with simulator mode (default)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_test()
    else:
        print("Launching EEG Dashboard...")
        print("(Use --test flag to run component tests without GUI)")
        print()
        run_dashboard()


if __name__ == "__main__":
    main()
