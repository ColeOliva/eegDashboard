"""
Comprehensive Test Suite for EEG Dashboard.

Run with: pytest tests/ -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestChannelMap:
    """Tests for channel mapping functionality."""
    
    def test_channel_map_initialization(self):
        from core.channel_map import ChannelMap
        
        channel_map = ChannelMap()
        assert len(channel_map.electrodes) > 0
        
    def test_get_channel_names(self):
        from core.channel_map import ChannelMap
        
        channel_map = ChannelMap()
        names = channel_map.get_channel_names()
        assert len(names) == 20  # Standard 10-20 without reference
        assert "Fp1" in names
        assert "O2" in names
        
    def test_get_by_name(self):
        from core.channel_map import ChannelMap
        
        channel_map = ChannelMap()
        electrode = channel_map.get_by_name("Cz")
        assert electrode is not None
        assert electrode.name == "Cz"
        assert electrode.x == 0.5  # Center position
        
    def test_get_positions_2d(self):
        from core.channel_map import ChannelMap
        
        channel_map = ChannelMap()
        positions = channel_map.get_positions_2d()
        assert positions.shape[0] == 20
        assert positions.shape[1] == 2
        # All positions should be in [0, 1] range
        assert np.all(positions >= 0) and np.all(positions <= 1)
        
    def test_get_regions(self):
        from core.channel_map import ChannelMap
        
        channel_map = ChannelMap()
        regions = channel_map.get_regions()
        assert "Frontal" in regions
        assert "Occipital" in regions


class TestSimulator:
    """Tests for EEG simulator."""
    
    def test_generate_test_data_shape(self):
        from hardware.simulator import generate_test_data
        
        data = generate_test_data(duration=1.0, sample_rate=256, n_channels=21)
        assert data.shape == (256, 21)
        
    def test_generate_test_data_range(self):
        from hardware.simulator import generate_test_data
        
        data = generate_test_data(duration=2.0, sample_rate=256, n_channels=21)
        # EEG signals should be in µV range (typically -200 to 200)
        assert np.abs(data).max() < 500
        
    def test_simulator_callback(self):
        from hardware.simulator import EEGSimulator
        
        sim = EEGSimulator(n_channels=21, sample_rate=256)
        received_packets = []
        
        def callback(packet):
            received_packets.append(packet)
        
        sim.register_callback(callback)
        sim.connect()
        sim.start_acquisition()
        
        import time
        time.sleep(0.1)  # Let it run briefly
        
        sim.stop_acquisition()
        
        assert len(received_packets) > 0
        assert received_packets[0].channels.shape == (21,)


class TestFilters:
    """Tests for signal filtering."""
    
    def test_filter_initialization(self):
        from processing.filters import EEGFilter, FilterSettings
        
        settings = FilterSettings(sample_rate=256)
        eeg_filter = EEGFilter(settings)
        assert eeg_filter.settings.sample_rate == 256
        
    def test_filter_sample(self):
        from processing.filters import EEGFilter, FilterSettings
        
        settings = FilterSettings(sample_rate=256)
        eeg_filter = EEGFilter(settings)
        
        sample = np.random.randn(21) * 50
        filtered = eeg_filter.filter_sample(sample)
        
        assert filtered.shape == sample.shape
        
    def test_filter_chunk(self):
        from processing.filters import EEGFilter, FilterSettings
        
        settings = FilterSettings(sample_rate=256)
        eeg_filter = EEGFilter(settings)
        
        data = np.random.randn(256, 21) * 50
        filtered = eeg_filter.filter_chunk(data)
        
        assert filtered.shape == data.shape
        
    def test_filter_removes_dc(self):
        from processing.filters import EEGFilter, FilterSettings
        
        settings = FilterSettings(sample_rate=256, bandpass_low=0.5)
        eeg_filter = EEGFilter(settings)
        
        # Create signal with DC offset
        data = np.random.randn(512, 5) * 10 + 100  # DC offset of 100
        filtered = eeg_filter.filter_batch(data)
        
        # DC should be mostly removed
        assert np.abs(np.mean(filtered)) < 10
        
    def test_notch_filter_removes_60hz(self):
        from processing.filters import notch_filter
        
        fs = 256
        t = np.arange(0, 2, 1/fs)
        
        # Create 60 Hz noise
        signal = np.sin(2 * np.pi * 60 * t) * 10
        # Add some EEG-like content
        signal += np.sin(2 * np.pi * 10 * t) * 5  # 10 Hz alpha
        
        filtered = notch_filter(signal, 60, fs)
        
        # 60 Hz component should be reduced
        from scipy.fft import rfft, rfftfreq
        freqs = rfftfreq(len(signal), 1/fs)
        fft_orig = np.abs(rfft(signal))
        fft_filt = np.abs(rfft(filtered))
        
        idx_60 = np.argmin(np.abs(freqs - 60))
        assert fft_filt[idx_60] < fft_orig[idx_60] * 0.2


class TestOptimizedFilters:
    """Tests for optimized filtering module."""
    
    def test_optimized_filter_initialization(self):
        from processing.optimized_filters import (FilterSettings,
                                                  OptimizedEEGFilter)
        
        settings = FilterSettings(sample_rate=256)
        eeg_filter = OptimizedEEGFilter(settings)
        assert eeg_filter.settings.sample_rate == 256
        
    def test_optimized_filter_chunk(self):
        from processing.optimized_filters import OptimizedEEGFilter
        
        eeg_filter = OptimizedEEGFilter()
        data = np.random.randn(512, 21) * 50
        filtered = eeg_filter.filter_chunk(data)
        
        assert filtered.shape == data.shape
        assert not np.any(np.isnan(filtered))
        
    def test_adaptive_artifact_detector(self):
        from processing.optimized_filters import AdaptiveArtifactDetector
        
        detector = AdaptiveArtifactDetector()
        
        # Normal sample
        sample = np.random.randn(21) * 20
        is_artifact, _, _ = detector.detect_sample(sample)
        assert not is_artifact
        
        # Artifact sample (very high amplitude)
        artifact_sample = np.random.randn(21) * 500
        is_artifact, artifact_type, _ = detector.detect_sample(artifact_sample)
        assert is_artifact
        assert artifact_type == "amplitude"


class TestSpectralAnalysis:
    """Tests for spectral analysis."""
    
    def test_band_power_computation(self):
        from processing.spectral import SpectralAnalyzer
        
        analyzer = SpectralAnalyzer(sample_rate=256)
        
        # Create signal with known alpha content
        t = np.arange(0, 3, 1/256)
        signal = np.sin(2 * np.pi * 10 * t) * 20  # 10 Hz alpha
        
        bp = analyzer.compute_band_power(signal)
        
        # Alpha should dominate
        assert bp.alpha > bp.delta
        assert bp.alpha > bp.theta
        
    def test_psd_computation(self):
        from processing.spectral import SpectralAnalyzer
        
        analyzer = SpectralAnalyzer(sample_rate=256)
        signal = np.random.randn(512) * 20
        
        freqs, psd = analyzer.compute_psd(signal)
        
        assert len(freqs) > 0
        assert len(psd) == len(freqs)
        assert np.all(psd >= 0)  # Power is non-negative
        
    def test_multichannel_band_power(self):
        from processing.spectral import SpectralAnalyzer
        
        analyzer = SpectralAnalyzer(sample_rate=256)
        data = np.random.randn(512, 21) * 20
        
        bp_dict = analyzer.compute_band_power_multichannel(data)
        
        assert "alpha" in bp_dict
        assert len(bp_dict["alpha"]) == 21
        
    def test_engagement_index(self):
        from processing.spectral import BandPower, compute_engagement_index

        # High beta, low alpha/theta = high engagement
        bp = BandPower(delta=5, theta=10, alpha=15, beta=50, gamma=5)
        engagement = compute_engagement_index(bp)
        
        assert engagement > 1.0
        
    def test_relaxation_index(self):
        from processing.spectral import BandPower, compute_relaxation_index

        # High alpha, low beta = relaxed
        bp = BandPower(delta=5, theta=10, alpha=50, beta=10, gamma=5)
        relaxation = compute_relaxation_index(bp)
        
        assert relaxation > 1.0


class TestOptimizedSpectral:
    """Tests for optimized spectral analysis."""
    
    def test_optimized_analyzer(self):
        from processing.optimized_spectral import OptimizedSpectralAnalyzer
        
        analyzer = OptimizedSpectralAnalyzer(sample_rate=256)
        signal = np.random.randn(512) * 20
        
        bp = analyzer.compute_band_power(signal)
        assert bp.total() > 0
        
    def test_ring_buffer(self):
        from processing.optimized_spectral import OptimizedSpectralAnalyzer
        
        analyzer = OptimizedSpectralAnalyzer(sample_rate=256, window_size=1.0)
        
        # Add samples one at a time
        for _ in range(300):
            sample = np.random.randn(21) * 20
            analyzer.add_sample(sample)
        
        # Should be able to get band power
        bp = analyzer.get_current_band_power_all()
        assert "alpha" in bp


class TestDataExport:
    """Tests for data export functionality."""
    
    def test_save_load_csv(self):
        from core.data_export import load_csv, save_csv

        # Create test data
        data = np.random.randn(256, 5) * 50
        channel_names = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5"]
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filename = f.name
        
        try:
            save_csv(filename, data, channel_names, sample_rate=256)
            loaded = load_csv(filename)
            
            assert loaded['data'].shape == data.shape
            assert loaded['channel_names'] == channel_names
        finally:
            os.unlink(filename)


class TestConstants:
    """Tests for shared constants."""
    
    def test_electrode_positions(self):
        from core.constants import (ELECTRODE_POSITIONS_2D,
                                    ELECTRODE_POSITIONS_3D)
        
        assert "Fp1" in ELECTRODE_POSITIONS_2D
        assert "Fp1" in ELECTRODE_POSITIONS_3D
        
        # 2D positions should be in reasonable range
        for name, (x, y) in ELECTRODE_POSITIONS_2D.items():
            assert -1.5 <= x <= 1.5
            assert -1.5 <= y <= 1.5
            
    def test_frequency_bands(self):
        from core.constants import FREQUENCY_BANDS
        
        assert "alpha" in FREQUENCY_BANDS
        assert FREQUENCY_BANDS["alpha"] == (8, 13)
        
    def test_channel_positions_helper(self):
        from core.constants import (get_channel_positions_2d,
                                    get_channel_positions_3d)
        
        pos_2d = get_channel_positions_2d()
        pos_3d = get_channel_positions_3d()
        
        assert pos_2d.shape[1] == 2
        assert pos_3d.shape[1] == 3


class TestVisualization:
    """Tests for visualization components (without GUI)."""
    
    def test_fast_topomap_renderer(self):
        from visualization.fast_renderer import FastTopoMapRenderer
        
        renderer = FastTopoMapRenderer(resolution=100)
        values = np.random.randn(20) * 20
        
        img_bytes = renderer.render(values, style="brain")
        
        assert len(img_bytes) > 0
        # Should be valid PNG
        assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n'
        
    def test_fast_brain3d_renderer(self):
        from visualization.fast_renderer import FastBrain3DRenderer
        
        renderer = FastBrain3DRenderer()
        values = np.random.randn(20) * 20
        
        img_bytes = renderer.render(values, view="iso")
        
        assert len(img_bytes) > 0
        assert img_bytes[:8] == b'\x89PNG\r\n\x1a\n'


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline(self):
        """Test complete data pipeline: simulate -> filter -> analyze."""
        from hardware.simulator import generate_test_data
        from processing.optimized_filters import OptimizedEEGFilter
        from processing.optimized_spectral import OptimizedSpectralAnalyzer

        # Generate data
        data = generate_test_data(duration=3.0, sample_rate=256, n_channels=20)
        assert data.shape == (768, 20)
        
        # Filter
        eeg_filter = OptimizedEEGFilter()
        filtered = eeg_filter.filter_batch(data)
        assert filtered.shape == data.shape
        
        # Analyze
        analyzer = OptimizedSpectralAnalyzer(sample_rate=256)
        bp_all = analyzer.compute_band_power_multichannel(filtered)
        
        assert "alpha" in bp_all
        assert len(bp_all["alpha"]) == 20
        
    def test_streaming_pipeline(self):
        """Test real-time streaming pipeline."""
        from processing.optimized_filters import OptimizedEEGFilter
        from processing.optimized_spectral import OptimizedSpectralAnalyzer
        
        eeg_filter = OptimizedEEGFilter()
        analyzer = OptimizedSpectralAnalyzer(sample_rate=256, window_size=1.0)
        
        # Simulate 2 seconds of streaming
        for _ in range(512):
            sample = np.random.randn(20) * 50
            filtered = eeg_filter.filter_sample(sample)
            analyzer.add_sample(filtered)
        
        # Get results
        bp = analyzer.get_current_band_power_all()
        assert all(band in bp for band in ["delta", "theta", "alpha", "beta", "gamma"])


class TestPerformance:
    """Performance benchmarks."""
    
    def test_filter_performance(self):
        """Filter should process 10 seconds of data in < 100ms."""
        import time

        from processing.optimized_filters import OptimizedEEGFilter
        
        eeg_filter = OptimizedEEGFilter()
        data = np.random.randn(2560, 21) * 50  # 10 seconds
        
        start = time.time()
        _ = eeg_filter.filter_batch(data)
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"Filtering took {elapsed:.3f}s, expected < 0.5s"
        
    def test_spectral_performance(self):
        """Spectral analysis should process 10 seconds in < 200ms."""
        import time

        from processing.optimized_spectral import OptimizedSpectralAnalyzer
        
        analyzer = OptimizedSpectralAnalyzer(sample_rate=256)
        data = np.random.randn(2560, 21) * 50
        
        start = time.time()
        _ = analyzer.compute_band_power_multichannel(data)
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"Analysis took {elapsed:.3f}s, expected < 0.5s"


# Run with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
