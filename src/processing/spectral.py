"""
Spectral Analysis Module.

Implements frequency-domain analysis for EEG signals:
- FFT computation
- Power spectral density (PSD)
- Band power extraction (delta, theta, alpha, beta, gamma)
- Real-time spectrum updates
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from collections import deque


# Standard EEG frequency bands (in Hz)
FREQUENCY_BANDS = {
    "delta": (0.5, 4),     # Deep sleep, unconscious processes
    "theta": (4, 8),       # Drowsiness, light sleep, meditation, memory
    "alpha": (8, 13),      # Relaxed, calm, eyes closed
    "beta": (13, 30),      # Active thinking, focus, alertness
    "gamma": (30, 100),    # Higher cognitive functions, perception
}

# More detailed sub-bands for research
FREQUENCY_BANDS_DETAILED = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "low_alpha": (8, 10),      # Relaxed attention
    "high_alpha": (10, 13),    # Active relaxation
    "low_beta": (13, 20),      # Relaxed focus
    "high_beta": (20, 30),     # Active concentration, anxiety
    "low_gamma": (30, 50),     # Cognitive processing
    "high_gamma": (50, 100),   # High-level cognition (often noisy)
}

# Band colors for visualization
BAND_COLORS = {
    "delta": "#8B5CF6",    # Purple
    "theta": "#06B6D4",    # Cyan
    "alpha": "#10B981",    # Green
    "beta": "#F59E0B",     # Orange
    "gamma": "#EF4444",    # Red
}


@dataclass
class BandPower:
    """Power values for each frequency band."""
    delta: float
    theta: float
    alpha: float
    beta: float
    gamma: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "theta": self.theta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }
    
    def total(self) -> float:
        """Total power across all bands."""
        return self.delta + self.theta + self.alpha + self.beta + self.gamma
    
    def relative(self) -> Dict[str, float]:
        """Get relative power (percentage) for each band."""
        total = self.total()
        if total == 0:
            return {k: 0.0 for k in self.to_dict()}
        return {k: v / total * 100 for k, v in self.to_dict().items()}


class SpectralAnalyzer:
    """
    Computes frequency-domain features from EEG signals.
    
    Supports both batch processing and real-time analysis
    with sliding windows.
    """
    
    def __init__(
        self,
        sample_rate: float = 256.0,
        window_size: float = 2.0,
        overlap: float = 0.5,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the spectral analyzer.
        
        Args:
            sample_rate: Sampling rate in Hz.
            window_size: Analysis window size in seconds.
            overlap: Window overlap ratio (0-1).
            bands: Custom frequency bands dict {name: (low, high)}.
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.bands = bands or FREQUENCY_BANDS
        
        # Calculate window parameters
        self.n_samples = int(window_size * sample_rate)
        self.hop_size = int(self.n_samples * (1 - overlap))
        
        # Windowing function (Hanning reduces spectral leakage)
        self.window = signal.windows.hann(self.n_samples)
        
        # Frequency axis for FFT
        self.freqs = rfftfreq(self.n_samples, 1 / sample_rate)
        
        # Buffer for real-time processing
        self._buffer: Optional[deque] = None
        self._n_channels: Optional[int] = None
        
    def _init_buffer(self, n_channels: int):
        """Initialize buffer for streaming data."""
        if self._n_channels != n_channels:
            self._n_channels = n_channels
            self._buffer = deque(maxlen=self.n_samples)
            
    def compute_psd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.
        
        Args:
            data: Array of shape (n_samples,) or (n_samples, n_channels).
            
        Returns:
            Tuple of (frequencies, psd) arrays.
        """
        if data.ndim == 1:
            freqs, psd = signal.welch(
                data,
                fs=self.sample_rate,
                window='hann',
                nperseg=self.n_samples,
                noverlap=int(self.n_samples * self.overlap),
            )
        else:
            # Multi-channel: compute PSD for each channel
            n_channels = data.shape[1]
            freqs, psd_ch0 = signal.welch(
                data[:, 0],
                fs=self.sample_rate,
                window='hann',
                nperseg=min(self.n_samples, len(data)),
                noverlap=int(min(self.n_samples, len(data)) * self.overlap),
            )
            
            psd = np.zeros((len(freqs), n_channels))
            psd[:, 0] = psd_ch0
            
            for ch in range(1, n_channels):
                _, psd[:, ch] = signal.welch(
                    data[:, ch],
                    fs=self.sample_rate,
                    window='hann',
                    nperseg=min(self.n_samples, len(data)),
                    noverlap=int(min(self.n_samples, len(data)) * self.overlap),
                )
        
        return freqs, psd
    
    def compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT magnitude spectrum.
        
        Args:
            data: Array of shape (n_samples,) for single channel.
            
        Returns:
            Tuple of (frequencies, magnitudes) arrays.
        """
        # Apply window
        if len(data) >= self.n_samples:
            windowed = data[-self.n_samples:] * self.window
        else:
            # Pad with zeros if not enough data
            padded = np.zeros(self.n_samples)
            padded[-len(data):] = data
            windowed = padded * self.window
        
        # Compute FFT
        fft_result = rfft(windowed)
        magnitudes = np.abs(fft_result) * 2 / self.n_samples
        
        return self.freqs, magnitudes
    
    def compute_band_power(
        self,
        data: np.ndarray,
        method: str = "welch"
    ) -> BandPower:
        """
        Compute power in each frequency band.
        
        Args:
            data: Array of shape (n_samples,) for single channel.
            method: "welch" for PSD or "fft" for simple FFT.
            
        Returns:
            BandPower dataclass with power values.
        """
        if method == "welch":
            freqs, psd = self.compute_psd(data)
        else:
            freqs, magnitudes = self.compute_fft(data)
            psd = magnitudes ** 2  # Power = magnitude^2
        
        # Calculate band powers by integrating PSD
        band_powers = {}
        for band_name, (low, high) in self.bands.items():
            # Find frequency indices in band
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if len(idx) > 0:
                # Integrate power in band (sum of PSD values * frequency resolution)
                freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1
                band_powers[band_name] = np.sum(psd[idx]) * freq_res
            else:
                band_powers[band_name] = 0.0
        
        return BandPower(
            delta=band_powers.get("delta", 0),
            theta=band_powers.get("theta", 0),
            alpha=band_powers.get("alpha", 0),
            beta=band_powers.get("beta", 0),
            gamma=band_powers.get("gamma", 0),
        )
    
    def compute_band_power_multichannel(
        self,
        data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute band power for all channels.
        
        Args:
            data: Array of shape (n_samples, n_channels).
            
        Returns:
            Dict mapping band name to array of powers per channel.
        """
        n_channels = data.shape[1]
        freqs, psd = self.compute_psd(data)
        
        result = {band: np.zeros(n_channels) for band in self.bands}
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1
        
        for band_name, (low, high) in self.bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if len(idx) > 0:
                result[band_name] = np.sum(psd[idx, :], axis=0) * freq_res
        
        return result
    
    def add_sample(self, sample: np.ndarray):
        """
        Add a sample to the streaming buffer.
        
        Args:
            sample: Array of shape (n_channels,).
        """
        self._init_buffer(len(sample))
        self._buffer.append(sample.copy())
        
    def add_chunk(self, data: np.ndarray):
        """
        Add a chunk of data to the streaming buffer.
        
        Args:
            data: Array of shape (n_samples, n_channels).
        """
        self._init_buffer(data.shape[1])
        for row in data:
            self._buffer.append(row.copy())
    
    def get_current_spectrum(self, channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current spectrum from buffer for one channel.
        
        Returns:
            Tuple of (frequencies, magnitudes).
        """
        if self._buffer is None or len(self._buffer) < 10:
            return self.freqs, np.zeros(len(self.freqs))
        
        data = np.array(self._buffer)[:, channel]
        return self.compute_fft(data)
    
    def get_current_band_power(self, channel: int = 0) -> BandPower:
        """
        Get current band power from buffer for one channel.
        
        Returns:
            BandPower with current values.
        """
        if self._buffer is None or len(self._buffer) < 10:
            return BandPower(0, 0, 0, 0, 0)
        
        data = np.array(self._buffer)[:, channel]
        return self.compute_band_power(data)
    
    def get_current_band_power_all(self) -> Dict[str, np.ndarray]:
        """
        Get current band power for all channels from buffer.
        
        Returns:
            Dict mapping band name to array of powers per channel.
        """
        if self._buffer is None or len(self._buffer) < 10:
            n_ch = self._n_channels or 1
            return {band: np.zeros(n_ch) for band in self.bands}
        
        data = np.array(self._buffer)
        return self.compute_band_power_multichannel(data)
    
    def clear_buffer(self):
        """Clear the streaming buffer."""
        if self._buffer:
            self._buffer.clear()


class BandPowerTracker:
    """
    Tracks band power over time for trend analysis.
    
    Useful for detecting changes in brain state over
    longer time periods.
    """
    
    def __init__(
        self,
        history_seconds: float = 60.0,
        update_rate: float = 4.0,  # Updates per second
    ):
        """
        Initialize the tracker.
        
        Args:
            history_seconds: How much history to keep.
            update_rate: Expected updates per second.
        """
        self.history_size = int(history_seconds * update_rate)
        
        self.history = {
            band: deque(maxlen=self.history_size)
            for band in FREQUENCY_BANDS
        }
        self.timestamps = deque(maxlen=self.history_size)
        
    def add(self, band_power: BandPower, timestamp: Optional[float] = None):
        """Add a band power measurement."""
        import time
        
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        
        for band, value in band_power.to_dict().items():
            self.history[band].append(value)
    
    def get_history(self, band: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time series for a band.
        
        Returns:
            Tuple of (timestamps, values).
        """
        return np.array(self.timestamps), np.array(self.history[band])
    
    def get_trend(self, band: str, window: int = 10) -> float:
        """
        Get recent trend (slope) for a band.
        
        Positive = increasing, negative = decreasing.
        """
        if len(self.history[band]) < window:
            return 0.0
        
        values = np.array(list(self.history[band])[-window:])
        x = np.arange(len(values))
        
        # Linear regression slope
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def get_average(self, band: str, window: int = 10) -> float:
        """Get recent average for a band."""
        if len(self.history[band]) == 0:
            return 0.0
        
        values = list(self.history[band])[-window:]
        return np.mean(values)


# Convenience functions

def compute_alpha_peak(
    data: np.ndarray,
    sample_rate: float = 256.0
) -> Tuple[float, float]:
    """
    Find the peak alpha frequency (Individual Alpha Frequency - IAF).
    
    The IAF is a biomarker that varies between individuals (~8-12 Hz).
    
    Args:
        data: EEG data array.
        sample_rate: Sampling rate in Hz.
        
    Returns:
        Tuple of (peak_frequency, peak_power).
    """
    analyzer = SpectralAnalyzer(sample_rate=sample_rate)
    freqs, psd = analyzer.compute_psd(data)
    
    # Find alpha range
    alpha_low, alpha_high = FREQUENCY_BANDS["alpha"]
    alpha_idx = np.where((freqs >= alpha_low) & (freqs <= alpha_high))[0]
    
    if len(alpha_idx) == 0:
        return 0.0, 0.0
    
    # Find peak in alpha range
    if psd.ndim > 1:
        psd = np.mean(psd, axis=1)  # Average across channels
    
    alpha_psd = psd[alpha_idx]
    peak_idx = np.argmax(alpha_psd)
    
    return freqs[alpha_idx[peak_idx]], alpha_psd[peak_idx]


def compute_alpha_asymmetry(
    left_data: np.ndarray,
    right_data: np.ndarray,
    sample_rate: float = 256.0
) -> float:
    """
    Compute frontal alpha asymmetry (FAA).
    
    FAA is associated with emotional processing:
    - Positive = greater left alpha (right hemisphere more active)
    - Often associated with approach motivation
    
    Args:
        left_data: Data from left frontal electrode (e.g., F3).
        right_data: Data from right frontal electrode (e.g., F4).
        sample_rate: Sampling rate.
        
    Returns:
        Asymmetry score (log(right) - log(left)).
    """
    analyzer = SpectralAnalyzer(sample_rate=sample_rate)
    
    left_power = analyzer.compute_band_power(left_data).alpha
    right_power = analyzer.compute_band_power(right_data).alpha
    
    # Avoid log(0)
    left_power = max(left_power, 1e-10)
    right_power = max(right_power, 1e-10)
    
    # Standard FAA formula
    return np.log(right_power) - np.log(left_power)


def compute_engagement_index(band_power: BandPower) -> float:
    """
    Compute cognitive engagement index.
    
    Higher values indicate more engagement/alertness.
    Formula: beta / (alpha + theta)
    
    Args:
        band_power: Band power values.
        
    Returns:
        Engagement index (typically 0.5-2.0).
    """
    denominator = band_power.alpha + band_power.theta
    if denominator == 0:
        return 0.0
    
    return band_power.beta / denominator


def compute_relaxation_index(band_power: BandPower) -> float:
    """
    Compute relaxation index.
    
    Higher values indicate more relaxation.
    Formula: alpha / beta
    
    Args:
        band_power: Band power values.
        
    Returns:
        Relaxation index.
    """
    if band_power.beta == 0:
        return 0.0
    
    return band_power.alpha / band_power.beta

