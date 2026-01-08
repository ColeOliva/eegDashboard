"""
Optimized Spectral Analysis Module.

High-performance frequency analysis with:
- Pre-computed FFT windows
- Cached frequency grids
- Vectorized band power computation
- Memory-efficient streaming
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.fft import next_fast_len, rfft, rfftfreq

# Standard EEG frequency bands (Hz)
FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
}


@dataclass(slots=True)
class BandPower:
    """Optimized band power container using slots for memory efficiency."""
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
        return self.delta + self.theta + self.alpha + self.beta + self.gamma
    
    def relative(self) -> Dict[str, float]:
        total = self.total()
        if total == 0:
            return {k: 0.0 for k in self.to_dict()}
        return {k: v / total * 100 for k, v in self.to_dict().items()}
    
    def as_array(self) -> np.ndarray:
        """Return as numpy array for vectorized operations."""
        return np.array([self.delta, self.theta, self.alpha, self.beta, self.gamma])


class OptimizedSpectralAnalyzer:
    """
    High-performance spectral analyzer with pre-computed lookup tables.
    
    Key optimizations:
    1. Pre-computed FFT-optimized window sizes
    2. Cached frequency bin indices per band
    3. Vectorized multi-channel processing
    4. Ring buffer for streaming (avoids allocations)
    """
    
    __slots__ = ['sample_rate', 'window_size', 'n_samples', 'hop_size',
                 '_window', '_freqs', '_band_indices', '_buffer', '_buffer_idx',
                 '_n_channels', '_fft_size', '_freq_resolution']
    
    def __init__(
        self,
        sample_rate: float = 256.0,
        window_size: float = 2.0,
        overlap: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Use FFT-optimal size for speed
        raw_n_samples = int(window_size * sample_rate)
        self._fft_size = next_fast_len(raw_n_samples)
        self.n_samples = raw_n_samples
        self.hop_size = int(self.n_samples * (1 - overlap))
        
        # Pre-compute window
        self._window = signal.windows.hann(self.n_samples, sym=False).astype(np.float64)
        
        # Pre-compute frequency grid
        self._freqs = rfftfreq(self._fft_size, 1 / sample_rate)
        self._freq_resolution = self._freqs[1] - self._freqs[0]
        
        # Pre-compute band indices (critical optimization)
        self._band_indices = {}
        for band_name, (low, high) in FREQUENCY_BANDS.items():
            indices = np.where((self._freqs >= low) & (self._freqs <= high))[0]
            self._band_indices[band_name] = indices
        
        # Ring buffer for streaming
        self._buffer: Optional[np.ndarray] = None
        self._buffer_idx = 0
        self._n_channels: Optional[int] = None
        
    def _init_buffer(self, n_channels: int):
        """Initialize ring buffer."""
        if self._n_channels != n_channels:
            self._n_channels = n_channels
            # Pre-allocate ring buffer
            self._buffer = np.zeros((self.n_samples, n_channels), dtype=np.float64)
            self._buffer_idx = 0
            
    def add_sample(self, sample: np.ndarray):
        """Add sample to ring buffer (O(1) operation)."""
        self._init_buffer(len(sample))
        self._buffer[self._buffer_idx] = sample
        self._buffer_idx = (self._buffer_idx + 1) % self.n_samples
        
    def add_chunk(self, data: np.ndarray):
        """Add chunk to ring buffer efficiently."""
        self._init_buffer(data.shape[1])
        n_samples = len(data)
        
        if n_samples >= self.n_samples:
            # Data larger than buffer - just take the last window
            self._buffer[:] = data[-self.n_samples:]
            self._buffer_idx = 0
        else:
            # Wrap around ring buffer
            end_idx = self._buffer_idx + n_samples
            if end_idx <= self.n_samples:
                self._buffer[self._buffer_idx:end_idx] = data
            else:
                first_part = self.n_samples - self._buffer_idx
                self._buffer[self._buffer_idx:] = data[:first_part]
                self._buffer[:end_idx - self.n_samples] = data[first_part:]
            self._buffer_idx = end_idx % self.n_samples
            
    def clear_buffer(self):
        """Clear the streaming buffer."""
        if self._buffer is not None:
            self._buffer.fill(0)
            self._buffer_idx = 0
    
    def _get_ordered_buffer(self) -> np.ndarray:
        """Get buffer data in time order."""
        if self._buffer is None:
            return np.array([])
        return np.roll(self._buffer, -self._buffer_idx, axis=0)
    
    def compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT magnitude spectrum - optimized.
        
        Args:
            data: 1D array of samples.
            
        Returns:
            (frequencies, magnitudes) tuple.
        """
        # Apply window and pad to FFT-optimal size
        n = len(data)
        if n < self.n_samples:
            padded = np.zeros(self._fft_size)
            padded[:n] = data * self._window[:n]
        else:
            padded = np.zeros(self._fft_size)
            padded[:self.n_samples] = data[-self.n_samples:] * self._window
        
        # Compute FFT
        fft_result = rfft(padded)
        magnitudes = np.abs(fft_result) * (2.0 / self.n_samples)
        
        return self._freqs, magnitudes
    
    def compute_psd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.
        
        Args:
            data: 1D or 2D array (n_samples,) or (n_samples, n_channels).
            
        Returns:
            (frequencies, psd) tuple.
        """
        nperseg = min(self.n_samples, len(data))
        noverlap = nperseg // 2
        
        if data.ndim == 1:
            freqs, psd = signal.welch(
                data, fs=self.sample_rate, window='hann',
                nperseg=nperseg, noverlap=noverlap
            )
        else:
            # Multi-channel - vectorized
            n_channels = data.shape[1]
            freqs, psd0 = signal.welch(
                data[:, 0], fs=self.sample_rate, window='hann',
                nperseg=nperseg, noverlap=noverlap
            )
            psd = np.zeros((len(freqs), n_channels))
            psd[:, 0] = psd0
            for ch in range(1, n_channels):
                _, psd[:, ch] = signal.welch(
                    data[:, ch], fs=self.sample_rate, window='hann',
                    nperseg=nperseg, noverlap=noverlap
                )
        
        return freqs, psd
    
    def compute_band_power(self, data: np.ndarray, method: str = "welch") -> BandPower:
        """
        Compute power in each frequency band - optimized.
        
        Uses pre-computed band indices for O(1) band lookup.
        """
        if method == "welch":
            freqs, psd = self.compute_psd(data)
        else:
            freqs, mags = self.compute_fft(data)
            psd = mags ** 2
            
        # Fast band power computation using pre-computed indices
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        
        powers = {}
        for band_name, indices in self._band_indices.items():
            if len(indices) > 0:
                # Find overlapping indices (freqs might differ from _freqs)
                band_low, band_high = FREQUENCY_BANDS[band_name]
                mask = (freqs >= band_low) & (freqs <= band_high)
                if np.any(mask):
                    powers[band_name] = np.sum(psd[mask]) * freq_res
                else:
                    powers[band_name] = 0.0
            else:
                powers[band_name] = 0.0
        
        return BandPower(**powers)
    
    def compute_band_power_multichannel(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute band power for all channels - fully vectorized.
        
        Args:
            data: Array of shape (n_samples, n_channels).
            
        Returns:
            Dict mapping band name to array of powers per channel.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        freqs, psd = self.compute_psd(data)
        n_channels = psd.shape[1]
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        
        result = {}
        for band_name, (low, high) in FREQUENCY_BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                # Vectorized sum across frequency dimension
                result[band_name] = np.sum(psd[mask, :], axis=0) * freq_res
            else:
                result[band_name] = np.zeros(n_channels)
        
        return result
    
    def get_current_band_power(self, channel: int = 0) -> BandPower:
        """Get band power from current buffer state."""
        if self._buffer is None or self._buffer_idx == 0:
            return BandPower(0, 0, 0, 0, 0)
        
        data = self._get_ordered_buffer()[:, channel]
        return self.compute_band_power(data)
    
    def get_current_band_power_all(self) -> Dict[str, np.ndarray]:
        """Get band power for all channels from buffer."""
        if self._buffer is None:
            n_ch = self._n_channels or 1
            return {band: np.zeros(n_ch) for band in FREQUENCY_BANDS}
        
        data = self._get_ordered_buffer()
        return self.compute_band_power_multichannel(data)


# Cognitive indices - vectorized versions
def compute_engagement_index(band_power: BandPower) -> float:
    """Compute cognitive engagement: beta / (alpha + theta)."""
    denom = band_power.alpha + band_power.theta
    return band_power.beta / denom if denom > 0 else 0.0


def compute_relaxation_index(band_power: BandPower) -> float:
    """Compute relaxation index: alpha / beta."""
    return band_power.alpha / band_power.beta if band_power.beta > 0 else 0.0


def compute_focus_index(band_power: BandPower) -> float:
    """Compute focus index: (beta + gamma) / (delta + theta)."""
    denom = band_power.delta + band_power.theta
    return (band_power.beta + band_power.gamma) / denom if denom > 0 else 0.0


def compute_drowsiness_index(band_power: BandPower) -> float:
    """Compute drowsiness index: (theta + delta) / (alpha + beta)."""
    denom = band_power.alpha + band_power.beta
    return (band_power.theta + band_power.delta) / denom if denom > 0 else 0.0


@lru_cache(maxsize=16)
def _get_welch_params(n_samples: int, fs: float) -> Tuple[int, int]:
    """Cached Welch parameters for given data length."""
    nperseg = min(256, n_samples)
    noverlap = nperseg // 2
    return nperseg, noverlap


def fast_band_power(data: np.ndarray, fs: float = 256.0) -> BandPower:
    """
    Fast single-call band power computation.
    
    Optimized for one-off calculations.
    """
    n = len(data)
    nperseg, noverlap = _get_welch_params(n, fs)
    
    freqs, psd = signal.welch(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    freq_res = freqs[1] - freqs[0]
    
    powers = {}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = np.sum(psd[mask]) * freq_res if np.any(mask) else 0.0
    
    return BandPower(**powers)
