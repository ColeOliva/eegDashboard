"""
Optimized EEG Signal Filtering Module.

High-performance filtering with:
- Vectorized NumPy operations
- Pre-allocated buffers
- Cascaded biquad filters for numerical stability
- SIMD-friendly memory layouts
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class FilterSettings:
    """Configuration for EEG filtering."""
    sample_rate: float = 256.0
    bandpass_low: float = 0.5
    bandpass_high: float = 50.0
    bandpass_order: int = 4
    notch_freq: float = 60.0
    notch_q: float = 30.0
    amplitude_threshold: float = 150.0
    gradient_threshold: float = 50.0


@dataclass
class FilterState:
    """Pre-allocated filter state for real-time processing."""
    bandpass_zi: np.ndarray = field(default_factory=lambda: np.array([]))
    notch_zi: np.ndarray = field(default_factory=lambda: np.array([]))
    n_channels: int = 0
    initialized: bool = False


class OptimizedEEGFilter:
    """
    High-performance EEG filter with vectorized operations.
    
    Key optimizations:
    1. Pre-allocated output buffers
    2. Vectorized multi-channel filtering
    3. Cascaded second-order sections for stability
    4. Contiguous memory access patterns
    """
    
    __slots__ = ['settings', '_bandpass_sos', '_notch_sos', '_state', 
                 '_output_buffer', '_temp_buffer']
    
    def __init__(self, settings: Optional[FilterSettings] = None):
        self.settings = settings or FilterSettings()
        self._state = FilterState()
        self._output_buffer: Optional[np.ndarray] = None
        self._temp_buffer: Optional[np.ndarray] = None
        self._init_filters()
        
    def _init_filters(self):
        """Initialize filter coefficients using second-order sections."""
        fs = self.settings.sample_rate
        
        # Bandpass using cascaded biquads (SOS) for numerical stability
        self._bandpass_sos = signal.butter(
            self.settings.bandpass_order,
            [self.settings.bandpass_low, self.settings.bandpass_high],
            btype='bandpass',
            fs=fs,
            output='sos'
        )
        
        # Notch filter as SOS for consistency
        b, a = signal.iirnotch(
            self.settings.notch_freq,
            self.settings.notch_q,
            fs=fs
        )
        # Convert to SOS format
        self._notch_sos = signal.tf2sos(b, a)
        
    def _init_state(self, n_channels: int):
        """Initialize filter state with pre-allocated buffers."""
        if self._state.initialized and self._state.n_channels == n_channels:
            return
            
        self._state.n_channels = n_channels
        
        # Initialize bandpass state for all channels at once
        # Shape: (n_sections, 2) per channel -> (n_channels, n_sections, 2)
        zi_bp = signal.sosfilt_zi(self._bandpass_sos)
        self._state.bandpass_zi = np.zeros((n_channels, *zi_bp.shape), dtype=np.float64)
        for i in range(n_channels):
            self._state.bandpass_zi[i] = zi_bp
        
        # Initialize notch state
        zi_notch = signal.sosfilt_zi(self._notch_sos)
        self._state.notch_zi = np.zeros((n_channels, *zi_notch.shape), dtype=np.float64)
        for i in range(n_channels):
            self._state.notch_zi[i] = zi_notch
            
        # Pre-allocate output buffer
        self._output_buffer = np.zeros(n_channels, dtype=np.float64)
        
        self._state.initialized = True
        
    def reset_state(self):
        """Reset filter states."""
        self._state = FilterState()
        self._output_buffer = None
        
    def filter_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Filter a single sample (all channels) - optimized version.
        
        Args:
            sample: Array of shape (n_channels,).
            
        Returns:
            Filtered sample array.
        """
        n_channels = len(sample)
        self._init_state(n_channels)
        
        # Process all channels (vectorized where possible)
        for ch in range(n_channels):
            # Bandpass
            val, self._state.bandpass_zi[ch] = signal.sosfilt(
                self._bandpass_sos,
                [sample[ch]],
                zi=self._state.bandpass_zi[ch]
            )
            
            # Notch
            val, self._state.notch_zi[ch] = signal.sosfilt(
                self._notch_sos,
                val,
                zi=self._state.notch_zi[ch]
            )
            
            self._output_buffer[ch] = val[0]
        
        return self._output_buffer.copy()
    
    def filter_chunk(self, data: np.ndarray) -> np.ndarray:
        """
        Filter a chunk of data - optimized for batch processing.
        
        Uses contiguous memory access and minimizes allocations.
        
        Args:
            data: Array of shape (n_samples, n_channels), C-contiguous.
            
        Returns:
            Filtered data of same shape.
        """
        # Ensure contiguous memory layout
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
            
        n_samples, n_channels = data.shape
        self._init_state(n_channels)
        
        # Pre-allocate output
        filtered = np.empty_like(data)
        
        # Process each channel (column-major iteration for cache efficiency)
        for ch in range(n_channels):
            # Bandpass
            filtered[:, ch], self._state.bandpass_zi[ch] = signal.sosfilt(
                self._bandpass_sos,
                data[:, ch],
                zi=self._state.bandpass_zi[ch]
            )
            
            # Notch
            filtered[:, ch], self._state.notch_zi[ch] = signal.sosfilt(
                self._notch_sos,
                filtered[:, ch],
                zi=self._state.notch_zi[ch]
            )
        
        return filtered
    
    def filter_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Filter complete batch (zero-phase, offline processing).
        
        Uses forward-backward filtering for zero phase distortion.
        
        Args:
            data: Array of shape (n_samples, n_channels).
            
        Returns:
            Filtered data.
        """
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
            
        n_samples, n_channels = data.shape
        filtered = np.empty_like(data)
        
        for ch in range(n_channels):
            # Zero-phase bandpass
            filtered[:, ch] = signal.sosfiltfilt(self._bandpass_sos, data[:, ch])
            # Zero-phase notch
            filtered[:, ch] = signal.sosfiltfilt(self._notch_sos, filtered[:, ch])
        
        return filtered


class AdaptiveArtifactDetector:
    """
    Adaptive artifact detection with automatic threshold adjustment.
    
    Uses running statistics to adapt thresholds to signal characteristics.
    """
    
    __slots__ = ['_amplitude_threshold', '_gradient_threshold', '_running_mean',
                 '_running_var', '_alpha', '_samples_seen', '_last_sample',
                 '_baseline_established']
    
    def __init__(
        self,
        initial_amplitude_threshold: float = 150.0,
        initial_gradient_threshold: float = 50.0,
        adaptation_rate: float = 0.001,
    ):
        self._amplitude_threshold = initial_amplitude_threshold
        self._gradient_threshold = initial_gradient_threshold
        self._alpha = adaptation_rate
        self._running_mean: Optional[np.ndarray] = None
        self._running_var: Optional[np.ndarray] = None
        self._samples_seen = 0
        self._last_sample: Optional[np.ndarray] = None
        self._baseline_established = False
        
    def reset(self):
        """Reset detector state."""
        self._running_mean = None
        self._running_var = None
        self._samples_seen = 0
        self._last_sample = None
        self._baseline_established = False
        
    def _update_statistics(self, sample: np.ndarray):
        """Update running mean and variance (Welford's algorithm)."""
        if self._running_mean is None:
            self._running_mean = sample.copy()
            self._running_var = np.zeros_like(sample)
        else:
            # Welford's online algorithm for numerical stability
            delta = sample - self._running_mean
            self._running_mean += self._alpha * delta
            delta2 = sample - self._running_mean
            self._running_var = (1 - self._alpha) * self._running_var + self._alpha * delta * delta2
            
        self._samples_seen += 1
        if self._samples_seen >= 256:  # ~1 second at 256 Hz
            self._baseline_established = True
            
    def detect_sample(self, sample: np.ndarray) -> Tuple[bool, str, np.ndarray]:
        """
        Check sample for artifacts with adaptive thresholds.
        
        Returns:
            Tuple of (is_artifact, artifact_type, channel_mask).
        """
        n_channels = len(sample)
        channel_mask = np.zeros(n_channels, dtype=bool)
        
        # Update running statistics
        self._update_statistics(sample)
        
        # Adaptive threshold based on running statistics
        if self._baseline_established and self._running_var is not None:
            std = np.sqrt(self._running_var)
            adaptive_amp_threshold = np.maximum(
                self._running_mean + 4 * std,
                self._amplitude_threshold
            )
        else:
            adaptive_amp_threshold = np.full(n_channels, self._amplitude_threshold)
        
        # Amplitude check
        amp_artifact = np.abs(sample) > adaptive_amp_threshold
        if np.any(amp_artifact):
            channel_mask |= amp_artifact
            if self._last_sample is not None:
                self._last_sample = sample.copy()
            return True, "amplitude", channel_mask
        
        # Gradient check
        if self._last_sample is not None:
            gradient = np.abs(sample - self._last_sample)
            grad_artifact = gradient > self._gradient_threshold
            if np.any(grad_artifact):
                channel_mask |= grad_artifact
                self._last_sample = sample.copy()
                return True, "gradient", channel_mask
        
        self._last_sample = sample.copy()
        return False, "", channel_mask
    
    def detect_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Detect artifacts in batch - vectorized implementation.
        
        Returns:
            Boolean array of shape (n_samples,) marking artifact samples.
        """
        n_samples, n_channels = data.shape
        
        # Vectorized amplitude detection
        max_amp = np.max(np.abs(data), axis=1)
        artifacts = max_amp > self._amplitude_threshold
        
        # Vectorized gradient detection
        gradients = np.abs(np.diff(data, axis=0))
        max_gradient = np.max(gradients, axis=1)
        artifacts[1:] |= max_gradient > self._gradient_threshold
        
        return artifacts


class RealTimeReReference:
    """
    Real-time re-referencing with optimized implementations.
    """
    
    @staticmethod
    def common_average_inplace(data: np.ndarray, exclude_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply CAR in-place for memory efficiency.
        
        Args:
            data: Array to modify in-place.
            exclude_indices: Channels to exclude from average.
        """
        if exclude_indices is not None and len(exclude_indices) > 0:
            mask = np.ones(data.shape[-1], dtype=bool)
            mask[exclude_indices] = False
            avg = np.mean(data[..., mask], axis=-1, keepdims=True)
        else:
            avg = np.mean(data, axis=-1, keepdims=True)
        
        data -= avg
        return data
    
    @staticmethod
    def linked_ears_inplace(data: np.ndarray, a1_idx: int = 20, a2_idx: int = 21) -> np.ndarray:
        """Apply linked ears reference in-place."""
        ear_avg = (data[..., a1_idx] + data[..., a2_idx]) / 2
        data -= ear_avg[..., np.newaxis]
        return data


# Fast utility functions
@lru_cache(maxsize=32)
def _get_bandpass_sos(low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Cached bandpass filter coefficients."""
    return signal.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')


@lru_cache(maxsize=32)
def _get_notch_sos(freq: float, q: float, fs: float) -> np.ndarray:
    """Cached notch filter coefficients."""
    b, a = signal.iirnotch(freq, q, fs=fs)
    return signal.tf2sos(b, a)


def fast_bandpass(data: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Fast bandpass with cached coefficients."""
    sos = _get_bandpass_sos(low, high, fs)
    if data.ndim == 1:
        return signal.sosfiltfilt(sos, data)
    return np.apply_along_axis(lambda x: signal.sosfiltfilt(sos, x), 0, data)


def fast_notch(data: np.ndarray, freq: float, fs: float, q: float = 30.0) -> np.ndarray:
    """Fast notch filter with cached coefficients."""
    sos = _get_notch_sos(freq, q, fs)
    if data.ndim == 1:
        return signal.sosfiltfilt(sos, data)
    return np.apply_along_axis(lambda x: signal.sosfiltfilt(sos, x), 0, data)
