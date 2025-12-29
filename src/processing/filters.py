"""
EEG Signal Filtering Module.

Implements common filters for EEG preprocessing:
- Bandpass filtering (isolate frequency range of interest)
- Notch filtering (remove 50/60 Hz line noise)
- Highpass/Lowpass filtering
- Artifact detection and removal
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, Literal
from dataclasses import dataclass


@dataclass
class FilterSettings:
    """Configuration for EEG filtering."""
    sample_rate: float = 256.0
    
    # Bandpass filter settings
    bandpass_low: float = 0.5      # Hz - remove slow drift
    bandpass_high: float = 50.0    # Hz - remove high freq noise
    bandpass_order: int = 4
    
    # Notch filter for line noise
    notch_freq: float = 60.0       # 60 Hz for US, 50 Hz for EU
    notch_q: float = 30.0          # Quality factor (higher = narrower notch)
    
    # Artifact thresholds
    amplitude_threshold: float = 150.0  # µV - reject if exceeded
    gradient_threshold: float = 50.0    # µV/sample - reject sudden jumps


class EEGFilter:
    """
    Real-time EEG signal filtering with support for streaming data.
    
    Implements IIR filters that can process data sample-by-sample
    while maintaining filter state between calls.
    """
    
    def __init__(self, settings: Optional[FilterSettings] = None):
        """
        Initialize the EEG filter.
        
        Args:
            settings: Filter configuration. Uses defaults if None.
        """
        self.settings = settings or FilterSettings()
        self._init_filters()
        
    def _init_filters(self):
        """Initialize all filter coefficients."""
        fs = self.settings.sample_rate
        
        # Bandpass filter (Butterworth)
        # Use second-order sections (sos) for numerical stability
        self._bandpass_sos = signal.butter(
            self.settings.bandpass_order,
            [self.settings.bandpass_low, self.settings.bandpass_high],
            btype='bandpass',
            fs=fs,
            output='sos'
        )
        
        # Notch filter for line noise (IIR notch)
        self._notch_b, self._notch_a = signal.iirnotch(
            self.settings.notch_freq,
            self.settings.notch_q,
            fs=fs
        )
        
        # Highpass filter (for DC removal / drift)
        self._highpass_sos = signal.butter(
            2,
            0.1,  # Very low cutoff just for DC
            btype='highpass',
            fs=fs,
            output='sos'
        )
        
        # Lowpass filter (anti-aliasing)
        self._lowpass_sos = signal.butter(
            4,
            self.settings.bandpass_high,
            btype='lowpass',
            fs=fs,
            output='sos'
        )
        
        # Filter states for real-time processing (per channel)
        self._bandpass_zi: Optional[np.ndarray] = None
        self._notch_zi: Optional[np.ndarray] = None
        self._n_channels: Optional[int] = None
        
    def reset_state(self):
        """Reset filter states (call when starting new recording)."""
        self._bandpass_zi = None
        self._notch_zi = None
        self._n_channels = None
        
    def _init_state(self, n_channels: int):
        """Initialize filter states for given number of channels."""
        if self._n_channels != n_channels:
            self._n_channels = n_channels
            
            # Initialize bandpass state for each channel
            zi = signal.sosfilt_zi(self._bandpass_sos)
            self._bandpass_zi = np.zeros((n_channels, zi.shape[0], zi.shape[1]))
            for i in range(n_channels):
                self._bandpass_zi[i] = zi
            
            # Initialize notch state for each channel
            zi_notch = signal.lfilter_zi(self._notch_b, self._notch_a)
            self._notch_zi = np.zeros((n_channels, len(zi_notch)))
            for i in range(n_channels):
                self._notch_zi[i] = zi_notch
    
    def filter_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Filter a single sample (all channels).
        
        This is the main method for real-time filtering.
        
        Args:
            sample: Array of shape (n_channels,) with one sample per channel.
            
        Returns:
            Filtered sample array of same shape.
        """
        n_channels = len(sample)
        self._init_state(n_channels)
        
        filtered = np.zeros(n_channels)
        
        for ch in range(n_channels):
            # Apply bandpass filter
            val, self._bandpass_zi[ch] = signal.sosfilt(
                self._bandpass_sos,
                [sample[ch]],
                zi=self._bandpass_zi[ch]
            )
            
            # Apply notch filter
            val, self._notch_zi[ch] = signal.lfilter(
                self._notch_b,
                self._notch_a,
                val,
                zi=self._notch_zi[ch]
            )
            
            filtered[ch] = val[0]
        
        return filtered
    
    def filter_chunk(self, data: np.ndarray) -> np.ndarray:
        """
        Filter a chunk of data (maintains state between calls).
        
        Args:
            data: Array of shape (n_samples, n_channels).
            
        Returns:
            Filtered data of same shape.
        """
        n_samples, n_channels = data.shape
        self._init_state(n_channels)
        
        filtered = np.zeros_like(data)
        
        for ch in range(n_channels):
            # Apply bandpass filter
            filtered[:, ch], self._bandpass_zi[ch] = signal.sosfilt(
                self._bandpass_sos,
                data[:, ch],
                zi=self._bandpass_zi[ch]
            )
            
            # Apply notch filter
            filtered[:, ch], self._notch_zi[ch] = signal.lfilter(
                self._notch_b,
                self._notch_a,
                filtered[:, ch],
                zi=self._notch_zi[ch]
            )
        
        return filtered
    
    def filter_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Filter a complete batch of data (no state, for offline processing).
        
        Args:
            data: Array of shape (n_samples, n_channels).
            
        Returns:
            Filtered data of same shape.
        """
        n_samples, n_channels = data.shape
        filtered = np.zeros_like(data)
        
        for ch in range(n_channels):
            # Apply bandpass (forward-backward for zero phase)
            filtered[:, ch] = signal.sosfiltfilt(
                self._bandpass_sos,
                data[:, ch]
            )
            
            # Apply notch (forward-backward for zero phase)
            filtered[:, ch] = signal.filtfilt(
                self._notch_b,
                self._notch_a,
                filtered[:, ch]
            )
        
        return filtered


class ArtifactDetector:
    """
    Detects and marks EEG artifacts for rejection or correction.
    
    Common artifacts:
    - Eye blinks (large frontal deflections)
    - Muscle artifacts (high-frequency bursts)
    - Movement artifacts (sudden baseline shifts)
    - Electrode pops (sharp spikes)
    """
    
    def __init__(self, settings: Optional[FilterSettings] = None):
        """
        Initialize artifact detector.
        
        Args:
            settings: Detection thresholds and parameters.
        """
        self.settings = settings or FilterSettings()
        
        # Buffer for gradient calculation
        self._last_sample: Optional[np.ndarray] = None
        
    def reset(self):
        """Reset detector state."""
        self._last_sample = None
    
    def detect_sample(self, sample: np.ndarray) -> Tuple[bool, str]:
        """
        Check a single sample for artifacts.
        
        Args:
            sample: Array of shape (n_channels,).
            
        Returns:
            Tuple of (is_artifact, artifact_type).
        """
        # Check amplitude threshold
        if np.any(np.abs(sample) > self.settings.amplitude_threshold):
            return True, "amplitude"
        
        # Check gradient (sudden changes)
        if self._last_sample is not None:
            gradient = np.abs(sample - self._last_sample)
            if np.any(gradient > self.settings.gradient_threshold):
                self._last_sample = sample
                return True, "gradient"
        
        self._last_sample = sample.copy()
        return False, ""
    
    def detect_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Detect artifacts in a batch of data.
        
        Args:
            data: Array of shape (n_samples, n_channels).
            
        Returns:
            Boolean array of shape (n_samples,) marking artifact samples.
        """
        n_samples = data.shape[0]
        artifacts = np.zeros(n_samples, dtype=bool)
        
        # Amplitude check
        max_amp = np.max(np.abs(data), axis=1)
        artifacts |= max_amp > self.settings.amplitude_threshold
        
        # Gradient check
        gradient = np.abs(np.diff(data, axis=0))
        max_gradient = np.max(gradient, axis=1)
        artifacts[1:] |= max_gradient > self.settings.gradient_threshold
        
        return artifacts
    
    def interpolate_artifacts(
        self,
        data: np.ndarray,
        artifact_mask: np.ndarray
    ) -> np.ndarray:
        """
        Replace artifact samples with interpolated values.
        
        Args:
            data: Array of shape (n_samples, n_channels).
            artifact_mask: Boolean array marking artifacts.
            
        Returns:
            Data with artifacts replaced by interpolated values.
        """
        cleaned = data.copy()
        n_samples, n_channels = data.shape
        
        for ch in range(n_channels):
            # Find artifact segments
            artifact_indices = np.where(artifact_mask)[0]
            
            if len(artifact_indices) == 0:
                continue
            
            # Simple linear interpolation
            good_indices = np.where(~artifact_mask)[0]
            if len(good_indices) < 2:
                continue
            
            cleaned[artifact_indices, ch] = np.interp(
                artifact_indices,
                good_indices,
                data[good_indices, ch]
            )
        
        return cleaned


class ReReference:
    """
    Re-reference EEG data to different reference schemes.
    
    Common reference schemes:
    - Common Average Reference (CAR): subtract mean of all channels
    - Linked Ears: average of A1 and A2
    - Bipolar: difference between adjacent electrodes
    - Laplacian: local spatial filtering
    """
    
    @staticmethod
    def common_average(data: np.ndarray, exclude_channels: Optional[list] = None) -> np.ndarray:
        """
        Apply Common Average Reference (CAR).
        
        Subtracts the mean of all channels from each channel.
        This removes common noise sources.
        
        Args:
            data: Array of shape (n_samples, n_channels).
            exclude_channels: Channel indices to exclude from average (e.g., reference channels).
            
        Returns:
            Re-referenced data.
        """
        if exclude_channels:
            mask = np.ones(data.shape[1], dtype=bool)
            mask[exclude_channels] = False
            avg = np.mean(data[:, mask], axis=1, keepdims=True)
        else:
            avg = np.mean(data, axis=1, keepdims=True)
        
        return data - avg
    
    @staticmethod
    def linked_ears(
        data: np.ndarray,
        a1_idx: int = 20,
        a2_idx: int = 21
    ) -> np.ndarray:
        """
        Apply linked ears reference.
        
        Uses average of left (A1) and right (A2) ear electrodes.
        
        Args:
            data: Array of shape (n_samples, n_channels).
            a1_idx: Index of A1 (left ear) channel.
            a2_idx: Index of A2 (right ear) channel.
            
        Returns:
            Re-referenced data.
        """
        ear_avg = (data[:, a1_idx] + data[:, a2_idx]) / 2
        return data - ear_avg[:, np.newaxis]
    
    @staticmethod
    def bipolar(data: np.ndarray, pairs: list[Tuple[int, int]]) -> np.ndarray:
        """
        Create bipolar montage (difference between electrode pairs).
        
        Args:
            data: Array of shape (n_samples, n_channels).
            pairs: List of (channel1, channel2) index tuples.
            
        Returns:
            Bipolar data of shape (n_samples, n_pairs).
        """
        result = np.zeros((data.shape[0], len(pairs)))
        
        for i, (ch1, ch2) in enumerate(pairs):
            result[:, i] = data[:, ch1] - data[:, ch2]
        
        return result


# Convenience functions for quick filtering

def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    sample_rate: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to data.
    
    Args:
        data: Array of shape (n_samples,) or (n_samples, n_channels).
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        sample_rate: Sampling frequency in Hz.
        order: Filter order.
        
    Returns:
        Filtered data of same shape.
    """
    sos = signal.butter(order, [low_freq, high_freq], btype='bandpass', fs=sample_rate, output='sos')
    
    if data.ndim == 1:
        return signal.sosfiltfilt(sos, data)
    else:
        return np.apply_along_axis(lambda x: signal.sosfiltfilt(sos, x), 0, data)


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    sample_rate: float,
    q: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove line noise.
    
    Args:
        data: Array of shape (n_samples,) or (n_samples, n_channels).
        notch_freq: Frequency to notch out (50 or 60 Hz typically).
        sample_rate: Sampling frequency in Hz.
        q: Quality factor (higher = narrower notch).
        
    Returns:
        Filtered data of same shape.
    """
    b, a = signal.iirnotch(notch_freq, q, fs=sample_rate)
    
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.apply_along_axis(lambda x: signal.filtfilt(b, a, x), 0, data)


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """
    Remove DC offset (mean) from data.
    
    Args:
        data: Array of any shape.
        
    Returns:
        Data with mean subtracted along first axis.
    """
    return data - np.mean(data, axis=0, keepdims=True)
