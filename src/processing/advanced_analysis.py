"""
Advanced EEG Analysis Features.

Includes:
- Connectivity analysis (coherence, phase locking)
- Real-time spectrogram
- Basic ICA for artifact removal
- Statistical analysis utilities
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.stats import zscore


@dataclass
class ConnectivityResult:
    """Results from connectivity analysis."""
    coherence: np.ndarray      # Coherence values per frequency
    frequencies: np.ndarray    # Frequency axis
    phase_diff: np.ndarray     # Phase difference per frequency
    mean_coherence: float      # Average coherence
    peak_frequency: float      # Frequency of peak coherence


class ConnectivityAnalyzer:
    """
    Analyze functional connectivity between EEG channels.
    
    Methods:
    - Coherence (magnitude squared coherence)
    - Phase locking value (PLV)
    - Cross-correlation
    """
    
    def __init__(self, sample_rate: float = 256.0, window_size: float = 2.0):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.nperseg = int(window_size * sample_rate)
        
    def coherence(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
    ) -> ConnectivityResult:
        """
        Compute magnitude squared coherence between two signals.
        
        Coherence measures how correlated the frequency components are.
        Range: 0 (uncorrelated) to 1 (perfectly correlated)
        """
        freqs, coh = signal.coherence(
            signal1, signal2,
            fs=self.sample_rate,
            window='hann',
            nperseg=min(self.nperseg, len(signal1)),
            noverlap=self.nperseg // 2
        )
        
        # Compute cross-spectral density for phase
        _, csd = signal.csd(
            signal1, signal2,
            fs=self.sample_rate,
            window='hann',
            nperseg=min(self.nperseg, len(signal1))
        )
        phase_diff = np.angle(csd)
        
        # Find peak
        peak_idx = np.argmax(coh)
        
        return ConnectivityResult(
            coherence=coh,
            frequencies=freqs,
            phase_diff=phase_diff,
            mean_coherence=np.mean(coh),
            peak_frequency=freqs[peak_idx]
        )
    
    def coherence_matrix(
        self,
        data: np.ndarray,
        freq_band: Tuple[float, float] = (8, 13),  # Alpha by default
    ) -> np.ndarray:
        """
        Compute coherence matrix between all channel pairs.
        
        Args:
            data: (n_samples, n_channels) array
            freq_band: Frequency range to average coherence over
            
        Returns:
            (n_channels, n_channels) coherence matrix
        """
        n_channels = data.shape[1]
        coh_matrix = np.eye(n_channels)
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                result = self.coherence(data[:, i], data[:, j])
                
                # Average coherence in frequency band
                mask = (result.frequencies >= freq_band[0]) & (result.frequencies <= freq_band[1])
                if np.any(mask):
                    band_coh = np.mean(result.coherence[mask])
                else:
                    band_coh = 0.0
                    
                coh_matrix[i, j] = band_coh
                coh_matrix[j, i] = band_coh
                
        return coh_matrix
    
    def phase_locking_value(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        freq_band: Tuple[float, float] = (8, 13),
    ) -> float:
        """
        Compute Phase Locking Value (PLV) between two signals.
        
        PLV measures phase synchronization regardless of amplitude.
        Range: 0 (no synchronization) to 1 (perfect synchronization)
        """
        from scipy.signal import hilbert

        # Bandpass filter
        sos = signal.butter(4, freq_band, btype='bandpass', fs=self.sample_rate, output='sos')
        filt1 = signal.sosfiltfilt(sos, signal1)
        filt2 = signal.sosfiltfilt(sos, signal2)
        
        # Hilbert transform to get instantaneous phase
        analytic1 = hilbert(filt1)
        analytic2 = hilbert(filt2)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # PLV
        phase_diff = phase1 - phase2
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv


class RealTimeSpectrogram:
    """
    Compute spectrogram in real-time for visualization.
    """
    
    def __init__(
        self,
        sample_rate: float = 256.0,
        window_size: float = 0.5,
        overlap: float = 0.75,
        freq_limit: float = 50.0,
    ):
        self.sample_rate = sample_rate
        self.nperseg = int(window_size * sample_rate)
        self.noverlap = int(self.nperseg * overlap)
        self.freq_limit = freq_limit
        
        # Pre-compute window
        self.window = signal.windows.hann(self.nperseg)
        
        # History buffer for time axis
        self._history: List[np.ndarray] = []
        self._max_history = 100  # Number of time slices to keep
        
    def compute(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram.
        
        Args:
            data: 1D signal array
            
        Returns:
            (frequencies, times, spectrogram) tuple
        """
        freqs, times, Sxx = signal.spectrogram(
            data,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )
        
        # Limit frequency range
        freq_mask = freqs <= self.freq_limit
        freqs = freqs[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return freqs, times, Sxx_db
    
    def add_window(self, data: np.ndarray) -> np.ndarray:
        """
        Add a window of data and get current spectrogram slice.
        
        Args:
            data: Window of samples (at least nperseg samples)
            
        Returns:
            Power spectrum for this window
        """
        if len(data) < self.nperseg:
            data = np.pad(data, (self.nperseg - len(data), 0))
            
        windowed = data[-self.nperseg:] * self.window
        fft = np.fft.rfft(windowed)
        power = np.abs(fft) ** 2
        
        # Limit frequency range
        freqs = np.fft.rfftfreq(self.nperseg, 1/self.sample_rate)
        freq_mask = freqs <= self.freq_limit
        power = power[freq_mask]
        
        # Update history
        self._history.append(power)
        if len(self._history) > self._max_history:
            self._history.pop(0)
            
        return 10 * np.log10(power + 1e-10)
    
    def get_spectrogram_image(self) -> np.ndarray:
        """Get current spectrogram as 2D array (freq x time)."""
        if not self._history:
            return np.array([])
        return np.column_stack(self._history)


class SimpleICA:
    """
    Simple ICA implementation for artifact removal.
    
    Uses FastICA algorithm for blind source separation.
    Useful for removing eye blink and muscle artifacts.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self._mixing_matrix: Optional[np.ndarray] = None
        self._unmixing_matrix: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        
    def fit(self, data: np.ndarray, max_iter: int = 200, tol: float = 1e-4):
        """
        Fit ICA model to data.
        
        Args:
            data: (n_samples, n_channels) array
        """
        n_samples, n_channels = data.shape
        n_components = self.n_components or n_channels
        
        # Center data
        self._mean = np.mean(data, axis=0)
        X = data - self._mean
        
        # Whiten data
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Handle numerical issues
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        X_white = X @ whitening.T
        
        # FastICA
        W = np.random.randn(n_components, n_channels)
        W = self._orthogonalize(W)
        
        for _ in range(max_iter):
            W_new = self._ica_step(X_white, W)
            W_new = self._orthogonalize(W_new)
            
            # Check convergence
            if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1)) < tol:
                break
                
            W = W_new
            
        self._unmixing_matrix = W @ whitening
        self._mixing_matrix = np.linalg.pinv(self._unmixing_matrix)
        
    def _ica_step(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """One step of FastICA using tanh nonlinearity."""
        S = X @ W.T
        G = np.tanh(S)
        G_prime = 1 - G ** 2
        W_new = (G.T @ X) / len(X) - np.diag(np.mean(G_prime, axis=0)) @ W
        return W_new
        
    def _orthogonalize(self, W: np.ndarray) -> np.ndarray:
        """Symmetric orthogonalization."""
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        return U @ Vt
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to independent components."""
        if self._unmixing_matrix is None:
            raise ValueError("Must fit ICA first")
        return (data - self._mean) @ self._unmixing_matrix.T
        
    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """Transform sources back to channel space."""
        if self._mixing_matrix is None:
            raise ValueError("Must fit ICA first")
        return sources @ self._mixing_matrix.T + self._mean
        
    def remove_component(
        self,
        data: np.ndarray,
        component_indices: List[int],
    ) -> np.ndarray:
        """
        Remove specific components (artifacts) from data.
        
        Args:
            data: (n_samples, n_channels) array
            component_indices: Indices of components to remove
            
        Returns:
            Cleaned data
        """
        sources = self.transform(data)
        sources[:, component_indices] = 0
        return self.inverse_transform(sources)
    
    def identify_artifact_components(
        self,
        data: np.ndarray,
        frontal_indices: List[int] = [0, 1],  # Fp1, Fp2
        threshold: float = 2.0,
    ) -> List[int]:
        """
        Automatically identify artifact components.
        
        Eye blink artifacts typically have:
        - High amplitude in frontal channels
        - Characteristic temporal pattern
        
        Args:
            data: EEG data
            frontal_indices: Indices of frontal channels
            threshold: Z-score threshold for detection
            
        Returns:
            List of component indices likely to be artifacts
        """
        if self._mixing_matrix is None:
            self.fit(data)
            
        artifact_indices = []
        n_components = self._mixing_matrix.shape[1]
        
        for i in range(n_components):
            # Check if this component loads heavily on frontal channels
            frontal_loading = np.mean(np.abs(self._mixing_matrix[frontal_indices, i]))
            other_loading = np.mean(np.abs(np.delete(self._mixing_matrix[:, i], frontal_indices)))
            
            if frontal_loading > threshold * other_loading:
                artifact_indices.append(i)
                
        return artifact_indices


class StatisticalAnalysis:
    """Statistical analysis utilities for EEG data."""
    
    @staticmethod
    def compute_snr(signal_data: np.ndarray, noise_data: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.
        """
        signal_power = np.mean(signal_data ** 2)
        noise_power = np.mean(noise_data ** 2)
        
        if noise_power == 0:
            return float('inf')
            
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def detect_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outlier samples using z-score.
        
        Returns boolean mask of outliers.
        """
        z_scores = np.abs(zscore(data, axis=0))
        return np.any(z_scores > threshold, axis=1)
    
    @staticmethod
    def compute_variance_ratio(
        data: np.ndarray,
        window_size: int = 256,
    ) -> np.ndarray:
        """
        Compute variance ratio (signal quality metric).
        
        High variance ratio can indicate artifacts.
        """
        n_windows = len(data) // window_size
        variances = []
        
        for i in range(n_windows):
            window = data[i * window_size:(i + 1) * window_size]
            variances.append(np.var(window, axis=0))
            
        variances = np.array(variances)
        mean_var = np.mean(variances, axis=0)
        
        return variances / mean_var
    
    @staticmethod
    def hjorth_parameters(data: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute Hjorth parameters (time-domain features).
        
        Returns:
            (activity, mobility, complexity) tuple
        """
        # Activity = variance
        activity = np.var(data)
        
        # First derivative
        d1 = np.diff(data)
        var_d1 = np.var(d1)
        
        # Second derivative  
        d2 = np.diff(d1)
        var_d2 = np.var(d2)
        
        # Mobility = sqrt(var(d1) / var(data))
        mobility = np.sqrt(var_d1 / activity) if activity > 0 else 0
        
        # Complexity = mobility(d1) / mobility(data)
        mobility_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 0 else 0
        complexity = mobility_d1 / mobility if mobility > 0 else 0
        
        return activity, mobility, complexity
