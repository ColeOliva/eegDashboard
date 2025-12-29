"""Signal processing modules for EEG data."""

from .filters import EEGFilter, ArtifactDetector, FilterSettings, ReReference
from .spectral import SpectralAnalyzer, BandPower, BandPowerTracker, FREQUENCY_BANDS

__all__ = [
    "EEGFilter", "ArtifactDetector", "FilterSettings", "ReReference",
    "SpectralAnalyzer", "BandPower", "BandPowerTracker", "FREQUENCY_BANDS",
]
