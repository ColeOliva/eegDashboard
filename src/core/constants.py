"""
Shared constants and utilities for EEG Dashboard.

This module consolidates common constants to avoid duplication.
"""

from typing import Dict, Tuple

import numpy as np

# Standard 10-20 electrode positions (normalized to unit circle)
# Positions are in (x, y) where nose is at top (y=1), left ear at x=-1
ELECTRODE_POSITIONS_2D: Dict[str, Tuple[float, float]] = {
    # Frontal polar
    "Fp1": (-0.22, 0.83), "Fp2": (0.22, 0.83),
    # Frontal
    "F7": (-0.59, 0.59), "F3": (-0.32, 0.50), "Fz": (0.0, 0.50),
    "F4": (0.32, 0.50), "F8": (0.59, 0.59),
    # Temporal
    "T3": (-0.81, 0.0), "T4": (0.81, 0.0),
    "T5": (-0.59, -0.59), "T6": (0.59, -0.59),
    # Central
    "C3": (-0.38, 0.0), "Cz": (0.0, 0.0), "C4": (0.38, 0.0),
    # Parietal
    "P3": (-0.32, -0.50), "Pz": (0.0, -0.50), "P4": (0.32, -0.50),
    # Occipital
    "O1": (-0.22, -0.83), "Oz": (0.0, -0.83), "O2": (0.22, -0.83),
    # Reference (ear lobes)
    "A1": (-0.95, 0.0), "A2": (0.95, 0.0),
}


# 3D electrode positions on unit sphere (x, y, z)
# Nose points in +Y direction, top of head is +Z
ELECTRODE_POSITIONS_3D: Dict[str, Tuple[float, float, float]] = {
    "Fp1": (-0.22, 0.95, 0.22), "Fp2": (0.22, 0.95, 0.22),
    "F7": (-0.70, 0.70, 0.14), "F3": (-0.40, 0.75, 0.53),
    "Fz": (0.0, 0.72, 0.69), "F4": (0.40, 0.75, 0.53),
    "F8": (0.70, 0.70, 0.14),
    "T3": (-0.99, 0.0, 0.14), "T4": (0.99, 0.0, 0.14),
    "T5": (-0.70, -0.70, 0.14), "T6": (0.70, -0.70, 0.14),
    "C3": (-0.55, 0.0, 0.83), "Cz": (0.0, 0.0, 1.0),
    "C4": (0.55, 0.0, 0.83),
    "P3": (-0.40, -0.55, 0.73), "Pz": (0.0, -0.55, 0.83),
    "P4": (0.40, -0.55, 0.73),
    "O1": (-0.22, -0.95, 0.22), "Oz": (0.0, -0.92, 0.38),
    "O2": (0.22, -0.95, 0.22),
    "A1": (-1.0, 0.0, -0.15), "A2": (1.0, 0.0, -0.15),
}


# Standard channel order (10-20 system, excluding reference)
STANDARD_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "Oz", "O2"
]


# Standard EEG frequency bands (in Hz)
FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4),     # Deep sleep, unconscious processes
    "theta": (4, 8),       # Drowsiness, light sleep, meditation
    "alpha": (8, 13),      # Relaxed, calm, eyes closed
    "beta": (13, 30),      # Active thinking, focus, alertness
    "gamma": (30, 100),    # Higher cognitive functions, perception
}


# Sub-bands for detailed analysis
FREQUENCY_BANDS_DETAILED: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "low_alpha": (8, 10),
    "high_alpha": (10, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
    "low_gamma": (30, 50),
    "high_gamma": (50, 100),
}


# Band colors for visualization
BAND_COLORS: Dict[str, str] = {
    "delta": "#8B5CF6",    # Purple
    "theta": "#06B6D4",    # Cyan
    "alpha": "#10B981",    # Green
    "beta": "#F59E0B",     # Orange
    "gamma": "#EF4444",    # Red
}


# Brain regions
BRAIN_REGIONS: Dict[str, Dict] = {
    "Frontal": {
        "channels": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
        "color": "#FF6B6B",
        "description": "Executive function, planning, motor control"
    },
    "Central": {
        "channels": ["C3", "Cz", "C4"],
        "color": "#4ECDC4",
        "description": "Motor and sensory cortex"
    },
    "Temporal": {
        "channels": ["T3", "T4", "T5", "T6"],
        "color": "#45B7D1",
        "description": "Auditory processing, memory, language"
    },
    "Parietal": {
        "channels": ["P3", "Pz", "P4"],
        "color": "#96CEB4",
        "description": "Sensory integration, spatial awareness"
    },
    "Occipital": {
        "channels": ["O1", "Oz", "O2"],
        "color": "#FFEAA7",
        "description": "Visual processing"
    },
}


# Default sample rates (Hz)
COMMON_SAMPLE_RATES = [128, 256, 512, 1024]
DEFAULT_SAMPLE_RATE = 256


# Default filter settings
DEFAULT_FILTER_SETTINGS = {
    "bandpass_low": 0.5,
    "bandpass_high": 50.0,
    "notch_freq_us": 60.0,
    "notch_freq_eu": 50.0,
    "notch_q": 30.0,
}


# Artifact thresholds (µV)
ARTIFACT_THRESHOLDS = {
    "amplitude": 150.0,   # Reject if amplitude exceeds this
    "gradient": 50.0,     # Reject sudden jumps > this per sample
    "eye_blink": 100.0,   # Frontal channels threshold for blinks
}


def get_channel_positions_2d(channel_names: list = None) -> np.ndarray:
    """Get 2D positions for given channels as numpy array."""
    channels = channel_names or STANDARD_CHANNEL_ORDER
    positions = []
    for name in channels:
        if name in ELECTRODE_POSITIONS_2D:
            positions.append(ELECTRODE_POSITIONS_2D[name])
    return np.array(positions)


def get_channel_positions_3d(channel_names: list = None) -> np.ndarray:
    """Get 3D positions for given channels as numpy array."""
    channels = channel_names or STANDARD_CHANNEL_ORDER
    positions = []
    for name in channels:
        if name in ELECTRODE_POSITIONS_3D:
            positions.append(ELECTRODE_POSITIONS_3D[name])
    return np.array(positions)


def get_region_for_channel(channel_name: str) -> str:
    """Get brain region for a channel."""
    for region, info in BRAIN_REGIONS.items():
        if channel_name in info["channels"]:
            return region
    return "Unknown"
