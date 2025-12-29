"""Hardware interface modules for EEG data acquisition."""

from .serial_interface import SerialEEGReader
from .simulator import EEGSimulator

__all__ = ["SerialEEGReader", "EEGSimulator"]
