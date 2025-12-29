"""
EEG Channel Mapping - 10-20 International System.

Maps the 25-pin connector pins to standard electrode positions
and provides brain region information for each electrode.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Electrode:
    """Represents a single EEG electrode."""
    name: str                    # Standard name (e.g., "Fp1", "C3")
    pin_number: int              # Pin on 25-pin connector (1-25)
    x: float                     # 2D position for topographic map (0-1)
    y: float                     # 2D position for topographic map (0-1)
    theta: float                 # Spherical coordinate (degrees from top)
    phi: float                   # Spherical coordinate (degrees from front)
    brain_region: str            # General brain area
    lobe: str                    # Brain lobe
    description: str             # What this area typically relates to


# Standard 10-20 System electrode definitions
# Positions are normalized for 2D head plotting (0,0 = top-left, 1,1 = bottom-right)
# The head is viewed from above, nose pointing up

STANDARD_10_20: list[Electrode] = [
    # Frontal Polar (Prefrontal Cortex) - Executive function, decision making
    Electrode("Fp1", 1, 0.35, 0.12, -90, -18, "Prefrontal", "Frontal", 
              "Left prefrontal - executive function, planning, impulse control"),
    Electrode("Fp2", 2, 0.65, 0.12, -90, 18, "Prefrontal", "Frontal",
              "Right prefrontal - executive function, emotional regulation"),
    
    # Frontal - Motor planning, speech (Broca's area on left)
    Electrode("F7", 3, 0.15, 0.28, -60, -54, "Lateral Frontal", "Frontal",
              "Left lateral frontal - language production (Broca's area nearby)"),
    Electrode("F3", 4, 0.32, 0.25, -60, -39, "Dorsolateral Frontal", "Frontal",
              "Left frontal - motor planning, working memory"),
    Electrode("Fz", 5, 0.50, 0.22, -60, 0, "Medial Frontal", "Frontal",
              "Midline frontal - attention, error monitoring"),
    Electrode("F4", 6, 0.68, 0.25, -60, 39, "Dorsolateral Frontal", "Frontal",
              "Right frontal - motor planning, spatial working memory"),
    Electrode("F8", 7, 0.85, 0.28, -60, 54, "Lateral Frontal", "Frontal",
              "Right lateral frontal - emotional processing"),
    
    # Temporal - Auditory processing, memory, language comprehension
    Electrode("T3", 8, 0.07, 0.50, 0, -90, "Anterior Temporal", "Temporal",
              "Left temporal - language comprehension (Wernicke's nearby), auditory"),
    Electrode("T4", 9, 0.93, 0.50, 0, 90, "Anterior Temporal", "Temporal",
              "Right temporal - music perception, voice recognition"),
    Electrode("T5", 10, 0.15, 0.72, 60, -54, "Posterior Temporal", "Temporal",
              "Left posterior temporal - visual word recognition"),
    Electrode("T6", 11, 0.85, 0.72, 60, 54, "Posterior Temporal", "Temporal",
              "Right posterior temporal - face recognition"),
    
    # Central - Motor and sensory cortex (motor strip)
    Electrode("C3", 12, 0.30, 0.50, 0, -45, "Left Motor/Sensory", "Central",
              "Left central - RIGHT hand/arm motor control and sensation"),
    Electrode("Cz", 13, 0.50, 0.50, 0, 0, "Vertex", "Central",
              "Vertex - leg motor/sensory, midline attention"),
    Electrode("C4", 14, 0.70, 0.50, 0, 45, "Right Motor/Sensory", "Central",
              "Right central - LEFT hand/arm motor control and sensation"),
    
    # Parietal - Sensory integration, spatial awareness
    Electrode("P3", 15, 0.32, 0.72, 60, -39, "Left Parietal", "Parietal",
              "Left parietal - math, logic, right visual field attention"),
    Electrode("Pz", 16, 0.50, 0.72, 60, 0, "Midline Parietal", "Parietal",
              "Midline parietal - sensory integration, body awareness"),
    Electrode("P4", 17, 0.68, 0.72, 60, 39, "Right Parietal", "Parietal",
              "Right parietal - spatial awareness, left visual field attention"),
    
    # Occipital - Visual processing
    Electrode("O1", 18, 0.35, 0.90, 90, -18, "Left Occipital", "Occipital",
              "Left occipital - right visual field processing"),
    Electrode("Oz", 19, 0.50, 0.90, 90, 0, "Midline Occipital", "Occipital",
              "Midline occipital - central vision, alpha rhythm source"),
    Electrode("O2", 20, 0.65, 0.90, 90, 18, "Right Occipital", "Occipital",
              "Right occipital - left visual field processing"),
    
    # Reference and Ground (typically earlobes or mastoids)
    Electrode("A1", 21, 0.02, 0.50, 0, -90, "Left Reference", "Reference",
              "Left earlobe/mastoid - reference electrode"),
    Electrode("A2", 22, 0.98, 0.50, 0, 90, "Right Reference", "Reference",
              "Right earlobe/mastoid - reference electrode"),
]

# Pins 23-25 often used for:
# - Ground
# - Auxiliary channels (EOG, EMG, ECG)
# - Status/trigger signals


class ChannelMap:
    """
    Manages mapping between hardware pins and electrode positions.
    
    Allows customization for different EEG cap configurations.
    """
    
    def __init__(self, electrodes: Optional[list[Electrode]] = None):
        """
        Initialize channel map.
        
        Args:
            electrodes: List of electrode definitions. Uses standard 10-20 if None.
        """
        self.electrodes = electrodes or STANDARD_10_20.copy()
        self._by_name = {e.name: e for e in self.electrodes}
        self._by_pin = {e.pin_number: e for e in self.electrodes}
    
    def get_by_name(self, name: str) -> Optional[Electrode]:
        """Get electrode by standard name (e.g., 'Fp1', 'C3')."""
        return self._by_name.get(name)
    
    def get_by_pin(self, pin: int) -> Optional[Electrode]:
        """Get electrode by pin number on connector."""
        return self._by_pin.get(pin)
    
    def get_channel_names(self) -> list[str]:
        """Get list of all channel names in order."""
        return [e.name for e in self.electrodes if not e.name.startswith('A')]
    
    def get_positions_2d(self) -> np.ndarray:
        """Get 2D positions for all channels (for topographic plotting)."""
        return np.array([[e.x, e.y] for e in self.electrodes if not e.name.startswith('A')])
    
    def get_regions(self) -> dict[str, list[str]]:
        """Get channels grouped by brain region."""
        regions = {}
        for e in self.electrodes:
            if e.lobe not in regions:
                regions[e.lobe] = []
            regions[e.lobe].append(e.name)
        return regions
    
    def get_channel_info(self, name: str) -> dict:
        """Get detailed info about a channel."""
        e = self._by_name.get(name)
        if not e:
            return {}
        return {
            "name": e.name,
            "pin": e.pin_number,
            "region": e.brain_region,
            "lobe": e.lobe,
            "description": e.description,
            "position_2d": (e.x, e.y),
            "position_spherical": (e.theta, e.phi),
        }
    
    def remap_pin(self, channel_name: str, new_pin: int):
        """
        Remap a channel to a different pin number.
        
        Use this if your hardware has a different pin assignment.
        """
        if channel_name in self._by_name:
            electrode = self._by_name[channel_name]
            old_pin = electrode.pin_number
            
            # Update the electrode
            new_electrode = Electrode(
                electrode.name, new_pin, electrode.x, electrode.y,
                electrode.theta, electrode.phi, electrode.brain_region,
                electrode.lobe, electrode.description
            )
            
            # Update mappings
            self._by_name[channel_name] = new_electrode
            if old_pin in self._by_pin:
                del self._by_pin[old_pin]
            self._by_pin[new_pin] = new_electrode
            
            # Update list
            for i, e in enumerate(self.electrodes):
                if e.name == channel_name:
                    self.electrodes[i] = new_electrode
                    break


# Frequency band definitions (in Hz)
FREQUENCY_BANDS = {
    "delta": (0.5, 4),    # Deep sleep, unconscious
    "theta": (4, 8),      # Drowsiness, light sleep, meditation
    "alpha": (8, 13),     # Relaxed, eyes closed, calm alertness  
    "beta": (13, 30),     # Active thinking, focus, anxiety
    "gamma": (30, 100),   # High-level cognition, perception binding
}

# Sub-bands often used in research
FREQUENCY_BANDS_DETAILED = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "low_alpha": (8, 10),
    "high_alpha": (10, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
    "low_gamma": (30, 50),
    "high_gamma": (50, 100),
}
