"""Visualization modules for EEG data display."""

# Lazy imports to avoid loading Qt when not needed
def __getattr__(name):
    if name in ("WaveformPlot", "MultiChannelPlot"):
        from .waveform_plot import WaveformPlot, MultiChannelPlot
        return locals()[name]
    elif name in ("TopoMap", "BandTopoMaps", "ELECTRODE_POSITIONS"):
        from .topomap import TopoMap, BandTopoMaps, ELECTRODE_POSITIONS
        return locals()[name]
    elif name in ("EnhancedTopoMap", "RealtimeTopoWidget"):
        from .topomap_enhanced import EnhancedTopoMap, RealtimeTopoWidget
        return locals()[name]
    elif name in ("Brain3D", "AnimatedBrain3D", "ELECTRODE_POSITIONS_3D"):
        from .brain_3d import Brain3D, AnimatedBrain3D, ELECTRODE_POSITIONS_3D
        return locals()[name]
    elif name in ("AnatomicalBrainMap", "Anatomical3DBrain", "FastBrainRenderer"):
        from .brain_anatomical import AnatomicalBrainMap, Anatomical3DBrain, FastBrainRenderer
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "WaveformPlot", "MultiChannelPlot",
    "TopoMap", "BandTopoMaps", "ELECTRODE_POSITIONS",
    "EnhancedTopoMap", "RealtimeTopoWidget",
    "Brain3D", "AnimatedBrain3D", "ELECTRODE_POSITIONS_3D",
    "AnatomicalBrainMap", "Anatomical3DBrain", "FastBrainRenderer",
]
