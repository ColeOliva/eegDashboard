"""
Real-time EEG Waveform Display.

Displays scrolling EEG traces for all channels using pyqtgraph
for high-performance real-time plotting.
"""

import numpy as np
from typing import Optional, List
from collections import deque

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

# Channel colors - distinct colors for each brain region
REGION_COLORS = {
    "Frontal": "#FF6B6B",      # Red
    "Central": "#4ECDC4",      # Teal
    "Temporal": "#45B7D1",     # Blue
    "Parietal": "#96CEB4",     # Green
    "Occipital": "#FFEAA7",    # Yellow
    "Reference": "#A0A0A0",    # Gray
}

# Default channel colors (rainbow-ish for distinction)
DEFAULT_COLORS = [
    "#FF6B6B", "#FF8E53", "#FFC93C", "#95E1D3", "#4ECDC4",
    "#45B7D1", "#6C5CE7", "#A29BFE", "#FD79A8", "#FDCB6E",
    "#00B894", "#0984E3", "#6C5B7B", "#C44569", "#F8B500",
    "#7158E2", "#3AE374", "#FF3838", "#17C0EB", "#3D3D3D",
    "#A0A0A0",  # 21 colors for 21 channels
]


class WaveformPlot(QWidget):
    """
    Real-time scrolling EEG waveform display.
    
    Features:
    - Stacked traces for all channels
    - Auto-scrolling time window
    - Adjustable amplitude scale
    - Channel labels with brain region info
    - Color coding by brain region
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        sample_rate: float = 256.0,
        time_window: float = 10.0,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the waveform plot.
        
        Args:
            channel_names: List of channel names (e.g., ["Fp1", "Fp2", ...]).
            sample_rate: Sampling rate in Hz.
            time_window: Visible time window in seconds.
            parent: Parent widget.
        """
        super().__init__(parent)
        
        # Default channel names (10-20 system)
        self.channel_names = channel_names or [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6",
            "O1", "Oz", "O2"
        ]
        self.n_channels = len(self.channel_names)
        self.sample_rate = sample_rate
        self.time_window = time_window
        
        # Data buffer (circular buffer for each channel)
        self.buffer_size = int(time_window * sample_rate)
        self.data_buffers = [deque(maxlen=self.buffer_size) for _ in range(self.n_channels)]
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        # Display settings
        self.amplitude_scale = 100.0  # µV per division
        self.channel_spacing = 1.0     # Vertical spacing multiplier
        self.paused = False
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1E1E1E')  # Dark background
        self.plot_widget.showGrid(x=True, y=False, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Channels')
        
        # Disable mouse interaction for cleaner real-time display
        self.plot_widget.setMouseEnabled(x=False, y=False)
        
        # Create plot curves for each channel
        self.curves = []
        self.channel_labels = []
        
        for i, name in enumerate(self.channel_names):
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            pen = pg.mkPen(color=color, width=1)
            curve = self.plot_widget.plot([], [], pen=pen, name=name)
            self.curves.append(curve)
        
        # Set up Y-axis with channel names
        self._setup_y_axis()
        
        layout.addWidget(self.plot_widget)
        
        # Controls bar
        controls = QHBoxLayout()
        
        # Amplitude scale slider
        controls.addWidget(QLabel("Scale:"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 500)
        self.scale_slider.setValue(int(self.amplitude_scale))
        self.scale_slider.setFixedWidth(150)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        controls.addWidget(self.scale_slider)
        
        self.scale_label = QLabel(f"{self.amplitude_scale:.0f} µV")
        self.scale_label.setFixedWidth(60)
        controls.addWidget(self.scale_label)
        
        controls.addStretch()
        
        # Time window selector
        controls.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["5 sec", "10 sec", "20 sec", "30 sec", "60 sec"])
        self.window_combo.setCurrentIndex(1)  # 10 sec default
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        controls.addWidget(self.window_combo)
        
        layout.addLayout(controls)
        
    def _setup_y_axis(self):
        """Configure Y-axis to show channel names."""
        # Create tick positions and labels
        ticks = []
        for i, name in enumerate(self.channel_names):
            y_pos = -i * self.channel_spacing
            ticks.append((y_pos, name))
        
        # Set custom Y-axis ticks
        y_axis = self.plot_widget.getAxis('left')
        y_axis.setTicks([ticks])
        
        # Set Y range to show all channels
        y_min = -(self.n_channels - 1) * self.channel_spacing - 0.5
        y_max = 0.5
        self.plot_widget.setYRange(y_min, y_max)
        
    def _on_scale_changed(self, value: int):
        """Handle amplitude scale change."""
        self.amplitude_scale = value
        self.scale_label.setText(f"{value} µV")
        self._update_plot()
        
    def _on_window_changed(self, text: str):
        """Handle time window change."""
        seconds = int(text.split()[0])
        self.time_window = seconds
        self.buffer_size = int(self.time_window * self.sample_rate)
        
        # Resize buffers
        for i in range(self.n_channels):
            old_data = list(self.data_buffers[i])
            self.data_buffers[i] = deque(old_data, maxlen=self.buffer_size)
        
        old_time = list(self.time_buffer)
        self.time_buffer = deque(old_time, maxlen=self.buffer_size)
        
        self._update_plot()
        
    def add_sample(self, sample: np.ndarray, timestamp: Optional[float] = None):
        """
        Add a single sample to the display.
        
        Args:
            sample: Array of shape (n_channels,) with values in µV.
            timestamp: Optional timestamp (auto-generated if None).
        """
        if self.paused:
            return
            
        if timestamp is None:
            if len(self.time_buffer) > 0:
                timestamp = self.time_buffer[-1] + 1.0 / self.sample_rate
            else:
                timestamp = 0.0
        
        self.time_buffer.append(timestamp)
        
        for i in range(min(len(sample), self.n_channels)):
            self.data_buffers[i].append(sample[i])
            
    def add_chunk(self, data: np.ndarray, start_time: Optional[float] = None):
        """
        Add a chunk of data to the display.
        
        Args:
            data: Array of shape (n_samples, n_channels) with values in µV.
            start_time: Timestamp of first sample.
        """
        if self.paused:
            return
            
        n_samples = data.shape[0]
        
        if start_time is None:
            if len(self.time_buffer) > 0:
                start_time = self.time_buffer[-1] + 1.0 / self.sample_rate
            else:
                start_time = 0.0
        
        # Generate timestamps
        times = start_time + np.arange(n_samples) / self.sample_rate
        
        for t in times:
            self.time_buffer.append(t)
        
        for i in range(min(data.shape[1], self.n_channels)):
            for val in data[:, i]:
                self.data_buffers[i].append(val)
                
    def _update_plot(self):
        """Update all plot curves with current buffer data."""
        if len(self.time_buffer) < 2:
            return
            
        times = np.array(self.time_buffer)
        
        for i, curve in enumerate(self.curves):
            if len(self.data_buffers[i]) > 0:
                data = np.array(self.data_buffers[i])
                
                # Normalize and offset for stacked display
                # Scale to ±0.5 range based on amplitude scale
                normalized = data / self.amplitude_scale * 0.4
                
                # Offset by channel index
                offset = -i * self.channel_spacing
                y_data = normalized + offset
                
                curve.setData(times, y_data)
        
        # Update X range to show latest data
        if len(times) > 0:
            x_max = times[-1]
            x_min = max(0, x_max - self.time_window)
            self.plot_widget.setXRange(x_min, x_max, padding=0)
            
    def update(self):
        """Call this periodically to refresh the display."""
        self._update_plot()
        
    def clear(self):
        """Clear all data from the display."""
        for i in range(self.n_channels):
            self.data_buffers[i].clear()
        self.time_buffer.clear()
        
        for curve in self.curves:
            curve.setData([], [])
            
    def pause(self):
        """Pause the display (stop accepting new data)."""
        self.paused = True
        
    def resume(self):
        """Resume the display."""
        self.paused = False
        
    def set_channel_colors(self, colors: List[str]):
        """Set custom colors for each channel."""
        for i, (curve, color) in enumerate(zip(self.curves, colors)):
            pen = pg.mkPen(color=color, width=1)
            curve.setPen(pen)


class MultiChannelPlot(QWidget):
    """
    Alternative display with separate subplot for each channel.
    
    Better for detailed analysis of individual channels,
    but uses more screen space.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        sample_rate: float = 256.0,
        time_window: float = 10.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        
        self.channel_names = channel_names or [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4"
        ]
        self.n_channels = len(self.channel_names)
        self.sample_rate = sample_rate
        self.time_window = time_window
        
        # Data buffers
        self.buffer_size = int(time_window * sample_rate)
        self.data_buffers = [deque(maxlen=self.buffer_size) for _ in range(self.n_channels)]
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        self._init_ui()
        
    def _init_ui(self):
        """Set up multi-plot layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create graphics layout for subplots
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground('#1E1E1E')
        
        self.plots = []
        self.curves = []
        
        for i, name in enumerate(self.channel_names):
            # Create subplot
            plot = self.graphics_layout.addPlot(row=i, col=0)
            plot.setLabel('left', name)
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideAxis('bottom')  # Hide x-axis except for last plot
            
            # Link X axes
            if i > 0:
                plot.setXLink(self.plots[0])
            
            # Create curve
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            pen = pg.mkPen(color=color, width=1)
            curve = plot.plot([], [], pen=pen)
            
            self.plots.append(plot)
            self.curves.append(curve)
        
        # Show X axis on last plot
        if self.plots:
            self.plots[-1].showAxis('bottom')
            self.plots[-1].setLabel('bottom', 'Time', units='s')
        
        layout.addWidget(self.graphics_layout)
        
    def add_sample(self, sample: np.ndarray, timestamp: Optional[float] = None):
        """Add a single sample."""
        if timestamp is None:
            if len(self.time_buffer) > 0:
                timestamp = self.time_buffer[-1] + 1.0 / self.sample_rate
            else:
                timestamp = 0.0
        
        self.time_buffer.append(timestamp)
        
        for i in range(min(len(sample), self.n_channels)):
            self.data_buffers[i].append(sample[i])
            
    def add_chunk(self, data: np.ndarray, start_time: Optional[float] = None):
        """Add a chunk of data."""
        n_samples = data.shape[0]
        
        if start_time is None:
            if len(self.time_buffer) > 0:
                start_time = self.time_buffer[-1] + 1.0 / self.sample_rate
            else:
                start_time = 0.0
        
        times = start_time + np.arange(n_samples) / self.sample_rate
        
        for t in times:
            self.time_buffer.append(t)
        
        for i in range(min(data.shape[1], self.n_channels)):
            for val in data[:, i]:
                self.data_buffers[i].append(val)
                
    def update(self):
        """Refresh the display."""
        if len(self.time_buffer) < 2:
            return
            
        times = np.array(self.time_buffer)
        
        for i, curve in enumerate(self.curves):
            if len(self.data_buffers[i]) > 0:
                data = np.array(self.data_buffers[i])
                curve.setData(times, data)
        
        # Update X range
        if len(times) > 0 and self.plots:
            x_max = times[-1]
            x_min = max(0, x_max - self.time_window)
            self.plots[0].setXRange(x_min, x_max, padding=0)
            
    def clear(self):
        """Clear all data."""
        for i in range(self.n_channels):
            self.data_buffers[i].clear()
        self.time_buffer.clear()
        
        for curve in self.curves:
            curve.setData([], [])
