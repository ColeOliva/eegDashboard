#!/usr/bin/env python3
"""
Test script for real-time EEG waveform display.

This demonstrates the waveform visualization using simulated EEG data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import QTimer
import numpy as np

from visualization.waveform_plot import WaveformPlot
from hardware.simulator import EEGSimulator
from processing.filters import EEGFilter


class TestWindow(QMainWindow):
    """Test window for waveform display."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Waveform Display Test")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create waveform plot
        self.waveform = WaveformPlot(
            sample_rate=256,
            time_window=10.0
        )
        layout.addWidget(self.waveform)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_stream)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_stream)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_data)
        btn_layout.addWidget(self.clear_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Set up simulator and filter
        self.simulator = EEGSimulator(n_channels=20, sample_rate=256)
        self.filter = EEGFilter()
        
        # Timer for updating display
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        
        # Data callback
        self.pending_data = []
        self.simulator.register_callback(self.on_data)
        
    def on_data(self, packet):
        """Callback for incoming EEG data."""
        # Filter the data
        filtered = self.filter.filter_sample(packet.channels)
        self.pending_data.append((filtered, packet.timestamp))
        
    def start_stream(self):
        """Start the EEG simulation."""
        self.filter.reset_state()
        self.simulator.start_acquisition()
        self.update_timer.start(33)  # ~30 FPS
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
    def stop_stream(self):
        """Stop the EEG simulation."""
        self.simulator.stop_acquisition()
        self.update_timer.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def clear_data(self):
        """Clear the display."""
        self.waveform.clear()
        self.pending_data.clear()
        
    def update_display(self):
        """Update the waveform display with pending data."""
        # Process pending data
        for data, timestamp in self.pending_data:
            self.waveform.add_sample(data, timestamp)
        self.pending_data.clear()
        
        # Refresh display
        self.waveform.update()
        
    def closeEvent(self, event):
        """Clean up on close."""
        self.stop_stream()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    
    window = TestWindow()
    window.show()
    
    print("=" * 50)
    print("EEG Waveform Display Test")
    print("=" * 50)
    print()
    print("Click 'Start' to begin streaming simulated EEG data.")
    print("Use the Scale slider to adjust amplitude.")
    print("Use the Window dropdown to change time scale.")
    print()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
