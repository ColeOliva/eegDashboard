"""
EEG Dashboard - Main Application Window.

Integrates all visualization components into a cohesive real-time dashboard:
- Waveform display (scrolling EEG traces)
- Spectral analysis (frequency band power)
- Brain topographic map
- Recording controls
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
from collections import deque
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QPushButton, QLabel, QComboBox,
    QSlider, QGroupBox, QStatusBar, QSplitter, QFrame,
    QFileDialog, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.waveform_plot import WaveformPlot
from visualization.topomap import TopoMap, BandTopoMaps
from processing.filters import EEGFilter, FilterSettings
from processing.spectral import SpectralAnalyzer, BandPower, compute_engagement_index
from hardware.simulator import EEGSimulator
from hardware.serial_interface import SerialEEGReader
from core.channel_map import ChannelMap, STANDARD_10_20


class BandPowerWidget(QWidget):
    """Widget displaying current band power as bars."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        title = QLabel("Band Power")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        self.bars = {}
        self.labels = {}
        
        bands = [
            ("delta", "Delta (0.5-4 Hz)", "#8B5CF6"),
            ("theta", "Theta (4-8 Hz)", "#06B6D4"),
            ("alpha", "Alpha (8-13 Hz)", "#10B981"),
            ("beta", "Beta (13-30 Hz)", "#F59E0B"),
            ("gamma", "Gamma (30-100 Hz)", "#EF4444"),
        ]
        
        for band_id, band_name, color in bands:
            row = QHBoxLayout()
            
            label = QLabel(band_name.split()[0])
            label.setFixedWidth(50)
            row.addWidget(label)
            
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #444;
                    border-radius: 3px;
                    background-color: #2a2a2a;
                    height: 16px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 2px;
                }}
            """)
            row.addWidget(bar)
            
            value_label = QLabel("0%")
            value_label.setFixedWidth(40)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(value_label)
            
            self.bars[band_id] = bar
            self.labels[band_id] = value_label
            
            layout.addLayout(row)
            
        layout.addStretch()
        
    def update_values(self, band_power: BandPower):
        """Update bar values from BandPower object."""
        relative = band_power.relative()
        
        for band_id, pct in relative.items():
            if band_id in self.bars:
                self.bars[band_id].setValue(int(pct))
                self.labels[band_id].setText(f"{pct:.0f}%")


class TopoMapWidget(QWidget):
    """Widget displaying brain topographic map."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.topomap = TopoMap()
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title and band selector
        header = QHBoxLayout()
        title = QLabel("Brain Map")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(title)
        
        self.band_combo = QComboBox()
        self.band_combo.addItems(["Alpha", "Beta", "Theta", "Delta", "Gamma"])
        self.band_combo.currentTextChanged.connect(self._on_band_changed)
        header.addWidget(self.band_combo)
        
        layout.addLayout(header)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border-radius: 5px;")
        layout.addWidget(self.image_label)
        
        self.current_band = "alpha"
        self.current_data = None
        
    def _on_band_changed(self, text: str):
        self.current_band = text.lower()
        if self.current_data is not None:
            self._render(self.current_data)
            
    def update_data(self, band_powers: dict):
        """Update topomap with new band power data."""
        self.current_data = band_powers
        self._render(band_powers)
        
    def _render(self, band_powers: dict):
        """Render the topomap image."""
        if self.current_band not in band_powers:
            return
            
        values = band_powers[self.current_band]
        
        # Get image bytes
        img_bytes = self.topomap.to_image_bytes(
            values,
            title=f"{self.current_band.capitalize()} Power",
            show_electrodes=True,
            show_colorbar=False,
            figsize=(4, 4)
        )
        
        # Convert to QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(img_bytes)
        
        # Scale to fit
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled)


class StatusWidget(QWidget):
    """Widget showing connection status and metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        title = QLabel("Status")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Status indicators
        self.connection_label = QLabel("⚫ Disconnected")
        layout.addWidget(self.connection_label)
        
        self.sample_rate_label = QLabel("Sample Rate: -- Hz")
        layout.addWidget(self.sample_rate_label)
        
        self.samples_label = QLabel("Samples: 0")
        layout.addWidget(self.samples_label)
        
        # Cognitive indices
        layout.addWidget(QLabel(""))
        indices_title = QLabel("Cognitive Indices")
        indices_title.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(indices_title)
        
        self.engagement_label = QLabel("Engagement: --")
        layout.addWidget(self.engagement_label)
        
        self.relaxation_label = QLabel("Relaxation: --")
        layout.addWidget(self.relaxation_label)
        
        layout.addStretch()
        
    def set_connected(self, connected: bool, mode: str = ""):
        if connected:
            self.connection_label.setText(f"🟢 Connected ({mode})")
            self.connection_label.setStyleSheet("color: #10B981;")
        else:
            self.connection_label.setText("⚫ Disconnected")
            self.connection_label.setStyleSheet("color: #888;")
            
    def set_sample_rate(self, rate: float):
        self.sample_rate_label.setText(f"Sample Rate: {rate:.0f} Hz")
        
    def set_sample_count(self, count: int):
        self.samples_label.setText(f"Samples: {count:,}")
        
    def set_engagement(self, value: float):
        self.engagement_label.setText(f"Engagement: {value:.2f}")
        
    def set_relaxation(self, value: float):
        self.relaxation_label.setText(f"Relaxation: {value:.2f}")


class EEGDashboard(QMainWindow):
    """Main EEG Dashboard application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("EEG Dashboard")
        self.setMinimumSize(1400, 800)
        
        # Data source
        self.data_source = None
        self.using_simulator = True
        
        # Processing
        self.filter = EEGFilter()
        self.spectral = SpectralAnalyzer(sample_rate=256, window_size=2.0)
        self.channel_map = ChannelMap()
        
        # Data buffers
        self.sample_count = 0
        self.data_buffer = deque(maxlen=256 * 10)  # 10 seconds
        self.pending_samples = []
        
        # Recording
        self.recording = False
        self.recorded_data = []
        
        # Initialize UI
        self._init_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        
    def _init_ui(self):
        """Set up the main user interface."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top control bar
        controls = self._create_controls()
        main_layout.addLayout(controls)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Waveform
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        waveform_group = QGroupBox("EEG Waveforms")
        waveform_layout = QVBoxLayout(waveform_group)
        self.waveform = WaveformPlot(sample_rate=256, time_window=10.0)
        waveform_layout.addWidget(self.waveform)
        left_layout.addWidget(waveform_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel: Analysis
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Status
        self.status_widget = StatusWidget()
        right_layout.addWidget(self.status_widget)
        
        # Band power
        self.band_power_widget = BandPowerWidget()
        right_layout.addWidget(self.band_power_widget)
        
        # Topomap
        self.topomap_widget = TopoMapWidget()
        right_layout.addWidget(self.topomap_widget)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([1000, 300])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready - Click 'Connect' to start")
        
    def _create_controls(self) -> QHBoxLayout:
        """Create the top control bar."""
        layout = QHBoxLayout()
        
        # Source selection
        layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Simulator", "Serial Port"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_combo)
        
        # Port selection (for serial)
        self.port_combo = QComboBox()
        self.port_combo.addItem("Auto-detect")
        self._refresh_ports()
        self.port_combo.setEnabled(False)
        layout.addWidget(self.port_combo)
        
        layout.addWidget(QLabel(" "))
        
        # Connect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        layout.addWidget(self.connect_btn)
        
        # Disconnect button
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        self.disconnect_btn.setEnabled(False)
        layout.addWidget(self.disconnect_btn)
        
        layout.addStretch()
        
        # Recording controls
        self.record_btn = QPushButton("⏺ Record")
        self.record_btn.clicked.connect(self._on_record)
        self.record_btn.setEnabled(False)
        layout.addWidget(self.record_btn)
        
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)
        
        return layout
        
    def _refresh_ports(self):
        """Refresh available serial ports."""
        self.port_combo.clear()
        self.port_combo.addItem("Auto-detect")
        
        ports = SerialEEGReader.list_available_ports()
        for port in ports:
            self.port_combo.addItem(f"{port['device']} - {port['description']}")
            
    def _on_source_changed(self, text: str):
        """Handle data source selection change."""
        self.port_combo.setEnabled(text == "Serial Port")
        if text == "Serial Port":
            self._refresh_ports()
            
    def _on_connect(self):
        """Connect to data source."""
        source = self.source_combo.currentText()
        
        if source == "Simulator":
            self.data_source = EEGSimulator(n_channels=20, sample_rate=256)
            self.using_simulator = True
            mode = "Simulator"
        else:
            # Get selected port
            port_text = self.port_combo.currentText()
            port = None if port_text == "Auto-detect" else port_text.split(" - ")[0]
            
            self.data_source = SerialEEGReader(
                port=port,
                baud_rate=115200,
                n_channels=20,
                sample_rate=256
            )
            self.using_simulator = False
            mode = "Serial"
            
        # Connect and start
        if self.data_source.connect():
            self.data_source.register_callback(self._on_data)
            self.data_source.start_acquisition()
            
            self.filter.reset_state()
            self.spectral.clear_buffer()
            self.sample_count = 0
            self.data_buffer.clear()
            self.waveform.clear()
            
            self.update_timer.start(33)  # ~30 FPS
            
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
            self.source_combo.setEnabled(False)
            self.port_combo.setEnabled(False)
            
            self.status_widget.set_connected(True, mode)
            self.status_widget.set_sample_rate(256)
            self.statusBar().showMessage(f"Connected to {mode}")
        else:
            QMessageBox.warning(self, "Connection Error", "Failed to connect to data source")
            
    def _on_disconnect(self):
        """Disconnect from data source."""
        if self.data_source:
            self.data_source.stop_acquisition()
            self.data_source.disconnect()
            self.data_source = None
            
        self.update_timer.stop()
        
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.source_combo.setEnabled(True)
        
        self.status_widget.set_connected(False)
        self.statusBar().showMessage("Disconnected")
        
    def _on_data(self, packet):
        """Callback for incoming EEG data."""
        # Filter the data
        filtered = self.filter.filter_sample(packet.channels)
        
        # Add to buffers
        self.pending_samples.append((filtered, packet.timestamp))
        self.data_buffer.append(filtered)
        self.spectral.add_sample(filtered)
        
        self.sample_count += 1
        
        # Recording
        if self.recording:
            self.recorded_data.append({
                'timestamp': packet.timestamp,
                'channels': filtered.copy()
            })
            
    def _update_display(self):
        """Update all visualizations."""
        # Process pending waveform samples
        for data, timestamp in self.pending_samples:
            self.waveform.add_sample(data, timestamp)
        self.pending_samples.clear()
        self.waveform.update()
        
        # Update sample count
        self.status_widget.set_sample_count(self.sample_count)
        
        # Update spectral analysis (less frequently)
        if len(self.data_buffer) >= 256:
            data = np.array(self.data_buffer)
            
            # Band power for first channel (or average)
            bp = self.spectral.compute_band_power(data[:, 0])
            self.band_power_widget.update_values(bp)
            
            # Cognitive indices
            from processing.spectral import compute_engagement_index, compute_relaxation_index
            engagement = compute_engagement_index(bp)
            relaxation = compute_relaxation_index(bp)
            self.status_widget.set_engagement(engagement)
            self.status_widget.set_relaxation(relaxation)
            
            # Topomap (update less frequently - every ~0.5 seconds)
            if self.sample_count % 128 == 0:
                band_powers = self.spectral.compute_band_power_multichannel(data)
                self.topomap_widget.update_data(band_powers)
                
    def _on_record(self):
        """Toggle recording."""
        if self.recording:
            self.recording = False
            self.record_btn.setText("⏺ Record")
            self.record_btn.setStyleSheet("")
            self.save_btn.setEnabled(True)
            self.statusBar().showMessage(f"Recording stopped - {len(self.recorded_data)} samples")
        else:
            self.recording = True
            self.recorded_data = []
            self.record_btn.setText("⏹ Stop")
            self.record_btn.setStyleSheet("background-color: #EF4444; color: white;")
            self.save_btn.setEnabled(False)
            self.statusBar().showMessage("Recording...")
            
    def _on_save(self):
        """Save recorded data."""
        if not self.recorded_data:
            QMessageBox.information(self, "No Data", "No recorded data to save")
            return
            
        # Ask for file location
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save EEG Recording",
            "eeg_recording.csv",
            "CSV Files (*.csv);;EDF Files (*.edf)"
        )
        
        if filename:
            if filename.endswith('.csv'):
                self._save_csv(filename)
            elif filename.endswith('.edf'):
                self._save_edf(filename)
            else:
                self._save_csv(filename + '.csv')
                
    def _save_csv(self, filename: str):
        """Save data as CSV."""
        import csv
        
        channel_names = self.channel_map.get_channel_names()[:20]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['timestamp'] + channel_names
            writer.writerow(header)
            
            # Data
            for sample in self.recorded_data:
                row = [sample['timestamp']] + list(sample['channels'])
                writer.writerow(row)
                
        self.statusBar().showMessage(f"Saved {len(self.recorded_data)} samples to {filename}")
        
    def _save_edf(self, filename: str):
        """Save data as EDF (placeholder)."""
        # EDF saving requires pyedflib - full implementation in data_export.py
        QMessageBox.information(
            self, "EDF Export",
            "EDF export will be implemented in the next update.\nSaving as CSV instead."
        )
        self._save_csv(filename.replace('.edf', '.csv'))
        
    def closeEvent(self, event):
        """Clean up on close."""
        self._on_disconnect()
        event.accept()


def main():
    """Launch the EEG Dashboard application."""
    app = QApplication(sys.argv)
    
    # Set dark fusion style
    app.setStyle("Fusion")
    
    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # Create and show window
    window = EEGDashboard()
    window.show()
    
    print("=" * 50)
    print("EEG Dashboard v0.1.0")
    print("=" * 50)
    print()
    print("Dashboard is running!")
    print("- Click 'Connect' to start streaming EEG data")
    print("- Use 'Record' to capture data for saving")
    print()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
