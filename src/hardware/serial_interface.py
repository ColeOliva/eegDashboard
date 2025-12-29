"""
Serial interface for reading EEG data from USB adapter.

This module handles communication with EEG hardware via a USB-to-serial
adapter connected to the 25-pin EEG cap connector.
"""

import serial
import serial.tools.list_ports
from typing import Optional, Callable
import threading
import numpy as np
from dataclasses import dataclass


@dataclass
class EEGPacket:
    """Represents a single EEG data packet from hardware."""
    timestamp: float
    channels: np.ndarray  # Shape: (n_channels,)
    sequence_number: int


class SerialEEGReader:
    """
    Reads EEG data from USB serial adapter.
    
    The 25-pin connector typically carries:
    - 19-21 EEG channels (10-20 system)
    - Ground and reference signals
    - Possibly auxiliary channels (EOG, EMG)
    
    Note: The exact protocol depends on your specific EEG hardware.
    Common configurations:
    - Baud rate: 115200 or 230400
    - Data format: Binary packets or ASCII
    """
    
    # Common baud rates for EEG devices
    COMMON_BAUD_RATES = [115200, 230400, 57600, 9600, 256000, 500000]
    
    def __init__(
        self,
        port: Optional[str] = None,
        baud_rate: int = 115200,
        n_channels: int = 21,
        sample_rate: int = 256,
    ):
        """
        Initialize the serial EEG reader.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0' on Linux, 'COM3' on Windows).
                  If None, will attempt auto-detection.
            baud_rate: Communication speed in bits per second.
            n_channels: Number of EEG channels (typically 19-21 for 10-20 system).
            sample_rate: Expected samples per second (common: 256, 512, 1024 Hz).
        """
        self.port = port
        self.baud_rate = baud_rate
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[EEGPacket], None]] = []
        self._sequence = 0
        
    @staticmethod
    def list_available_ports() -> list[dict]:
        """
        List all available serial ports.
        
        Returns:
            List of dicts with port info (device, description, hwid).
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
                "manufacturer": port.manufacturer,
            })
        return ports
    
    def connect(self) -> bool:
        """
        Establish connection to the EEG device.
        
        Returns:
            True if connection successful, False otherwise.
        """
        if self._serial and self._serial.is_open:
            return True
            
        port = self.port or self._auto_detect_port()
        if not port:
            print("No serial port specified or detected.")
            return False
            
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,
            )
            print(f"Connected to {port} at {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False
    
    def _auto_detect_port(self) -> Optional[str]:
        """Attempt to auto-detect the EEG device port."""
        ports = self.list_available_ports()
        
        # Look for common USB-serial adapters
        keywords = ["USB", "Serial", "UART", "CH340", "FTDI", "CP210"]
        
        for port in ports:
            desc = f"{port['description']} {port['hwid']}".upper()
            if any(kw.upper() in desc for kw in keywords):
                print(f"Auto-detected port: {port['device']} ({port['description']})")
                return port["device"]
        
        # Fall back to first available port
        if ports:
            print(f"Using first available port: {ports[0]['device']}")
            return ports[0]["device"]
            
        return None
    
    def disconnect(self):
        """Close the serial connection."""
        self.stop_acquisition()
        if self._serial and self._serial.is_open:
            self._serial.close()
            print("Disconnected from EEG device")
    
    def register_callback(self, callback: Callable[[EEGPacket], None]):
        """Register a callback to receive EEG data packets."""
        self._callbacks.append(callback)
    
    def start_acquisition(self):
        """Start reading EEG data in a background thread."""
        if self._running:
            return
            
        if not self._serial or not self._serial.is_open:
            if not self.connect():
                return
        
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("Started EEG acquisition")
    
    def stop_acquisition(self):
        """Stop the data acquisition thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("Stopped EEG acquisition")
    
    def _read_loop(self):
        """
        Main reading loop - runs in background thread.
        
        NOTE: The actual parsing depends on your specific EEG hardware protocol.
        This is a template that assumes binary packets. You'll need to modify
        this based on your device's documentation.
        """
        import time
        
        # Buffer for incoming data
        buffer = bytearray()
        
        # Expected packet size (this is hardware-specific!)
        # Common format: [SYNC][SEQ][CH1_H][CH1_L][CH2_H][CH2_L]...[CHECKSUM]
        # For 21 channels with 16-bit samples: 2 + 1 + (21 * 2) + 1 = 46 bytes
        SYNC_BYTES = b'\xAA\x55'  # Example sync pattern
        PACKET_SIZE = 2 + 1 + (self.n_channels * 2) + 1
        
        while self._running:
            try:
                # Read available data
                if self._serial.in_waiting > 0:
                    buffer.extend(self._serial.read(self._serial.in_waiting))
                
                # Look for complete packets
                while len(buffer) >= PACKET_SIZE:
                    # Find sync bytes
                    sync_idx = buffer.find(SYNC_BYTES)
                    
                    if sync_idx == -1:
                        # No sync found, keep last byte in case it's start of sync
                        buffer = buffer[-1:]
                        break
                    
                    if sync_idx > 0:
                        # Discard bytes before sync
                        buffer = buffer[sync_idx:]
                    
                    if len(buffer) < PACKET_SIZE:
                        break
                    
                    # Parse packet
                    packet = self._parse_packet(bytes(buffer[:PACKET_SIZE]))
                    if packet:
                        self._dispatch_packet(packet)
                    
                    buffer = buffer[PACKET_SIZE:]
                
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
                
            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(0.1)
    
    def _parse_packet(self, data: bytes) -> Optional[EEGPacket]:
        """
        Parse a raw data packet into an EEGPacket.
        
        NOTE: This is a template - modify based on your hardware's protocol!
        """
        import time
        
        try:
            # Skip sync bytes (first 2 bytes)
            seq = data[2]
            
            # Parse channel data (assuming 16-bit signed, big-endian)
            channels = np.zeros(self.n_channels, dtype=np.float64)
            for i in range(self.n_channels):
                offset = 3 + (i * 2)
                # Convert to signed 16-bit value
                raw_value = int.from_bytes(data[offset:offset+2], byteorder='big', signed=True)
                # Convert to microvolts (scaling depends on hardware gain)
                # Typical EEG ADC: 24-bit with ~0.02 µV/LSB, or 16-bit with ~0.5 µV/LSB
                channels[i] = raw_value * 0.5  # Adjust scaling factor!
            
            return EEGPacket(
                timestamp=time.time(),
                channels=channels,
                sequence_number=seq,
            )
            
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def _dispatch_packet(self, packet: EEGPacket):
        """Send packet to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(packet)
            except Exception as e:
                print(f"Callback error: {e}")


# Protocol detection helper
def detect_protocol(port: str, timeout: float = 5.0) -> dict:
    """
    Attempt to detect the EEG device protocol.
    
    Args:
        port: Serial port to test.
        timeout: How long to wait for data.
        
    Returns:
        Dict with detected protocol info.
    """
    import time
    
    results = {
        "detected": False,
        "baud_rate": None,
        "data_format": None,
        "sample_bytes": None,
    }
    
    for baud in SerialEEGReader.COMMON_BAUD_RATES:
        try:
            with serial.Serial(port, baud, timeout=1.0) as ser:
                start = time.time()
                data = bytearray()
                
                while time.time() - start < timeout:
                    if ser.in_waiting:
                        data.extend(ser.read(ser.in_waiting))
                        if len(data) > 100:
                            break
                    time.sleep(0.1)
                
                if len(data) > 50:
                    results["detected"] = True
                    results["baud_rate"] = baud
                    results["sample_bytes"] = bytes(data[:100])
                    
                    # Try to detect format
                    if all(32 <= b < 127 or b in (10, 13) for b in data[:50]):
                        results["data_format"] = "ASCII"
                    else:
                        results["data_format"] = "BINARY"
                    
                    return results
                    
        except serial.SerialException:
            continue
    
    return results
