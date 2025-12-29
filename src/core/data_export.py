"""
EEG Data Export Module.

Supports saving EEG recordings to standard formats:
- CSV (simple, universal)
- EDF/EDF+ (European Data Format - standard for EEG)
- NumPy (.npy) for fast loading
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime
import csv


def save_csv(
    filename: Union[str, Path],
    data: np.ndarray,
    channel_names: List[str],
    sample_rate: float,
    timestamps: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save EEG data to CSV format.
    
    Args:
        filename: Output file path.
        data: EEG data array of shape (n_samples, n_channels).
        channel_names: List of channel names.
        sample_rate: Sampling rate in Hz.
        timestamps: Optional array of timestamps.
        metadata: Optional metadata dict to include in header.
    """
    filename = Path(filename)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write metadata as comments
        writer.writerow([f"# EEG Recording"])
        writer.writerow([f"# Date: {datetime.now().isoformat()}"])
        writer.writerow([f"# Sample Rate: {sample_rate} Hz"])
        writer.writerow([f"# Channels: {len(channel_names)}"])
        writer.writerow([f"# Samples: {len(data)}"])
        writer.writerow([f"# Duration: {len(data) / sample_rate:.2f} seconds"])
        
        if metadata:
            for key, value in metadata.items():
                writer.writerow([f"# {key}: {value}"])
        
        # Header row
        if timestamps is not None:
            header = ['timestamp'] + channel_names
        else:
            header = ['sample'] + channel_names
        writer.writerow(header)
        
        # Data rows
        for i, row in enumerate(data):
            if timestamps is not None:
                line = [timestamps[i]] + list(row)
            else:
                line = [i] + list(row)
            writer.writerow(line)


def load_csv(filename: Union[str, Path]) -> Dict:
    """
    Load EEG data from CSV format.
    
    Args:
        filename: Input file path.
        
    Returns:
        Dict with 'data', 'channel_names', 'timestamps', 'metadata'.
    """
    filename = Path(filename)
    
    metadata = {}
    header = None
    data_rows = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        
        for row in reader:
            if not row:
                continue
            
            # Metadata comments
            if row[0].startswith('#'):
                line = row[0][1:].strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                continue
            
            # Header
            if header is None:
                header = row
                continue
            
            # Data
            data_rows.append([float(x) for x in row])
    
    data_array = np.array(data_rows)
    
    # Extract timestamps if present
    if header and header[0] in ('timestamp', 'sample'):
        timestamps = data_array[:, 0]
        data = data_array[:, 1:]
        channel_names = header[1:]
    else:
        timestamps = None
        data = data_array
        channel_names = header
    
    return {
        'data': data,
        'channel_names': channel_names,
        'timestamps': timestamps,
        'metadata': metadata,
    }


def save_edf(
    filename: Union[str, Path],
    data: np.ndarray,
    channel_names: List[str],
    sample_rate: float,
    patient_info: Optional[Dict] = None,
) -> None:
    """
    Save EEG data to EDF (European Data Format).
    
    EDF is the standard format for EEG recordings and is
    supported by most EEG analysis software.
    
    Args:
        filename: Output file path.
        data: EEG data array of shape (n_samples, n_channels).
        channel_names: List of channel names.
        sample_rate: Sampling rate in Hz.
        patient_info: Optional patient information dict.
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib is required for EDF export. Install with: pip install pyedflib")
    
    filename = Path(filename)
    n_samples, n_channels = data.shape
    
    # Create EDF file
    f = pyedflib.EdfWriter(str(filename), n_channels)
    
    try:
        # Set patient info
        if patient_info:
            f.setPatientName(patient_info.get('name', ''))
            f.setPatientCode(patient_info.get('code', ''))
            f.setGender(patient_info.get('gender', ''))
            f.setBirthdate(patient_info.get('birthdate', datetime(1900, 1, 1)))
        
        # Set recording info
        f.setStartdatetime(datetime.now())
        
        # Configure channels
        for i, name in enumerate(channel_names):
            # Channel header
            channel_info = {
                'label': name,
                'dimension': 'uV',
                'sample_frequency': sample_rate,
                'physical_min': -500.0,
                'physical_max': 500.0,
                'digital_min': -32768,
                'digital_max': 32767,
                'transducer': 'EEG electrode',
                'prefilter': 'HP:0.1Hz LP:100Hz',
            }
            f.setSignalHeader(i, channel_info)
        
        # Write data
        # EDF stores data per channel, so we need to transpose
        for i in range(n_channels):
            f.writePhysicalSamples(data[:, i])
            
    finally:
        f.close()


def load_edf(filename: Union[str, Path]) -> Dict:
    """
    Load EEG data from EDF format.
    
    Args:
        filename: Input file path.
        
    Returns:
        Dict with 'data', 'channel_names', 'sample_rate', 'metadata'.
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib is required for EDF import. Install with: pip install pyedflib")
    
    filename = Path(filename)
    
    f = pyedflib.EdfReader(str(filename))
    
    try:
        n_channels = f.signals_in_file
        channel_names = f.getSignalLabels()
        sample_rate = f.getSampleFrequency(0)  # Assume all same rate
        
        # Read all channels
        n_samples = f.getNSamples()[0]
        data = np.zeros((n_samples, n_channels))
        
        for i in range(n_channels):
            data[:, i] = f.readSignal(i)
        
        # Get metadata
        metadata = {
            'patient_name': f.getPatientName(),
            'start_datetime': f.getStartdatetime(),
            'duration': f.getFileDuration(),
        }
        
    finally:
        f.close()
    
    return {
        'data': data,
        'channel_names': channel_names,
        'sample_rate': sample_rate,
        'metadata': metadata,
    }


def save_numpy(
    filename: Union[str, Path],
    data: np.ndarray,
    channel_names: List[str],
    sample_rate: float,
    **metadata
) -> None:
    """
    Save EEG data to NumPy format (.npz).
    
    Fast for loading back into Python, but not universal.
    
    Args:
        filename: Output file path.
        data: EEG data array.
        channel_names: List of channel names.
        sample_rate: Sampling rate in Hz.
        **metadata: Additional metadata to save.
    """
    filename = Path(filename)
    
    np.savez(
        filename,
        data=data,
        channel_names=channel_names,
        sample_rate=sample_rate,
        **metadata
    )


def load_numpy(filename: Union[str, Path]) -> Dict:
    """
    Load EEG data from NumPy format.
    
    Args:
        filename: Input file path.
        
    Returns:
        Dict with data and metadata.
    """
    filename = Path(filename)
    
    loaded = np.load(filename, allow_pickle=True)
    
    result = {}
    for key in loaded.files:
        value = loaded[key]
        # Convert 0-d arrays back to scalars
        if value.ndim == 0:
            result[key] = value.item()
        else:
            result[key] = value
    
    return result


class EEGRecorder:
    """
    Handles real-time recording of EEG data.
    
    Buffers incoming data and saves to file.
    """
    
    def __init__(
        self,
        channel_names: List[str],
        sample_rate: float = 256.0,
    ):
        """
        Initialize recorder.
        
        Args:
            channel_names: List of channel names.
            sample_rate: Sampling rate in Hz.
        """
        self.channel_names = channel_names
        self.sample_rate = sample_rate
        
        self.recording = False
        self.data_buffer: List[np.ndarray] = []
        self.timestamp_buffer: List[float] = []
        self.start_time: Optional[float] = None
        
    def start(self):
        """Start recording."""
        self.recording = True
        self.data_buffer = []
        self.timestamp_buffer = []
        self.start_time = None
        
    def stop(self) -> int:
        """
        Stop recording.
        
        Returns:
            Number of samples recorded.
        """
        self.recording = False
        return len(self.data_buffer)
    
    def add_sample(self, sample: np.ndarray, timestamp: Optional[float] = None):
        """
        Add a sample to the recording.
        
        Args:
            sample: Array of channel values.
            timestamp: Optional timestamp.
        """
        if not self.recording:
            return
            
        if timestamp is None:
            import time
            timestamp = time.time()
            
        if self.start_time is None:
            self.start_time = timestamp
            
        self.data_buffer.append(sample.copy())
        self.timestamp_buffer.append(timestamp - self.start_time)
    
    def get_data(self) -> np.ndarray:
        """Get recorded data as array."""
        if not self.data_buffer:
            return np.array([])
        return np.array(self.data_buffer)
    
    def get_timestamps(self) -> np.ndarray:
        """Get timestamps as array."""
        return np.array(self.timestamp_buffer)
    
    def get_duration(self) -> float:
        """Get recording duration in seconds."""
        return len(self.data_buffer) / self.sample_rate
    
    def save(self, filename: Union[str, Path], format: str = 'csv') -> str:
        """
        Save recording to file.
        
        Args:
            filename: Output file path.
            format: Format to save ('csv', 'edf', 'numpy').
            
        Returns:
            Actual filename used.
        """
        filename = Path(filename)
        data = self.get_data()
        timestamps = self.get_timestamps()
        
        if len(data) == 0:
            raise ValueError("No data to save")
        
        if format == 'csv':
            if not filename.suffix:
                filename = filename.with_suffix('.csv')
            save_csv(filename, data, self.channel_names, self.sample_rate, timestamps)
            
        elif format == 'edf':
            if not filename.suffix:
                filename = filename.with_suffix('.edf')
            save_edf(filename, data, self.channel_names, self.sample_rate)
            
        elif format == 'numpy':
            if not filename.suffix:
                filename = filename.with_suffix('.npz')
            save_numpy(filename, data, self.channel_names, self.sample_rate, timestamps=timestamps)
            
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return str(filename)
    
    def clear(self):
        """Clear all recorded data."""
        self.data_buffer = []
        self.timestamp_buffer = []
        self.start_time = None
