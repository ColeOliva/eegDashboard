"""
Session Logging and Data Management.

Provides structured logging for EEG recording sessions with:
- Session metadata
- Event markers
- Performance metrics
- Automatic file naming
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SessionMetadata:
    """Metadata for an EEG recording session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    sample_rate: float = 256.0
    n_channels: int = 21
    channel_names: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    total_samples: int = 0
    source: str = "unknown"  # "simulator" or "serial:<port>"
    filter_settings: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass  
class EventMarker:
    """An event marker during recording."""
    timestamp: float
    event_type: str
    description: str
    data: Optional[Dict] = None


class SessionLogger:
    """
    Manages logging for EEG recording sessions.
    
    Features:
    - Automatic session ID generation
    - Event markers with timestamps
    - Performance metrics tracking
    - JSON export of session data
    """
    
    def __init__(self, output_dir: Optional[Path] = None, log_level: int = logging.INFO):
        self.output_dir = Path(output_dir) if output_dir else Path.home() / "eeg_sessions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[SessionMetadata] = None
        self.events: List[EventMarker] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "filter_latency_ms": [],
            "render_latency_ms": [],
            "samples_per_second": [],
        }
        
        # Set up logging
        self.logger = logging.getLogger("eeg_session")
        self.logger.setLevel(log_level)
        
        # File handler for session log
        self._file_handler: Optional[logging.FileHandler] = None
        
    def start_session(
        self,
        source: str,
        sample_rate: float = 256.0,
        n_channels: int = 21,
        channel_names: Optional[List[str]] = None,
        filter_settings: Optional[Dict] = None,
        notes: str = "",
    ) -> str:
        """Start a new recording session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_session = SessionMetadata(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            sample_rate=sample_rate,
            n_channels=n_channels,
            channel_names=channel_names or [],
            source=source,
            filter_settings=filter_settings or {},
            notes=notes,
        )
        
        self.events = []
        self.performance_metrics = {k: [] for k in self.performance_metrics}
        
        # Set up file logging
        log_file = self.output_dir / f"session_{session_id}.log"
        self._file_handler = logging.FileHandler(log_file)
        self._file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(self._file_handler)
        
        self.logger.info(f"Session started: {session_id}")
        self.logger.info(f"Source: {source}, Sample rate: {sample_rate} Hz, Channels: {n_channels}")
        
        return session_id
        
    def end_session(self) -> Optional[Path]:
        """End the current session and save metadata."""
        if not self.current_session:
            return None
            
        self.current_session.end_time = datetime.now().isoformat()
        
        # Calculate performance summary
        perf_summary = {}
        for metric, values in self.performance_metrics.items():
            if values:
                perf_summary[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
        
        # Save session JSON
        session_data = {
            "metadata": asdict(self.current_session),
            "events": [asdict(e) for e in self.events],
            "performance": perf_summary,
        }
        
        json_file = self.output_dir / f"session_{self.current_session.session_id}.json"
        with open(json_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.logger.info(f"Session ended: {self.current_session.session_id}")
        self.logger.info(f"Duration: {self.current_session.duration_seconds:.1f}s, Samples: {self.current_session.total_samples}")
        
        # Clean up
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        
        session_id = self.current_session.session_id
        self.current_session = None
        
        return json_file
        
    def add_event(self, event_type: str, description: str, data: Optional[Dict] = None):
        """Add an event marker."""
        if not self.current_session:
            return
            
        import time
        event = EventMarker(
            timestamp=time.time(),
            event_type=event_type,
            description=description,
            data=data,
        )
        self.events.append(event)
        self.logger.info(f"Event: [{event_type}] {description}")
        
    def update_sample_count(self, count: int, duration: float):
        """Update sample count and duration."""
        if self.current_session:
            self.current_session.total_samples = count
            self.current_session.duration_seconds = duration
            
    def record_latency(self, metric: str, latency_ms: float):
        """Record a latency measurement."""
        if metric in self.performance_metrics:
            self.performance_metrics[metric].append(latency_ms)
            
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message)


class DataBuffer:
    """
    Efficient data buffer for recording.
    
    Uses pre-allocated numpy arrays that grow as needed.
    """
    
    def __init__(self, n_channels: int, initial_capacity: int = 256 * 60):
        self.n_channels = n_channels
        self._capacity = initial_capacity
        self._data = np.zeros((initial_capacity, n_channels), dtype=np.float32)
        self._timestamps = np.zeros(initial_capacity, dtype=np.float64)
        self._count = 0
        
    def add_sample(self, sample: np.ndarray, timestamp: float):
        """Add a single sample."""
        if self._count >= self._capacity:
            self._grow()
            
        self._data[self._count] = sample
        self._timestamps[self._count] = timestamp
        self._count += 1
        
    def add_chunk(self, data: np.ndarray, timestamps: np.ndarray):
        """Add a chunk of samples."""
        n_new = len(data)
        while self._count + n_new > self._capacity:
            self._grow()
            
        self._data[self._count:self._count + n_new] = data
        self._timestamps[self._count:self._count + n_new] = timestamps
        self._count += n_new
        
    def _grow(self):
        """Double the buffer capacity."""
        new_capacity = self._capacity * 2
        new_data = np.zeros((new_capacity, self.n_channels), dtype=np.float32)
        new_timestamps = np.zeros(new_capacity, dtype=np.float64)
        
        new_data[:self._count] = self._data[:self._count]
        new_timestamps[:self._count] = self._timestamps[:self._count]
        
        self._data = new_data
        self._timestamps = new_timestamps
        self._capacity = new_capacity
        
    def get_data(self) -> np.ndarray:
        """Get recorded data."""
        return self._data[:self._count].copy()
        
    def get_timestamps(self) -> np.ndarray:
        """Get timestamps."""
        return self._timestamps[:self._count].copy()
        
    def clear(self):
        """Clear the buffer."""
        self._count = 0
        
    @property
    def count(self) -> int:
        return self._count
        
    @property
    def duration(self) -> float:
        """Duration in seconds based on timestamps."""
        if self._count < 2:
            return 0.0
        return self._timestamps[self._count - 1] - self._timestamps[0]
