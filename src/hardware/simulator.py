"""
EEG Simulator - Generate realistic synthetic EEG data for testing.

This allows testing the dashboard before hardware arrives.
"""

import numpy as np
from typing import Callable, Optional
import threading
import time
from dataclasses import dataclass


@dataclass
class EEGPacket:
    """Represents a single EEG data packet."""
    timestamp: float
    channels: np.ndarray
    sequence_number: int


class EEGSimulator:
    """
    Generates realistic synthetic EEG signals for testing.
    
    Simulates:
    - Background brain rhythms (alpha, beta, theta, delta)
    - Eye blink artifacts
    - Muscle artifacts
    - 50/60 Hz line noise
    - Realistic channel correlations (nearby electrodes are correlated)
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        sample_rate: int = 256,
        noise_level: float = 1.0,
    ):
        """
        Initialize the EEG simulator.
        
        Args:
            n_channels: Number of EEG channels to simulate.
            sample_rate: Samples per second.
            noise_level: Multiplier for noise amplitude (1.0 = realistic).
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[EEGPacket], None]] = []
        self._sequence = 0
        self._time = 0.0
        
        # Pre-generate some parameters for each channel
        self._channel_params = self._init_channel_params()
        
    def _init_channel_params(self) -> list[dict]:
        """Initialize random parameters for each channel to create variety."""
        params = []
        for i in range(self.n_channels):
            params.append({
                # Alpha wave (8-13 Hz) - dominant in occipital
                "alpha_freq": np.random.uniform(9, 11),
                "alpha_amp": np.random.uniform(10, 30),  # µV
                # Beta wave (13-30 Hz) - frontal/central
                "beta_freq": np.random.uniform(15, 25),
                "beta_amp": np.random.uniform(5, 15),
                # Theta wave (4-8 Hz)
                "theta_freq": np.random.uniform(5, 7),
                "theta_amp": np.random.uniform(5, 10),
                # Delta wave (0.5-4 Hz)
                "delta_freq": np.random.uniform(1, 3),
                "delta_amp": np.random.uniform(20, 50),
                # Phase offsets for variety
                "phase": np.random.uniform(0, 2 * np.pi),
            })
        return params
    
    def register_callback(self, callback: Callable[[EEGPacket], None]):
        """Register a callback to receive simulated EEG data."""
        self._callbacks.append(callback)
    
    def start_acquisition(self):
        """Start generating simulated EEG data."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        print("Started EEG simulation")
    
    def stop_acquisition(self):
        """Stop the simulation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("Stopped EEG simulation")
    
    def connect(self) -> bool:
        """Fake connect for API compatibility."""
        print("Simulator connected (no hardware)")
        return True
    
    def disconnect(self):
        """Fake disconnect for API compatibility."""
        self.stop_acquisition()
        print("Simulator disconnected")
    
    def _generate_loop(self):
        """Main generation loop."""
        interval = 1.0 / self.sample_rate
        next_sample_time = time.time()
        
        while self._running:
            current_time = time.time()
            
            if current_time >= next_sample_time:
                # Generate and dispatch sample
                packet = self._generate_sample()
                self._dispatch_packet(packet)
                
                self._time += interval
                self._sequence = (self._sequence + 1) % 256
                next_sample_time += interval
                
                # If we're falling behind, catch up
                if current_time - next_sample_time > 0.1:
                    next_sample_time = current_time
            else:
                # Sleep until next sample
                sleep_time = next_sample_time - current_time
                if sleep_time > 0.0001:
                    time.sleep(sleep_time * 0.9)
    
    def _generate_sample(self) -> EEGPacket:
        """Generate a single EEG sample with realistic characteristics."""
        t = self._time
        channels = np.zeros(self.n_channels)
        
        for i, params in enumerate(self._channel_params):
            phase = params["phase"]
            
            # Alpha rhythm (most prominent when eyes closed, relaxed)
            alpha = params["alpha_amp"] * np.sin(2 * np.pi * params["alpha_freq"] * t + phase)
            
            # Beta rhythm
            beta = params["beta_amp"] * np.sin(2 * np.pi * params["beta_freq"] * t + phase * 1.3)
            
            # Theta rhythm
            theta = params["theta_amp"] * np.sin(2 * np.pi * params["theta_freq"] * t + phase * 0.7)
            
            # Delta rhythm (slow, high amplitude)
            delta = params["delta_amp"] * np.sin(2 * np.pi * params["delta_freq"] * t + phase * 0.5)
            
            # Combine rhythms (alpha usually dominates in awake, relaxed state)
            signal = 0.4 * alpha + 0.2 * beta + 0.2 * theta + 0.1 * delta
            
            # Add pink noise (1/f noise characteristic of EEG)
            pink_noise = self._generate_pink_noise() * 5 * self.noise_level
            
            # Add 60 Hz line noise (common artifact)
            line_noise = 2 * np.sin(2 * np.pi * 60 * t) * self.noise_level
            
            # Random eye blink artifacts (frontal channels)
            if i < 4 and np.random.random() < 0.002:  # Occasional blinks
                signal += np.random.uniform(50, 150)  # Large deflection
            
            channels[i] = signal + pink_noise + line_noise
        
        # Add spatial correlation (nearby channels should be similar)
        channels = self._add_spatial_correlation(channels)
        
        return EEGPacket(
            timestamp=time.time(),
            channels=channels,
            sequence_number=self._sequence,
        )
    
    def _generate_pink_noise(self) -> float:
        """Generate a single sample of pink (1/f) noise."""
        # Simple approximation using filtered white noise
        # For true pink noise, you'd use a more sophisticated filter
        return np.random.randn() * 0.5
    
    def _add_spatial_correlation(self, channels: np.ndarray) -> np.ndarray:
        """Add realistic spatial correlation between nearby electrodes."""
        # Simple correlation: average with neighbors
        correlated = channels.copy()
        for i in range(1, len(channels) - 1):
            correlated[i] = 0.6 * channels[i] + 0.2 * channels[i-1] + 0.2 * channels[i+1]
        return correlated
    
    def _dispatch_packet(self, packet: EEGPacket):
        """Send packet to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(packet)
            except Exception as e:
                print(f"Callback error: {e}")


# Convenience function for quick testing
def generate_test_data(duration: float = 10.0, sample_rate: int = 256, n_channels: int = 21) -> np.ndarray:
    """
    Generate a batch of synthetic EEG data for testing.
    
    Args:
        duration: Duration in seconds.
        sample_rate: Samples per second.
        n_channels: Number of channels.
        
    Returns:
        Array of shape (n_samples, n_channels) in microvolts.
    """
    n_samples = int(duration * sample_rate)
    data = []
    
    sim = EEGSimulator(n_channels=n_channels, sample_rate=sample_rate)
    
    for _ in range(n_samples):
        packet = sim._generate_sample()
        data.append(packet.channels)
        sim._time += 1.0 / sample_rate
    
    return np.array(data)
