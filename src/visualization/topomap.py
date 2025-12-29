"""
Brain Topographic Map (Topomap) Visualization.

Creates 2D scalp topography plots showing the spatial distribution
of EEG activity across the head - the colorful "heat map" brain images
you see in EEG research papers.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.interpolate import CloughTocher2DInterpolator, griddata
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
from io import BytesIO


# Standard 10-20 electrode positions (normalized to unit circle)
# Positions are in (x, y) where the nose is at top (y=1)
# and left ear is at x=-1
ELECTRODE_POSITIONS = {
    # Frontal polar
    "Fp1": (-0.22, 0.83),
    "Fp2": (0.22, 0.83),
    
    # Frontal
    "F7": (-0.59, 0.59),
    "F3": (-0.32, 0.50),
    "Fz": (0.0, 0.50),
    "F4": (0.32, 0.50),
    "F8": (0.59, 0.59),
    
    # Temporal
    "T3": (-0.81, 0.0),
    "T4": (0.81, 0.0),
    "T5": (-0.59, -0.59),
    "T6": (0.59, -0.59),
    
    # Central
    "C3": (-0.38, 0.0),
    "Cz": (0.0, 0.0),
    "C4": (0.38, 0.0),
    
    # Parietal
    "P3": (-0.32, -0.50),
    "Pz": (0.0, -0.50),
    "P4": (0.32, -0.50),
    
    # Occipital
    "O1": (-0.22, -0.83),
    "Oz": (0.0, -0.83),
    "O2": (0.22, -0.83),
    
    # Reference (ear lobes)
    "A1": (-0.95, 0.0),
    "A2": (0.95, 0.0),
}

# Channel order matching the standard 10-20 layout
STANDARD_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "Oz", "O2"
]


class TopoMap:
    """
    Creates topographic maps of EEG activity.
    
    Interpolates electrode values across the scalp surface
    and displays as a colored 2D head map.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        colormap: str = "RdBu_r",
        head_radius: float = 1.0,
        resolution: int = 100,
    ):
        """
        Initialize the topomap.
        
        Args:
            channel_names: List of channel names in data order.
            colormap: Matplotlib colormap name.
            head_radius: Radius of head circle.
            resolution: Grid resolution for interpolation.
        """
        self.channel_names = channel_names or STANDARD_CHANNEL_ORDER
        self.colormap = colormap
        self.head_radius = head_radius
        self.resolution = resolution
        
        # Get electrode positions for our channels
        self.positions = []
        self.valid_channels = []
        for i, name in enumerate(self.channel_names):
            if name in ELECTRODE_POSITIONS:
                self.positions.append(ELECTRODE_POSITIONS[name])
                self.valid_channels.append(i)
        
        self.positions = np.array(self.positions)
        
        # Create interpolation grid
        self._create_grid()
        
    def _create_grid(self):
        """Create the interpolation grid."""
        # Create circular grid
        x = np.linspace(-self.head_radius, self.head_radius, self.resolution)
        y = np.linspace(-self.head_radius, self.head_radius, self.resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        
        # Create mask for points outside head
        self.head_mask = (self.grid_x**2 + self.grid_y**2) <= self.head_radius**2
        
    def interpolate(self, values: np.ndarray, method: str = 'cubic') -> np.ndarray:
        """
        Interpolate electrode values across the scalp.
        
        Args:
            values: Array of values for each channel.
            method: Interpolation method ('cubic', 'linear', 'nearest').
            
        Returns:
            2D array of interpolated values.
        """
        # Get values for valid channels only
        valid_values = values[self.valid_channels]
        
        # Interpolate
        grid_values = griddata(
            self.positions,
            valid_values,
            (self.grid_x, self.grid_y),
            method=method,
            fill_value=np.nan
        )
        
        # Apply head mask
        grid_values[~self.head_mask] = np.nan
        
        return grid_values
    
    def create_figure(
        self,
        values: np.ndarray,
        title: str = "",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_electrodes: bool = True,
        show_names: bool = False,
        show_colorbar: bool = True,
        figsize: Tuple[float, float] = (6, 6),
    ) -> plt.Figure:
        """
        Create a topomap figure.
        
        Args:
            values: Array of values for each channel.
            title: Plot title.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.
            show_electrodes: Show electrode positions as dots.
            show_names: Show electrode names.
            show_colorbar: Show colorbar.
            figsize: Figure size in inches.
            
        Returns:
            Matplotlib Figure object.
        """
        # Interpolate values
        grid_values = self.interpolate(values)
        
        # Set color limits
        if vmin is None:
            vmin = np.nanmin(grid_values)
        if vmax is None:
            vmax = np.nanmax(grid_values)
        
        # Symmetric limits for diverging colormap
        if 'RdBu' in self.colormap or 'coolwarm' in self.colormap:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis('off')
        
        # Draw head outline
        self._draw_head(ax)
        
        # Draw interpolated values
        im = ax.contourf(
            self.grid_x, self.grid_y, grid_values,
            levels=50,
            cmap=self.colormap,
            vmin=vmin, vmax=vmax,
            extend='both'
        )
        
        # Draw contour lines
        ax.contour(
            self.grid_x, self.grid_y, grid_values,
            levels=10,
            colors='gray',
            linewidths=0.5,
            alpha=0.5
        )
        
        # Draw electrodes
        if show_electrodes:
            ax.scatter(
                self.positions[:, 0],
                self.positions[:, 1],
                c='black',
                s=20,
                zorder=5
            )
        
        # Draw electrode names
        if show_names:
            for i, name in enumerate(self.channel_names):
                if name in ELECTRODE_POSITIONS:
                    pos = ELECTRODE_POSITIONS[name]
                    ax.annotate(
                        name,
                        pos,
                        fontsize=6,
                        ha='center',
                        va='bottom',
                        xytext=(0, 3),
                        textcoords='offset points'
                    )
        
        # Colorbar
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            cbar.ax.tick_params(labelsize=8)
        
        # Title
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _draw_head(self, ax):
        """Draw the head outline, nose, and ears."""
        # Head circle
        head = Circle(
            (0, 0),
            self.head_radius,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(head)
        
        # Nose (triangle at top)
        nose_x = [0, -0.08, 0.08, 0]
        nose_y = [1.15, 1.0, 1.0, 1.15]
        ax.plot(nose_x, nose_y, 'k-', linewidth=2)
        
        # Left ear
        ear_left = Wedge(
            (-1.0, 0), 0.12, 90, 270,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(ear_left)
        
        # Right ear
        ear_right = Wedge(
            (1.0, 0), 0.12, -90, 90,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(ear_right)
        
    def to_image_bytes(
        self,
        values: np.ndarray,
        format: str = 'png',
        **kwargs
    ) -> bytes:
        """
        Create topomap and return as image bytes.
        
        Useful for embedding in Qt or web interfaces.
        
        Args:
            values: Array of values for each channel.
            format: Image format ('png', 'jpg', 'svg').
            **kwargs: Additional arguments for create_figure.
            
        Returns:
            Image as bytes.
        """
        fig = self.create_figure(values, **kwargs)
        
        buf = BytesIO()
        fig.savefig(buf, format=format, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        buf.seek(0)
        return buf.read()
    
    def to_numpy_image(self, values: np.ndarray, **kwargs) -> np.ndarray:
        """
        Create topomap and return as numpy array (RGB).
        
        Args:
            values: Array of values for each channel.
            **kwargs: Additional arguments for create_figure.
            
        Returns:
            RGB image as numpy array (H, W, 3).
        """
        fig = self.create_figure(values, **kwargs)
        
        # Draw canvas to buffer
        fig.canvas.draw()
        
        # Get RGB array
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Drop alpha
        
        plt.close(fig)
        return img


class BandTopoMaps:
    """
    Creates multiple topomaps for different frequency bands.
    
    Shows spatial distribution of delta, theta, alpha, beta, gamma
    activity across the scalp.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
    ):
        """
        Initialize band topomaps.
        
        Args:
            channel_names: List of channel names.
        """
        self.channel_names = channel_names or STANDARD_CHANNEL_ORDER
        
        # Colormaps for each band
        self.band_colormaps = {
            "delta": "Purples",
            "theta": "Blues", 
            "alpha": "Greens",
            "beta": "Oranges",
            "gamma": "Reds",
        }
        
        # Create topomap for each band
        self.topomaps = {
            band: TopoMap(channel_names=self.channel_names, colormap=cmap)
            for band, cmap in self.band_colormaps.items()
        }
    
    def create_figure(
        self,
        band_powers: Dict[str, np.ndarray],
        title: str = "Band Power Distribution",
        figsize: Tuple[float, float] = (15, 3),
    ) -> plt.Figure:
        """
        Create a figure with topomaps for all bands.
        
        Args:
            band_powers: Dict mapping band name to power values per channel.
            title: Overall figure title.
            figsize: Figure size.
            
        Returns:
            Matplotlib Figure object.
        """
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
        n_bands = len(bands)
        
        fig, axes = plt.subplots(1, n_bands, figsize=figsize)
        
        for i, band in enumerate(bands):
            ax = axes[i]
            ax.set_aspect('equal')
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.axis('off')
            
            topomap = self.topomaps[band]
            values = band_powers.get(band, np.zeros(len(self.channel_names)))
            
            # Interpolate
            grid_values = topomap.interpolate(values)
            
            # Draw head
            topomap._draw_head(ax)
            
            # Draw heatmap
            vmax = np.nanmax(np.abs(grid_values)) if not np.all(np.isnan(grid_values)) else 1
            im = ax.contourf(
                topomap.grid_x, topomap.grid_y, grid_values,
                levels=30,
                cmap=self.band_colormaps[band],
                vmin=0, vmax=vmax
            )
            
            # Draw electrodes
            ax.scatter(
                topomap.positions[:, 0],
                topomap.positions[:, 1],
                c='black',
                s=10,
                zorder=5
            )
            
            ax.set_title(f"{band.capitalize()}\n({self._get_freq_range(band)})", fontsize=10)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
    
    def _get_freq_range(self, band: str) -> str:
        """Get frequency range string for band."""
        ranges = {
            "delta": "0.5-4 Hz",
            "theta": "4-8 Hz",
            "alpha": "8-13 Hz",
            "beta": "13-30 Hz",
            "gamma": "30-100 Hz",
        }
        return ranges.get(band, "")


def create_topomap_animation_frame(
    topomap: TopoMap,
    values: np.ndarray,
    frame_number: int,
    total_frames: int,
    **kwargs
) -> np.ndarray:
    """
    Create a single frame for topomap animation.
    
    Args:
        topomap: TopoMap instance.
        values: Values for this frame.
        frame_number: Current frame index.
        total_frames: Total number of frames.
        **kwargs: Additional arguments for create_figure.
        
    Returns:
        RGB image array.
    """
    title = kwargs.pop('title', '')
    if total_frames > 1:
        title = f"{title} (frame {frame_number+1}/{total_frames})"
    
    return topomap.to_numpy_image(values, title=title, **kwargs)
