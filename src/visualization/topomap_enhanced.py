"""
Enhanced Brain Topographic Map Visualization.

Creates publication-quality 2D and pseudo-3D scalp topography plots
with realistic head rendering, brain region overlays, and smooth
interpolation for professional EEG visualization.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Ellipse, FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
from io import BytesIO


# Standard 10-20 electrode positions (normalized to unit circle)
ELECTRODE_POSITIONS = {
    # Frontal polar
    "Fp1": (-0.22, 0.83), "Fp2": (0.22, 0.83),
    # Frontal
    "F7": (-0.59, 0.59), "F3": (-0.32, 0.50), "Fz": (0.0, 0.50),
    "F4": (0.32, 0.50), "F8": (0.59, 0.59),
    # Temporal
    "T3": (-0.81, 0.0), "T4": (0.81, 0.0),
    "T5": (-0.59, -0.59), "T6": (0.59, -0.59),
    # Central
    "C3": (-0.38, 0.0), "Cz": (0.0, 0.0), "C4": (0.38, 0.0),
    # Parietal
    "P3": (-0.32, -0.50), "Pz": (0.0, -0.50), "P4": (0.32, -0.50),
    # Occipital
    "O1": (-0.22, -0.83), "Oz": (0.0, -0.83), "O2": (0.22, -0.83),
    # Reference
    "A1": (-0.95, 0.0), "A2": (0.95, 0.0),
}

# Brain region boundaries (approximate)
BRAIN_REGIONS = {
    "Frontal Lobe": {"center": (0, 0.55), "color": "#E8B4B8", "y_range": (0.3, 1.0)},
    "Parietal Lobe": {"center": (0, -0.2), "color": "#B8D4E8", "y_range": (-0.4, 0.3)},
    "Occipital Lobe": {"center": (0, -0.7), "color": "#B8E8C8", "y_range": (-1.0, -0.4)},
    "Temporal (L)": {"center": (-0.6, 0.0), "color": "#E8E4B8", "x_range": (-1.0, -0.4)},
    "Temporal (R)": {"center": (0.6, 0.0), "color": "#E8E4B8", "x_range": (0.4, 1.0)},
}

STANDARD_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "Oz", "O2"
]


def create_brain_colormap(base_color: str = "RdBu_r") -> LinearSegmentedColormap:
    """Create a smooth colormap optimized for brain activity."""
    # Custom colormap: deep blue -> white -> deep red
    colors = [
        (0.0, '#0d47a1'),   # Deep blue (low activity)
        (0.25, '#42a5f5'),  # Light blue
        (0.45, '#e3f2fd'),  # Very light blue
        (0.5, '#ffffff'),   # White (neutral)
        (0.55, '#ffebee'),  # Very light red
        (0.75, '#ef5350'),  # Light red
        (1.0, '#b71c1c'),   # Deep red (high activity)
    ]
    
    cmap_data = {
        'red': [],
        'green': [],
        'blue': [],
    }
    
    for pos, color in colors:
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        cmap_data['red'].append((pos, r, r))
        cmap_data['green'].append((pos, g, g))
        cmap_data['blue'].append((pos, b, b))
    
    return LinearSegmentedColormap('brain_activity', cmap_data)


class EnhancedTopoMap:
    """
    Enhanced topographic map with realistic head rendering and
    smooth interpolation for professional EEG visualization.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        resolution: int = 200,
        smoothing: float = 0.8,
    ):
        """
        Initialize enhanced topomap.
        
        Args:
            channel_names: Channel names in data order.
            resolution: Grid resolution (higher = smoother).
            smoothing: Gaussian smoothing sigma.
        """
        self.channel_names = channel_names or STANDARD_CHANNEL_ORDER
        self.resolution = resolution
        self.smoothing = smoothing
        self.head_radius = 1.0
        
        # Get electrode positions
        self.positions = []
        self.valid_channels = []
        self.electrode_names = []
        
        for i, name in enumerate(self.channel_names):
            if name in ELECTRODE_POSITIONS:
                self.positions.append(ELECTRODE_POSITIONS[name])
                self.valid_channels.append(i)
                self.electrode_names.append(name)
        
        self.positions = np.array(self.positions)
        self._create_grid()
        self._colormap = create_brain_colormap()
    
    def _create_grid(self):
        """Create high-resolution interpolation grid."""
        x = np.linspace(-1.2, 1.2, self.resolution)
        y = np.linspace(-1.2, 1.2, self.resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        
        # Head mask with soft edges
        dist = np.sqrt(self.grid_x**2 + self.grid_y**2)
        self.head_mask = dist <= self.head_radius
        
        # Soft edge mask for fading at boundaries
        edge_width = 0.1
        self.soft_mask = np.clip((self.head_radius - dist) / edge_width, 0, 1)
    
    def interpolate_rbf(self, values: np.ndarray) -> np.ndarray:
        """
        Interpolate using Radial Basis Functions for smoother results.
        """
        valid_values = values[self.valid_channels]
        
        # Use thin-plate spline RBF for smooth interpolation
        rbf = Rbf(
            self.positions[:, 0],
            self.positions[:, 1],
            valid_values,
            function='thin_plate',
            smooth=0.1
        )
        
        grid_values = rbf(self.grid_x, self.grid_y)
        
        # Apply Gaussian smoothing
        if self.smoothing > 0:
            grid_values = gaussian_filter(grid_values, sigma=self.smoothing)
        
        # Apply soft mask
        grid_values = grid_values * self.soft_mask
        grid_values[~self.head_mask] = np.nan
        
        return grid_values
    
    def create_figure(
        self,
        values: np.ndarray,
        title: str = "",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        style: str = "modern",
        show_electrodes: bool = True,
        show_names: bool = True,
        show_colorbar: bool = True,
        show_regions: bool = False,
        show_gradient_ring: bool = True,
        figsize: Tuple[float, float] = (8, 8),
        dpi: int = 100,
    ) -> plt.Figure:
        """
        Create an enhanced topomap figure.
        
        Args:
            values: Array of values for each channel.
            title: Plot title.
            vmin/vmax: Color scale limits.
            style: Visual style ('modern', 'classic', 'clinical').
            show_electrodes: Show electrode markers.
            show_names: Show electrode labels.
            show_colorbar: Show color scale bar.
            show_regions: Show brain region labels.
            show_gradient_ring: Show 3D-like gradient ring around head.
            figsize: Figure size.
            dpi: Figure DPI.
        """
        # Interpolate values
        grid_values = self.interpolate_rbf(values)
        
        # Auto-scale
        if vmin is None:
            vmin = np.nanpercentile(grid_values, 2)
        if vmax is None:
            vmax = np.nanpercentile(grid_values, 98)
        
        # Symmetric scale
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        
        # Create figure with dark or light background based on style
        if style == "modern":
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            text_color = 'white'
            head_color = '#e0e0e0'
            electrode_color = '#00ff88'
            ring_colors = ['#2d2d44', '#1a1a2e']
        elif style == "clinical":
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')
            ax.set_facecolor('white')
            text_color = 'black'
            head_color = '#333333'
            electrode_color = '#ff4444'
            ring_colors = ['#f0f0f0', 'white']
        else:  # classic
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='#f5f5f5')
            ax.set_facecolor('#f5f5f5')
            text_color = '#333333'
            head_color = '#333333'
            electrode_color = 'black'
            ring_colors = ['#e0e0e0', '#f5f5f5']
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        
        # Draw 3D-like gradient ring
        if show_gradient_ring:
            self._draw_gradient_ring(ax, ring_colors)
        
        # Draw brain activity (main visualization)
        im = ax.imshow(
            grid_values,
            extent=[-1.2, 1.2, -1.2, 1.2],
            origin='lower',
            cmap=self._colormap,
            vmin=vmin, vmax=vmax,
            interpolation='bilinear',
            zorder=1
        )
        
        # Add contour lines for depth
        contour_levels = np.linspace(vmin, vmax, 12)
        ax.contour(
            self.grid_x, self.grid_y, grid_values,
            levels=contour_levels,
            colors='white' if style == "modern" else 'gray',
            linewidths=0.3,
            alpha=0.4,
            zorder=2
        )
        
        # Draw head outline
        self._draw_head_3d(ax, head_color, style)
        
        # Draw electrodes
        if show_electrodes:
            self._draw_electrodes(ax, electrode_color, show_names, style)
        
        # Draw brain regions
        if show_regions:
            self._draw_regions(ax, text_color)
        
        # Colorbar
        if show_colorbar:
            self._draw_colorbar(fig, ax, im, text_color, style)
        
        # Title
        if title:
            ax.set_title(
                title,
                fontsize=16,
                fontweight='bold',
                color=text_color,
                pad=20
            )
        
        plt.tight_layout()
        return fig
    
    def _draw_gradient_ring(self, ax, colors):
        """Draw a gradient ring for 3D effect."""
        for i in range(20):
            radius = 1.0 + (i * 0.02)
            alpha = 0.5 * (1 - i / 20)
            circle = Circle(
                (0, 0), radius,
                fill=False,
                edgecolor=colors[0],
                linewidth=2,
                alpha=alpha,
                zorder=0
            )
            ax.add_patch(circle)
    
    def _draw_head_3d(self, ax, color, style):
        """Draw head with 3D-like shading."""
        # Main head circle
        head = Circle(
            (0, 0), self.head_radius,
            fill=False,
            edgecolor=color,
            linewidth=3,
            zorder=10
        )
        ax.add_patch(head)
        
        # Inner shadow for depth
        for i in range(3):
            offset = 0.02 * (i + 1)
            shadow = Circle(
                (0, 0), self.head_radius - offset,
                fill=False,
                edgecolor=color,
                linewidth=0.5,
                alpha=0.2,
                zorder=9
            )
            ax.add_patch(shadow)
        
        # Nose - more detailed
        nose_verts = [
            (0, 1.0),       # Base left
            (-0.06, 1.05),  # Left side
            (-0.04, 1.12),  # Left tip
            (0, 1.18),      # Tip
            (0.04, 1.12),   # Right tip
            (0.06, 1.05),   # Right side
            (0, 1.0),       # Base right
        ]
        nose_codes = [Path.MOVETO] + [Path.CURVE3] * 5 + [Path.CLOSEPOLY]
        nose_path = Path(nose_verts, nose_codes)
        nose_patch = PathPatch(
            nose_path,
            facecolor='none',
            edgecolor=color,
            linewidth=2.5,
            zorder=10
        )
        ax.add_patch(nose_patch)
        
        # Ears - more realistic curved shape
        # Left ear
        ear_l_verts = [
            (-1.0, 0.12),
            (-1.12, 0.10),
            (-1.15, 0.0),
            (-1.12, -0.10),
            (-1.0, -0.12),
        ]
        ear_l_codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ear_l_path = Path(ear_l_verts, ear_l_codes)
        ear_l = PathPatch(
            ear_l_path,
            facecolor='none',
            edgecolor=color,
            linewidth=2.5,
            zorder=10
        )
        ax.add_patch(ear_l)
        
        # Right ear
        ear_r_verts = [
            (1.0, 0.12),
            (1.12, 0.10),
            (1.15, 0.0),
            (1.12, -0.10),
            (1.0, -0.12),
        ]
        ear_r_path = Path(ear_r_verts, ear_l_codes)
        ear_r = PathPatch(
            ear_r_path,
            facecolor='none',
            edgecolor=color,
            linewidth=2.5,
            zorder=10
        )
        ax.add_patch(ear_r)
        
        # Central sulcus hint (subtle line)
        if style == "modern":
            ax.plot([0, 0], [0.3, -0.3], color='white', alpha=0.1, linewidth=1, zorder=3)
            ax.plot([-0.3, 0.3], [0, 0], color='white', alpha=0.1, linewidth=1, zorder=3)
    
    def _draw_electrodes(self, ax, color, show_names, style):
        """Draw electrode positions with glow effect."""
        for i, (name, pos) in enumerate(zip(self.electrode_names, self.positions)):
            # Glow effect
            for r in [12, 8, 5]:
                alpha = 0.1 if r == 12 else (0.2 if r == 8 else 0.8)
                ax.scatter(
                    pos[0], pos[1],
                    s=r * 8,
                    c=color,
                    alpha=alpha,
                    zorder=11
                )
            
            # Main dot
            ax.scatter(
                pos[0], pos[1],
                s=30,
                c=color,
                edgecolors='white' if style == "modern" else 'black',
                linewidths=0.5,
                zorder=12
            )
            
            # Labels
            if show_names:
                txt = ax.text(
                    pos[0], pos[1] + 0.1,
                    name,
                    fontsize=7,
                    ha='center',
                    va='bottom',
                    color='white' if style == "modern" else '#333333',
                    fontweight='bold',
                    zorder=13
                )
                if style == "modern":
                    txt.set_path_effects([
                        path_effects.withStroke(linewidth=2, foreground='#1a1a2e')
                    ])
    
    def _draw_regions(self, ax, text_color):
        """Draw brain region labels."""
        regions = [
            ("Frontal", (0, 0.65)),
            ("Parietal", (0, -0.25)),
            ("Occipital", (0, -0.75)),
            ("Temporal L", (-0.65, 0)),
            ("Temporal R", (0.65, 0)),
        ]
        
        for name, pos in regions:
            ax.text(
                pos[0], pos[1],
                name,
                fontsize=9,
                ha='center',
                va='center',
                color=text_color,
                alpha=0.5,
                style='italic',
                zorder=5
            )
    
    def _draw_colorbar(self, fig, ax, im, text_color, style):
        """Draw a custom colorbar."""
        # Create colorbar axes
        cax = fig.add_axes([0.88, 0.25, 0.03, 0.5])
        cbar = fig.colorbar(im, cax=cax)
        
        cbar.ax.tick_params(
            labelsize=9,
            colors=text_color,
            length=0
        )
        cbar.outline.set_edgecolor(text_color)
        cbar.outline.set_linewidth(0.5)
        
        # Add labels
        cbar.ax.set_ylabel(
            'µV',
            fontsize=10,
            color=text_color,
            rotation=0,
            labelpad=15
        )
    
    def to_image_bytes(self, values: np.ndarray, format: str = 'png', **kwargs) -> bytes:
        """Return topomap as image bytes."""
        fig = self.create_figure(values, **kwargs)
        buf = BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    
    def create_band_comparison(
        self,
        band_data: Dict[str, np.ndarray],
        figsize: Tuple[float, float] = (16, 4),
        style: str = "modern",
    ) -> plt.Figure:
        """
        Create side-by-side topomaps for different frequency bands.
        
        Args:
            band_data: Dict mapping band name to channel values.
            figsize: Figure size.
            style: Visual style.
        """
        n_bands = len(band_data)
        
        if style == "modern":
            fig, axes = plt.subplots(1, n_bands, figsize=figsize, facecolor='#1a1a2e')
            text_color = 'white'
        else:
            fig, axes = plt.subplots(1, n_bands, figsize=figsize, facecolor='white')
            text_color = 'black'
        
        if n_bands == 1:
            axes = [axes]
        
        band_colors = {
            'Delta': '#9c27b0',  # Purple
            'Theta': '#2196f3',  # Blue
            'Alpha': '#4caf50',  # Green
            'Beta': '#ff9800',   # Orange
            'Gamma': '#f44336',  # Red
        }
        
        for ax, (band_name, values) in zip(axes, band_data.items()):
            ax.set_facecolor(fig.get_facecolor())
            ax.set_aspect('equal')
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.axis('off')
            
            # Interpolate
            grid_values = self.interpolate_rbf(values)
            
            # Use band-specific colormap
            cmap = plt.cm.get_cmap('viridis')
            
            # Draw
            im = ax.imshow(
                grid_values,
                extent=[-1.2, 1.2, -1.2, 1.2],
                origin='lower',
                cmap=cmap,
                interpolation='bilinear'
            )
            
            # Head outline
            head = Circle((0, 0), 1.0, fill=False, edgecolor=text_color, linewidth=2)
            ax.add_patch(head)
            
            # Nose
            ax.plot([0, -0.05, 0.05, 0], [1.0, 1.1, 1.1, 1.0], color=text_color, linewidth=2)
            
            # Title with band color
            ax.set_title(
                band_name,
                fontsize=14,
                fontweight='bold',
                color=band_colors.get(band_name, text_color),
                pad=10
            )
        
        plt.tight_layout()
        return fig


class RealtimeTopoWidget:
    """
    Widget for real-time topomap updates in Qt applications.
    
    Optimized for fast updates with minimal redraw.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        update_interval: float = 0.1,
    ):
        """
        Initialize realtime widget.
        
        Args:
            channel_names: Channel names.
            update_interval: Minimum time between updates.
        """
        self.topo = EnhancedTopoMap(channel_names, resolution=100)
        self.update_interval = update_interval
        self._last_update = 0
        self._cached_image = None
    
    def update(self, values: np.ndarray, style: str = "modern") -> bytes:
        """
        Update and return image bytes.
        
        Returns cached image if called too frequently.
        """
        import time
        current_time = time.time()
        
        if current_time - self._last_update < self.update_interval and self._cached_image:
            return self._cached_image
        
        self._cached_image = self.topo.to_image_bytes(
            values,
            style=style,
            show_names=False,
            show_colorbar=False,
            figsize=(4, 4),
            dpi=80
        )
        self._last_update = current_time
        
        return self._cached_image
