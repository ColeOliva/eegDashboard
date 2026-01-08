"""
Fast Brain Visualization Renderer.

Optimized rendering with:
- Pre-computed interpolation grids
- Cached colormaps and textures
- Frame rate limiting
- Lazy figure creation
"""

import time
from functools import lru_cache
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle

# Standard electrode positions
ELECTRODE_POSITIONS = {
    "Fp1": (-0.22, 0.83), "Fp2": (0.22, 0.83),
    "F7": (-0.59, 0.59), "F3": (-0.32, 0.50), "Fz": (0.0, 0.50),
    "F4": (0.32, 0.50), "F8": (0.59, 0.59),
    "T3": (-0.81, 0.0), "T4": (0.81, 0.0),
    "T5": (-0.59, -0.59), "T6": (0.59, -0.59),
    "C3": (-0.38, 0.0), "Cz": (0.0, 0.0), "C4": (0.38, 0.0),
    "P3": (-0.32, -0.50), "Pz": (0.0, -0.50), "P4": (0.32, -0.50),
    "O1": (-0.22, -0.83), "Oz": (0.0, -0.83), "O2": (0.22, -0.83),
}

STANDARD_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "Oz", "O2"
]


@lru_cache(maxsize=4)
def _create_colormap(style: str) -> LinearSegmentedColormap:
    """Cached colormap creation."""
    if style == "pet":
        colors = [
            (0.0, '#000033'), (0.15, '#0000aa'), (0.3, '#00aa00'),
            (0.45, '#aaaa00'), (0.6, '#ff8800'), (0.75, '#ff0000'),
            (0.9, '#ff00ff'), (1.0, '#ffffff'),
        ]
    elif style == "hot":
        colors = [
            (0.0, '#000000'), (0.2, '#220000'), (0.35, '#880000'),
            (0.5, '#ff0000'), (0.65, '#ff8800'), (0.8, '#ffff00'),
            (0.9, '#ffffaa'), (1.0, '#ffffff'),
        ]
    else:  # brain activity (RdBu style)
        colors = [
            (0.0, '#0d47a1'), (0.25, '#42a5f5'), (0.45, '#e3f2fd'),
            (0.5, '#ffffff'), (0.55, '#ffebee'), (0.75, '#ef5350'),
            (1.0, '#b71c1c'),
        ]
    
    cmap_data = {'red': [], 'green': [], 'blue': []}
    for pos, color in colors:
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        cmap_data['red'].append((pos, r, r))
        cmap_data['green'].append((pos, g, g))
        cmap_data['blue'].append((pos, b, b))
    
    return LinearSegmentedColormap(f'{style}_cmap', cmap_data)


class FastTopoMapRenderer:
    """
    High-performance topographic map renderer.
    
    Optimizations:
    - Pre-computed interpolation grid
    - Cached figure and axes
    - Lazy redraws with frame limiting
    - Pre-computed electrode coordinates
    """
    
    __slots__ = ['resolution', 'channel_names', 'positions', 'valid_channels',
                 '_grid_x', '_grid_y', '_head_mask', '_soft_mask',
                 '_last_render_time', '_min_frame_interval', '_last_values_hash',
                 '_cached_bytes', '_interpolator_cache']
    
    def __init__(
        self,
        resolution: int = 150,
        channel_names: Optional[list] = None,
        min_fps: float = 30.0,
    ):
        self.resolution = resolution
        self.channel_names = channel_names or STANDARD_CHANNEL_ORDER
        
        # Get electrode positions
        self.positions = []
        self.valid_channels = []
        for i, name in enumerate(self.channel_names):
            if name in ELECTRODE_POSITIONS:
                self.positions.append(ELECTRODE_POSITIONS[name])
                self.valid_channels.append(i)
        self.positions = np.array(self.positions)
        
        # Pre-compute interpolation grid
        self._create_grid()
        
        # Frame rate limiting
        self._min_frame_interval = 1.0 / min_fps
        self._last_render_time = 0.0
        self._last_values_hash = 0
        self._cached_bytes: Optional[bytes] = None
        self._interpolator_cache: Dict[int, np.ndarray] = {}
        
    def _create_grid(self):
        """Create pre-computed interpolation grid."""
        x = np.linspace(-1.1, 1.1, self.resolution)
        y = np.linspace(-1.1, 1.1, self.resolution)
        self._grid_x, self._grid_y = np.meshgrid(x, y)
        
        # Masks
        dist = np.sqrt(self._grid_x**2 + self._grid_y**2)
        self._head_mask = dist <= 1.0
        edge_width = 0.1
        self._soft_mask = np.clip((1.0 - dist) / edge_width, 0, 1)
        
    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """
        Fast RBF interpolation with caching.
        """
        # Use simple hash for cache key
        values_hash = hash(values.tobytes())
        
        if values_hash in self._interpolator_cache:
            return self._interpolator_cache[values_hash]
        
        valid_values = values[self.valid_channels]
        
        # Use RBFInterpolator (newer, faster API)
        interp = RBFInterpolator(
            self.positions,
            valid_values,
            kernel='thin_plate_spline',
            smoothing=0.1
        )
        
        # Evaluate on grid
        grid_points = np.column_stack([self._grid_x.ravel(), self._grid_y.ravel()])
        grid_values = interp(grid_points).reshape(self.resolution, self.resolution)
        
        # Apply smoothing and mask
        grid_values = gaussian_filter(grid_values, sigma=0.8)
        grid_values = grid_values * self._soft_mask
        grid_values[~self._head_mask] = np.nan
        
        # Cache (limited size)
        if len(self._interpolator_cache) > 10:
            # Remove oldest entry
            self._interpolator_cache.pop(next(iter(self._interpolator_cache)))
        self._interpolator_cache[values_hash] = grid_values
        
        return grid_values
    
    def render(
        self,
        values: np.ndarray,
        style: str = "brain",
        show_electrodes: bool = True,
        show_names: bool = False,
        show_colorbar: bool = True,
        figsize: Tuple[float, float] = (5, 5),
        dpi: int = 80,
    ) -> bytes:
        """
        Render topomap to image bytes with frame rate limiting.
        """
        # Frame rate limiting
        current_time = time.time()
        values_hash = hash(values.tobytes())
        
        if (current_time - self._last_render_time < self._min_frame_interval and 
            values_hash == self._last_values_hash and 
            self._cached_bytes is not None):
            return self._cached_bytes
        
        # Interpolate
        grid_values = self.interpolate(values)
        
        # Create figure
        cmap = _create_colormap(style)
        
        # Auto-scale symmetrically
        vmax = np.nanpercentile(np.abs(grid_values), 95)
        vmin = -vmax
        
        # Fast figure creation
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        ax.set_aspect('equal')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis('off')
        
        # Draw activity
        im = ax.imshow(
            grid_values,
            extent=[-1.1, 1.1, -1.1, 1.1],
            origin='lower',
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation='bilinear',
        )
        
        # Head outline
        head = Circle((0, 0), 1.0, fill=False, edgecolor='white', linewidth=2)
        ax.add_patch(head)
        
        # Nose
        ax.plot([0, -0.06, 0, 0.06, 0], [1.0, 1.05, 1.15, 1.05, 1.0], 'w-', linewidth=2)
        
        # Electrodes
        if show_electrodes:
            ax.scatter(
                self.positions[:, 0], self.positions[:, 1],
                c='#00ff88', s=25, edgecolors='white', linewidths=0.5, zorder=10
            )
        
        # Names
        if show_names:
            for i, (name, pos) in enumerate(zip(
                [self.channel_names[j] for j in self.valid_channels], 
                self.positions
            )):
                ax.text(pos[0], pos[1] + 0.08, name, fontsize=6, 
                       color='white', ha='center', va='bottom')
        
        # Colorbar
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            cbar.ax.tick_params(labelsize=8, colors='white')
            cbar.outline.set_edgecolor('white')
        
        plt.tight_layout()
        
        # Save to bytes
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        
        result = buf.getvalue()
        
        # Update cache
        self._last_render_time = current_time
        self._last_values_hash = values_hash
        self._cached_bytes = result
        
        return result


class FastBrain3DRenderer:
    """
    Optimized 3D brain renderer with view caching.
    """
    
    # 3D electrode positions
    POSITIONS_3D = {
        "Fp1": (-0.22, 0.95, 0.22), "Fp2": (0.22, 0.95, 0.22),
        "F7": (-0.70, 0.70, 0.14), "F3": (-0.40, 0.75, 0.53),
        "Fz": (0.0, 0.72, 0.69), "F4": (0.40, 0.75, 0.53),
        "F8": (0.70, 0.70, 0.14),
        "T3": (-0.99, 0.0, 0.14), "T4": (0.99, 0.0, 0.14),
        "T5": (-0.70, -0.70, 0.14), "T6": (0.70, -0.70, 0.14),
        "C3": (-0.55, 0.0, 0.83), "Cz": (0.0, 0.0, 1.0),
        "C4": (0.55, 0.0, 0.83),
        "P3": (-0.40, -0.55, 0.73), "Pz": (0.0, -0.55, 0.83),
        "P4": (0.40, -0.55, 0.73),
        "O1": (-0.22, -0.95, 0.22), "Oz": (0.0, -0.92, 0.38),
        "O2": (0.22, -0.95, 0.22),
    }
    
    VIEWS = {
        'top': (90, 0), 'front': (0, 90), 'back': (0, -90),
        'left': (0, 180), 'right': (0, 0), 'iso': (30, 45),
    }
    
    def __init__(self, channel_names: Optional[list] = None, resolution: int = 40):
        self.channel_names = channel_names or STANDARD_CHANNEL_ORDER
        self.resolution = resolution
        
        # Get 3D positions
        self.positions_3d = []
        self.valid_channels = []
        for i, name in enumerate(self.channel_names):
            if name in self.POSITIONS_3D:
                self.positions_3d.append(self.POSITIONS_3D[name])
                self.valid_channels.append(i)
        self.positions_3d = np.array(self.positions_3d)
        
        # Pre-compute head mesh
        self._head_mesh = self._create_head_mesh()
        self._brain_mesh = self._create_brain_mesh()
        
        # Cache
        self._cache: Dict[str, bytes] = {}
        self._rotation_angle = 0.0
        
    def _create_head_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create head mesh."""
        u = np.linspace(0, 2 * np.pi, self.resolution)
        v = np.linspace(0, np.pi, self.resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v)) * 1.05
        z = np.outer(np.ones_like(u), np.cos(v))
        return x, y, z
    
    def _create_brain_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create brain surface mesh."""
        u = np.linspace(0, 2 * np.pi, self.resolution)
        v = np.linspace(0, np.pi / 2, self.resolution // 2)
        x = np.outer(np.cos(u), np.sin(v)) * 0.85
        y = np.outer(np.sin(u), np.sin(v)) * 0.90
        z = np.outer(np.ones_like(u), np.cos(v)) * 0.85 + 0.1
        return x, y, z
    
    def render(
        self,
        values: np.ndarray,
        view: str = "iso",
        figsize: Tuple[float, float] = (6, 6),
        dpi: int = 80,
    ) -> bytes:
        """Render 3D brain view."""
        valid_values = values[self.valid_channels]
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='#1a1a2e')
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
        
        # Get colormap
        cmap = _create_colormap("brain")
        vmax = np.max(np.abs(valid_values)) or 1.0
        norm = Normalize(vmin=-vmax, vmax=vmax)
        
        # Draw brain surface (simplified for speed)
        bx, by, bz = self._brain_mesh
        ax.plot_surface(bx, by, bz, color=(0.8, 0.75, 0.7, 0.3), 
                       shade=True, linewidth=0, antialiased=True)
        
        # Draw electrodes with values
        colors = cmap(norm(valid_values))
        ax.scatter(
            self.positions_3d[:, 0],
            self.positions_3d[:, 1],
            self.positions_3d[:, 2],
            s=100, c=colors, edgecolors='white', linewidths=1,
            depthshade=False, zorder=10
        )
        
        # Set view
        elev, azim = self.VIEWS.get(view, (30, 45))
        ax.view_init(elev=elev, azim=azim)
        
        # Clean up
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-0.5, 1.2)
        ax.set_box_aspect([1, 1, 0.7])
        ax.axis('off')
        
        # Save
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
    
    def render_rotating(self, values: np.ndarray, speed: float = 5.0) -> bytes:
        """Render with auto-rotation."""
        self._rotation_angle = (self._rotation_angle + speed) % 360
        
        fig = plt.figure(figsize=(6, 6), dpi=80, facecolor='#1a1a2e')
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
        
        valid_values = values[self.valid_channels]
        cmap = _create_colormap("brain")
        vmax = np.max(np.abs(valid_values)) or 1.0
        norm = Normalize(vmin=-vmax, vmax=vmax)
        
        bx, by, bz = self._brain_mesh
        ax.plot_surface(bx, by, bz, color=(0.8, 0.75, 0.7, 0.3),
                       shade=True, linewidth=0)
        
        colors = cmap(norm(valid_values))
        ax.scatter(
            self.positions_3d[:, 0], self.positions_3d[:, 1], self.positions_3d[:, 2],
            s=100, c=colors, edgecolors='white', linewidths=1
        )
        
        ax.view_init(elev=25, azim=self._rotation_angle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-0.5, 1.2)
        ax.axis('off')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
