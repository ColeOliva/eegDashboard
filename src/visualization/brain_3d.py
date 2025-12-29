"""
3D Brain Visualization Module.

Creates interactive 3D brain/head models showing EEG activity
mapped onto a realistic head surface.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from io import BytesIO


# 3D electrode positions on unit sphere (x, y, z)
# Nose points in +Y direction, top of head is +Z
ELECTRODE_POSITIONS_3D = {
    # Frontal polar
    "Fp1": (-0.22, 0.95, 0.22),
    "Fp2": (0.22, 0.95, 0.22),
    
    # Frontal
    "F7": (-0.70, 0.70, 0.14),
    "F3": (-0.40, 0.75, 0.53),
    "Fz": (0.0, 0.72, 0.69),
    "F4": (0.40, 0.75, 0.53),
    "F8": (0.70, 0.70, 0.14),
    
    # Temporal
    "T3": (-0.99, 0.0, 0.14),
    "T4": (0.99, 0.0, 0.14),
    "T5": (-0.70, -0.70, 0.14),
    "T6": (0.70, -0.70, 0.14),
    
    # Central
    "C3": (-0.55, 0.0, 0.83),
    "Cz": (0.0, 0.0, 1.0),
    "C4": (0.55, 0.0, 0.83),
    
    # Parietal
    "P3": (-0.40, -0.55, 0.73),
    "Pz": (0.0, -0.55, 0.83),
    "P4": (0.40, -0.55, 0.73),
    
    # Occipital
    "O1": (-0.22, -0.95, 0.22),
    "Oz": (0.0, -0.92, 0.38),
    "O2": (0.22, -0.95, 0.22),
    
    # Reference (ears)
    "A1": (-1.0, 0.0, -0.15),
    "A2": (1.0, 0.0, -0.15),
}

STANDARD_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "Oz", "O2"
]


def create_brain_colormap():
    """Create colormap for brain activity visualization."""
    colors = [
        (0.0, '#0d47a1'),
        (0.25, '#42a5f5'),
        (0.45, '#e3f2fd'),
        (0.5, '#ffffff'),
        (0.55, '#ffebee'),
        (0.75, '#ef5350'),
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
    
    return LinearSegmentedColormap('brain_3d', cmap_data)


def create_head_mesh(resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a realistic head-shaped mesh.
    
    Returns spherical coordinates modified to look more head-like.
    """
    # Create base sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    # Base sphere
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Modify to be more head-shaped
    # Slightly elongate front-to-back
    y = y * 1.05
    
    # Flatten the bottom (chin area)
    mask = z < -0.3
    z[mask] = z[mask] * 0.7 - 0.1
    
    # Slight bulge at back (occipital)
    back_mask = (y < -0.3) & (z > 0)
    y[back_mask] = y[back_mask] * 1.08
    
    return x, y, z


def create_brain_surface(resolution: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a brain-like surface (upper hemisphere with folds suggestion).
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi / 2, resolution // 2)  # Only top hemisphere
    
    x = np.outer(np.cos(u), np.sin(v)) * 0.85
    y = np.outer(np.sin(u), np.sin(v)) * 0.90
    z = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.85 + 0.1
    
    # Add subtle wrinkles (sulci)
    wrinkle = 0.02 * np.sin(8 * np.outer(u, np.ones_like(v))) * np.sin(6 * np.outer(np.ones_like(u), v))
    x = x + wrinkle * x
    y = y + wrinkle * y
    
    return x, y, z


class Brain3D:
    """
    Creates 3D brain/head visualizations with EEG activity mapping.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        resolution: int = 50,
    ):
        """
        Initialize 3D brain visualization.
        
        Args:
            channel_names: List of channel names in data order.
            resolution: Mesh resolution (higher = smoother).
        """
        self.channel_names = channel_names or STANDARD_CHANNEL_ORDER
        self.resolution = resolution
        
        # Get electrode positions
        self.positions_3d = []
        self.valid_channels = []
        self.electrode_names = []
        
        for i, name in enumerate(self.channel_names):
            if name in ELECTRODE_POSITIONS_3D:
                self.positions_3d.append(ELECTRODE_POSITIONS_3D[name])
                self.valid_channels.append(i)
                self.electrode_names.append(name)
        
        self.positions_3d = np.array(self.positions_3d)
        
        # Create meshes
        self.head_x, self.head_y, self.head_z = create_head_mesh(resolution)
        self.brain_x, self.brain_y, self.brain_z = create_brain_surface(resolution)
        
        self._colormap = create_brain_colormap()
    
    def interpolate_on_surface(
        self,
        values: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate electrode values onto a 3D surface.
        """
        valid_values = values[self.valid_channels]
        
        # Use RBF interpolation in 3D
        rbf = Rbf(
            self.positions_3d[:, 0],
            self.positions_3d[:, 1],
            self.positions_3d[:, 2],
            valid_values,
            function='thin_plate',
            smooth=0.1
        )
        
        # Interpolate on surface
        surface_values = rbf(x, y, z)
        
        return surface_values
    
    def create_figure(
        self,
        values: np.ndarray,
        title: str = "3D Brain Activity",
        view: str = "top",
        show_electrodes: bool = True,
        show_names: bool = True,
        show_head: bool = True,
        show_brain: bool = True,
        transparency: float = 0.7,
        figsize: Tuple[float, float] = (10, 10),
        style: str = "dark",
    ) -> plt.Figure:
        """
        Create a 3D brain visualization.
        
        Args:
            values: Array of values for each channel.
            title: Plot title.
            view: View angle ('top', 'front', 'left', 'right', 'back', 'iso').
            show_electrodes: Show electrode positions.
            show_names: Show electrode labels.
            show_head: Show transparent head surface.
            show_brain: Show colored brain surface.
            transparency: Head surface transparency (0-1).
            figsize: Figure size.
            style: 'dark' or 'light' background.
        """
        # Set up figure
        if style == "dark":
            fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
            ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
            text_color = 'white'
            head_color = (0.9, 0.85, 0.8, 0.15)
            electrode_color = '#00ff88'
        else:
            fig = plt.figure(figsize=figsize, facecolor='white')
            ax = fig.add_subplot(111, projection='3d', facecolor='white')
            text_color = 'black'
            head_color = (0.95, 0.9, 0.85, 0.2)
            electrode_color = '#ff4444'
        
        # Interpolate values on brain surface
        brain_values = self.interpolate_on_surface(
            values, self.brain_x, self.brain_y, self.brain_z
        )
        
        # Normalize for colormap
        vmin = np.nanpercentile(brain_values, 5)
        vmax = np.nanpercentile(brain_values, 95)
        abs_max = max(abs(vmin), abs(vmax))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        colors = self._colormap(norm(brain_values))
        
        # Draw brain surface with activity colors
        if show_brain:
            ax.plot_surface(
                self.brain_x, self.brain_y, self.brain_z,
                facecolors=colors,
                shade=True,
                alpha=0.9,
                linewidth=0,
                antialiased=True,
                zorder=2
            )
        
        # Draw transparent head
        if show_head:
            ax.plot_surface(
                self.head_x, self.head_y, self.head_z,
                color=head_color,
                shade=True,
                linewidth=0,
                antialiased=True,
                zorder=1
            )
            
            # Draw head outline wireframe
            ax.plot_wireframe(
                self.head_x, self.head_y, self.head_z,
                color=(0.5, 0.5, 0.5, 0.1),
                linewidth=0.3,
                rstride=5,
                cstride=5,
                zorder=1
            )
        
        # Draw electrodes
        if show_electrodes:
            for i, (name, pos) in enumerate(zip(self.electrode_names, self.positions_3d)):
                # Get value for this electrode
                val = values[self.valid_channels[i]]
                point_color = self._colormap(norm(val))
                
                # Electrode marker
                ax.scatter(
                    [pos[0]], [pos[1]], [pos[2]],
                    s=100,
                    c=[point_color],
                    edgecolors='white',
                    linewidths=1,
                    zorder=10,
                    depthshade=False
                )
                
                # Electrode label
                if show_names:
                    ax.text(
                        pos[0] * 1.15, pos[1] * 1.15, pos[2] * 1.15,
                        name,
                        fontsize=8,
                        color=text_color,
                        ha='center',
                        va='center',
                        zorder=11
                    )
        
        # Draw nose indicator
        nose_y = [1.0, 1.2, 1.0]
        nose_x = [0.05, 0, -0.05]
        nose_z = [0, 0.05, 0]
        ax.plot(nose_x, nose_y, nose_z, color=text_color, linewidth=2, zorder=5)
        
        # Set view angle
        views = {
            'top': (90, 0),
            'front': (0, 90),
            'back': (0, -90),
            'left': (0, 180),
            'right': (0, 0),
            'iso': (30, 45),
            'iso_front': (30, 60),
        }
        elev, azim = views.get(view, (30, 45))
        ax.view_init(elev=elev, azim=azim)
        
        # Clean up axes
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-0.8, 1.3)
        ax.set_box_aspect([1, 1, 0.8])
        
        # Hide axes
        ax.set_axis_off()
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self._colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
        cbar.ax.tick_params(labelsize=9, colors=text_color)
        cbar.outline.set_edgecolor(text_color)
        cbar.ax.set_ylabel('µV', fontsize=10, color=text_color, rotation=0, labelpad=15)
        
        # Title
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', color=text_color, pad=20)
        
        plt.tight_layout()
        return fig
    
    def create_rotating_frames(
        self,
        values: np.ndarray,
        n_frames: int = 36,
        **kwargs
    ) -> List[bytes]:
        """
        Create frames for a rotating animation.
        
        Args:
            values: Channel values.
            n_frames: Number of frames for full rotation.
            **kwargs: Additional arguments for create_figure.
            
        Returns:
            List of PNG image bytes for each frame.
        """
        frames = []
        
        for i in range(n_frames):
            azim = (i / n_frames) * 360
            fig = plt.figure(figsize=(8, 8), facecolor='#1a1a2e')
            ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
            
            # Similar to create_figure but with custom azimuth
            brain_values = self.interpolate_on_surface(
                values, self.brain_x, self.brain_y, self.brain_z
            )
            
            vmin, vmax = np.nanpercentile(brain_values, [5, 95])
            abs_max = max(abs(vmin), abs(vmax))
            norm = Normalize(vmin=-abs_max, vmax=abs_max)
            colors = self._colormap(norm(brain_values))
            
            ax.plot_surface(
                self.brain_x, self.brain_y, self.brain_z,
                facecolors=colors,
                shade=True,
                alpha=0.9,
                linewidth=0,
                antialiased=True
            )
            
            ax.view_init(elev=25, azim=azim)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_zlim(-0.8, 1.3)
            ax.set_axis_off()
            
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            frames.append(buf.getvalue())
        
        return frames
    
    def to_image_bytes(
        self,
        values: np.ndarray,
        format: str = 'png',
        dpi: int = 100,
        **kwargs
    ) -> bytes:
        """
        Create 3D brain visualization and return as image bytes.
        """
        fig = self.create_figure(values, **kwargs)
        buf = BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    
    def create_multiview(
        self,
        values: np.ndarray,
        figsize: Tuple[float, float] = (16, 8),
        style: str = "dark",
    ) -> plt.Figure:
        """
        Create a multi-view panel showing brain from different angles.
        """
        views = ['top', 'front', 'left', 'right']
        
        if style == "dark":
            fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
        else:
            fig = plt.figure(figsize=figsize, facecolor='white')
        
        brain_values = self.interpolate_on_surface(
            values, self.brain_x, self.brain_y, self.brain_z
        )
        vmin, vmax = np.nanpercentile(brain_values, [5, 95])
        abs_max = max(abs(vmin), abs(vmax))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        colors = self._colormap(norm(brain_values))
        
        view_angles = {
            'top': (90, 0),
            'front': (0, 90),
            'left': (0, 180),
            'right': (0, 0),
        }
        
        for idx, view in enumerate(views):
            ax = fig.add_subplot(1, 4, idx + 1, projection='3d', facecolor=fig.get_facecolor())
            
            ax.plot_surface(
                self.brain_x, self.brain_y, self.brain_z,
                facecolors=colors,
                shade=True,
                alpha=0.9,
                linewidth=0,
                antialiased=True
            )
            
            elev, azim = view_angles[view]
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_zlim(-0.6, 1.2)
            ax.set_axis_off()
            
            text_color = 'white' if style == 'dark' else 'black'
            ax.set_title(view.capitalize(), fontsize=12, color=text_color, pad=5)
        
        plt.tight_layout()
        return fig


class AnimatedBrain3D:
    """
    Manages animated 3D brain visualization for real-time display.
    """
    
    def __init__(self, channel_names: Optional[List[str]] = None):
        self.brain = Brain3D(channel_names, resolution=30)  # Lower res for speed
        self.current_view = 'iso'
        self.rotation_angle = 0
        self.auto_rotate = False
        
    def get_frame(
        self,
        values: np.ndarray,
        rotate: bool = False,
    ) -> bytes:
        """
        Get a single frame of the 3D brain.
        
        Args:
            values: Channel values.
            rotate: If True, increment rotation angle.
        """
        if rotate:
            self.rotation_angle = (self.rotation_angle + 5) % 360
        
        # Create figure with current rotation
        fig = plt.figure(figsize=(6, 6), facecolor='#1a1a2e')
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
        
        brain_values = self.brain.interpolate_on_surface(
            values,
            self.brain.brain_x,
            self.brain.brain_y,
            self.brain.brain_z
        )
        
        vmin, vmax = np.nanpercentile(brain_values, [5, 95])
        abs_max = max(abs(vmin), abs(vmax)) if abs(vmin) + abs(vmax) > 0 else 1
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        colors = self.brain._colormap(norm(brain_values))
        
        ax.plot_surface(
            self.brain.brain_x,
            self.brain.brain_y,
            self.brain.brain_z,
            facecolors=colors,
            shade=True,
            alpha=0.9,
            linewidth=0,
            antialiased=True
        )
        
        ax.view_init(elev=25, azim=self.rotation_angle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-0.6, 1.2)
        ax.set_axis_off()
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
