"""
Anatomical Brain Visualization.

Creates realistic brain visualizations with actual anatomical structure,
similar to PET/fMRI scans, with EEG activity overlaid as a heatmap.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import gaussian_filter, zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Ellipse, Polygon, FancyBboxPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from io import BytesIO


# Standard 10-20 electrode positions (normalized to brain image coordinates)
ELECTRODE_POSITIONS = {
    "Fp1": (-0.22, 0.83), "Fp2": (0.22, 0.83),
    "F7": (-0.59, 0.59), "F3": (-0.32, 0.50), "Fz": (0.0, 0.50),
    "F4": (0.32, 0.50), "F8": (0.59, 0.59),
    "T3": (-0.81, 0.0), "T4": (0.81, 0.0),
    "T5": (-0.59, -0.59), "T6": (0.59, -0.59),
    "C3": (-0.38, 0.0), "Cz": (0.0, 0.0), "C4": (0.38, 0.0),
    "P3": (-0.32, -0.50), "Pz": (0.0, -0.50), "P4": (0.32, -0.50),
    "O1": (-0.22, -0.83), "Oz": (0.0, -0.83), "O2": (0.22, -0.83),
    "A1": (-0.95, 0.0), "A2": (0.95, 0.0),
}

STANDARD_CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "Oz", "O2"
]

# Brain region definitions with anatomical names
BRAIN_REGIONS = {
    "Prefrontal Cortex": {
        "center": (0, 0.7),
        "size": (0.5, 0.25),
        "electrodes": ["Fp1", "Fp2"],
        "function": "Executive function, decision making"
    },
    "Frontal Lobe": {
        "center": (0, 0.45),
        "size": (0.8, 0.3),
        "electrodes": ["F3", "Fz", "F4", "F7", "F8"],
        "function": "Motor control, planning"
    },
    "Motor Cortex": {
        "center": (0, 0.15),
        "size": (0.7, 0.2),
        "electrodes": ["C3", "Cz", "C4"],
        "function": "Movement control"
    },
    "Temporal Lobe (L)": {
        "center": (-0.7, 0),
        "size": (0.25, 0.5),
        "electrodes": ["T3", "T5"],
        "function": "Auditory processing, memory"
    },
    "Temporal Lobe (R)": {
        "center": (0.7, 0),
        "size": (0.25, 0.5),
        "electrodes": ["T4", "T6"],
        "function": "Auditory processing, memory"
    },
    "Parietal Lobe": {
        "center": (0, -0.35),
        "size": (0.7, 0.3),
        "electrodes": ["P3", "Pz", "P4"],
        "function": "Sensory integration, spatial awareness"
    },
    "Occipital Lobe": {
        "center": (0, -0.75),
        "size": (0.5, 0.25),
        "electrodes": ["O1", "Oz", "O2"],
        "function": "Visual processing"
    },
}


def create_pet_colormap():
    """Create a PET/fMRI-style colormap (rainbow thermal)."""
    colors = [
        (0.0, '#000033'),   # Deep blue/black (lowest)
        (0.15, '#0000aa'),  # Blue
        (0.3, '#00aa00'),   # Green
        (0.45, '#aaaa00'),  # Yellow
        (0.6, '#ff8800'),   # Orange
        (0.75, '#ff0000'),  # Red
        (0.9, '#ff00ff'),   # Magenta
        (1.0, '#ffffff'),   # White (highest)
    ]
    
    cmap_data = {'red': [], 'green': [], 'blue': []}
    for pos, color in colors:
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        cmap_data['red'].append((pos, r, r))
        cmap_data['green'].append((pos, g, g))
        cmap_data['blue'].append((pos, b, b))
    
    return LinearSegmentedColormap('pet_scan', cmap_data)


def create_hot_metal_colormap():
    """Create a hot metal colormap like thermal imaging."""
    colors = [
        (0.0, '#000000'),   # Black
        (0.2, '#220000'),   # Very dark red
        (0.35, '#880000'),  # Dark red
        (0.5, '#ff0000'),   # Red
        (0.65, '#ff8800'),  # Orange
        (0.8, '#ffff00'),   # Yellow
        (0.9, '#ffffaa'),   # Light yellow
        (1.0, '#ffffff'),   # White
    ]
    
    cmap_data = {'red': [], 'green': [], 'blue': []}
    for pos, color in colors:
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        cmap_data['red'].append((pos, r, r))
        cmap_data['green'].append((pos, g, g))
        cmap_data['blue'].append((pos, b, b))
    
    return LinearSegmentedColormap('hot_metal', cmap_data)


def generate_brain_texture(resolution: int = 200) -> np.ndarray:
    """
    Generate a realistic brain-like texture pattern.
    
    Creates the folded surface (gyri and sulci) appearance.
    """
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create brain mask (slightly oval)
    brain_mask = (X**2 / 0.85**2 + Y**2 / 0.95**2) <= 1
    
    # Generate sulci (brain folds) pattern
    np.random.seed(42)  # Consistent pattern
    
    # Multiple frequency components for realistic folds
    texture = np.zeros_like(X)
    
    # Large-scale structure
    texture += 0.3 * np.sin(4 * X + 0.5) * np.cos(3 * Y)
    texture += 0.2 * np.sin(6 * Y + 1) * np.cos(5 * X)
    
    # Medium folds
    texture += 0.15 * np.sin(12 * X) * np.sin(10 * Y)
    texture += 0.15 * np.cos(8 * X + 2 * Y)
    
    # Fine detail (small gyri)
    noise = np.random.randn(resolution, resolution) * 0.1
    noise = gaussian_filter(noise, sigma=3)
    texture += noise
    
    # Normalize
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    
    # Apply brain mask
    texture[~brain_mask] = 0
    
    return texture, brain_mask


def generate_brain_slice(
    view: str = "axial",
    resolution: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a highly detailed realistic brain slice image.
    
    Creates an MRI-like brain slice with clearly visible anatomical structures:
    - Gray matter (cortex) with gyri and sulci
    - White matter (internal)
    - Ventricles
    - Corpus callosum
    - Central structures
    
    Args:
        view: Slice orientation ('axial', 'coronal', 'sagittal')
        resolution: Image resolution
    """
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    if view == "axial":
        # Top-down view (most common for EEG)
        
        # Outer brain boundary (slightly irregular for realism)
        boundary_mod = 0.02 * np.sin(12 * np.arctan2(Y, X))
        brain_outline = (X**2 / (0.82 + boundary_mod)**2 + Y**2 / (0.92 + boundary_mod)**2) <= 1
        
        # CSF space (thin layer around brain)
        csf_outer = (X**2 / 0.86**2 + Y**2 / 0.96**2) <= 1
        csf_space = csf_outer & ~brain_outline
        
        # Lateral ventricles (butterfly/horn shape)
        # Anterior horns (front)
        vent_ant_l = ((X + 0.12)**2 / 0.06**2 + (Y - 0.15)**2 / 0.12**2) <= 1
        vent_ant_r = ((X - 0.12)**2 / 0.06**2 + (Y - 0.15)**2 / 0.12**2) <= 1
        # Body of ventricles
        vent_body_l = ((X + 0.18)**2 / 0.08**2 + Y**2 / 0.18**2) <= 1
        vent_body_r = ((X - 0.18)**2 / 0.08**2 + Y**2 / 0.18**2) <= 1
        # Posterior horns (back)  
        vent_post_l = ((X + 0.15)**2 / 0.05**2 + (Y + 0.2)**2 / 0.10**2) <= 1
        vent_post_r = ((X - 0.15)**2 / 0.05**2 + (Y + 0.2)**2 / 0.10**2) <= 1
        
        ventricles = (vent_ant_l | vent_ant_r | vent_body_l | vent_body_r | 
                     vent_post_l | vent_post_r) & (np.abs(Y) < 0.35)
        
        # Third ventricle (midline)
        third_vent = (np.abs(X) < 0.02) & (Y > -0.1) & (Y < 0.2)
        ventricles = ventricles | third_vent
        
        # Central structures (thalamus, basal ganglia)
        thalamus_l = ((X + 0.12)**2 + (Y + 0.02)**2) < 0.08**2
        thalamus_r = ((X - 0.12)**2 + (Y + 0.02)**2) < 0.08**2
        caudate_l = ((X + 0.18)**2 / 0.04**2 + (Y - 0.08)**2 / 0.08**2) <= 1
        caudate_r = ((X - 0.18)**2 / 0.04**2 + (Y - 0.08)**2 / 0.08**2) <= 1
        putamen_l = ((X + 0.25)**2 / 0.05**2 + Y**2 / 0.12**2) <= 1
        putamen_r = ((X - 0.25)**2 / 0.05**2 + Y**2 / 0.12**2) <= 1
        deep_gray = thalamus_l | thalamus_r | caudate_l | caudate_r | putamen_l | putamen_r
        
        # Corpus callosum (connects hemispheres)
        corpus = (np.abs(X) < 0.25) & (np.abs(Y) < 0.08) & ~ventricles
        
        # White matter boundary (more detailed)
        wm_mod = 0.03 * np.sin(8 * np.arctan2(Y, X))
        inner_boundary = (X**2 / (0.55 + wm_mod)**2 + Y**2 / (0.65 + wm_mod)**2) <= 1
        
        # Gray matter (cortex) - outer region with gyri/sulci
        gray_matter = brain_outline & ~inner_boundary
        
        # White matter - inner region
        white_matter = inner_boundary & ~ventricles & ~deep_gray
        
        # Create detailed intensity image
        brain_img = np.zeros((resolution, resolution, 3), dtype=np.float32)
        
        # Background (dark)
        brain_img[:, :] = [0.05, 0.05, 0.08]
        
        # CSF (dark, slightly blue)
        brain_img[csf_space] = [0.1, 0.12, 0.15]
        
        # Base brain structure
        brain_img[brain_outline] = [0.35, 0.32, 0.30]
        
        # White matter (bright, slightly warm)
        brain_img[white_matter] = [0.85, 0.82, 0.78]
        
        # Corpus callosum (bright white)
        brain_img[corpus & ~ventricles] = [0.95, 0.93, 0.90]
        
        # Gray matter (medium gray, cooler)
        brain_img[gray_matter] = [0.55, 0.52, 0.50]
        
        # Deep gray matter structures (darker gray)
        brain_img[deep_gray] = [0.45, 0.43, 0.42]
        
        # Ventricles (very dark - CSF)
        brain_img[ventricles] = [0.08, 0.10, 0.12]
        
        # Add detailed sulci texture to gray matter
        texture, _ = generate_brain_texture(resolution)
        
        # Create sulci pattern (darker lines in gray matter)
        sulci_threshold = 0.4
        sulci_mask = (texture < sulci_threshold) & gray_matter
        
        # Darken sulci
        brain_img[sulci_mask] *= 0.6
        
        # Add subtle variation to white matter
        wm_variation = gaussian_filter(np.random.randn(resolution, resolution), sigma=8) * 0.05
        brain_img[white_matter, 0] += wm_variation[white_matter]
        brain_img[white_matter, 1] += wm_variation[white_matter]
        brain_img[white_matter, 2] += wm_variation[white_matter]
        
        # Add fiber tract hints in white matter
        fiber_pattern = np.sin(20 * X) * np.cos(15 * Y) * 0.03
        brain_img[white_matter, 0] += fiber_pattern[white_matter]
        brain_img[white_matter, 1] += fiber_pattern[white_matter]
        
        # Smooth the whole image slightly
        for c in range(3):
            brain_img[:, :, c] = gaussian_filter(brain_img[:, :, c], sigma=0.8)
        
        # Ensure values in range
        brain_img = np.clip(brain_img, 0, 1)
        
        # Create grayscale version for compatibility
        brain_gray = np.mean(brain_img, axis=2)
        brain_gray[~brain_outline] = 0
        
    else:
        # Simplified for other views
        texture, brain_outline = generate_brain_texture(resolution)
        brain_gray = texture
        brain_img = np.stack([texture, texture, texture], axis=2)
        brain_img[~brain_outline] = 0
    
    return brain_img, brain_outline, brain_gray


class AnatomicalBrainMap:
    """
    Creates anatomically realistic brain maps with EEG activity overlay.
    
    Similar to PET/fMRI visualizations with actual brain structure visible.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        resolution: int = 300,
        colormap: str = "pet",
    ):
        """
        Initialize anatomical brain map.
        
        Args:
            channel_names: Channel names in data order.
            resolution: Image resolution.
            colormap: 'pet', 'hot', or matplotlib colormap name.
        """
        self.channel_names = channel_names or STANDARD_CHANNEL_ORDER
        self.resolution = resolution
        
        # Set up colormap
        if colormap == "pet":
            self._colormap = create_pet_colormap()
        elif colormap == "hot":
            self._colormap = create_hot_metal_colormap()
        else:
            self._colormap = plt.cm.get_cmap(colormap)
        
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
        
        # Generate brain anatomy (RGB and grayscale)
        self.brain_img_rgb, self.brain_mask, self.brain_img = generate_brain_slice("axial", resolution)
        
        # Create interpolation grid
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        
    def interpolate_activity(self, values: np.ndarray, smoothing: float = 2.0) -> np.ndarray:
        """Interpolate electrode values to brain surface."""
        valid_values = values[self.valid_channels]
        
        # RBF interpolation for smooth activity map
        try:
            rbf = Rbf(
                self.positions[:, 0],
                self.positions[:, 1],
                valid_values,
                function='thin_plate',
                smooth=0.1
            )
            activity = rbf(self.grid_x, self.grid_y)
        except:
            # Fallback to griddata
            activity = griddata(
                self.positions,
                valid_values,
                (self.grid_x, self.grid_y),
                method='cubic',
                fill_value=0
            )
        
        # Smooth the activity
        activity = gaussian_filter(activity, sigma=smoothing)
        
        # Mask to brain shape
        activity[~self.brain_mask] = np.nan
        
        return activity
    
    def create_figure(
        self,
        values: np.ndarray,
        title: str = "",
        show_anatomy: bool = True,
        show_electrodes: bool = True,
        show_names: bool = False,
        show_regions: bool = False,
        show_colorbar: bool = True,
        activity_alpha: float = 0.55,
        figsize: Tuple[float, float] = (8, 8),
        style: str = "dark",
    ) -> plt.Figure:
        """
        Create an anatomical brain map with activity overlay.
        
        The brain anatomy is ALWAYS visible underneath the activity heatmap.
        Activity colors are blended with the anatomical structure.
        
        Args:
            values: Array of values for each channel.
            title: Plot title.
            show_anatomy: Show underlying brain structure (always recommended).
            show_electrodes: Show electrode positions.
            show_names: Show electrode labels.
            show_regions: Show brain region labels.
            show_colorbar: Show activity colorbar.
            activity_alpha: Transparency of activity overlay (lower = more brain visible).
            figsize: Figure size.
            style: 'dark' or 'light' theme.
        """
        # Interpolate activity
        activity = self.interpolate_activity(values)
        
        # Normalize activity
        vmin = np.nanpercentile(activity, 5)
        vmax = np.nanpercentile(activity, 95)
        if vmax <= vmin:
            vmax = vmin + 1
        
        # Set up figure
        if style == "dark":
            fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a0f')
            ax.set_facecolor('#0a0a0f')
            text_color = 'white'
        else:
            fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            text_color = 'white'
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.axis('off')
        
        # FIRST: Draw the anatomical brain (RGB image) as the base
        if show_anatomy:
            # Show the full-color MRI-style brain image
            ax.imshow(
                self.brain_img_rgb,
                extent=[-1, 1, -1, 1],
                origin='lower',
                alpha=1.0,  # Full opacity for base brain
                interpolation='bilinear'
            )
        
        # SECOND: Normalize activity to 0-1 for colormap
        activity_norm = (activity - vmin) / (vmax - vmin)
        activity_norm = np.clip(activity_norm, 0, 1)
        
        # Create RGBA activity overlay
        activity_rgba = self._colormap(activity_norm)
        
        # Make background transparent (where no brain)
        activity_rgba[~self.brain_mask] = [0, 0, 0, 0]
        
        # Set alpha channel for activity overlay
        activity_rgba[:, :, 3] = np.where(self.brain_mask, activity_alpha, 0)
        
        # Draw activity overlay on top of brain
        im = ax.imshow(
            activity_rgba,
            extent=[-1, 1, -1, 1],
            origin='lower',
            interpolation='bilinear'
        )
        
        # Create a dummy mappable for colorbar
        sm = plt.cm.ScalarMappable(cmap=self._colormap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        
        # Draw brain outline
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(0.85 * np.cos(theta), 0.95 * np.sin(theta), 
                color=text_color, linewidth=2, alpha=0.7)
        
        # Draw nose indicator (pointing up = front of brain)
        nose_x = [0, -0.08, 0, 0.08, 0]
        nose_y = [0.95, 1.02, 1.1, 1.02, 0.95]
        ax.fill(nose_x, nose_y, color=text_color, alpha=0.6)
        ax.text(0, 1.12, 'Front', fontsize=8, ha='center', color=text_color, alpha=0.5)
        
        # Draw ears
        ear_theta = np.linspace(-np.pi/2, np.pi/2, 20)
        ax.fill(-0.88 + 0.08*np.cos(ear_theta), 0.12*np.sin(ear_theta),
                color=text_color, alpha=0.4)
        ax.fill(0.88 - 0.08*np.cos(ear_theta), 0.12*np.sin(ear_theta),
                color=text_color, alpha=0.4)
        ax.text(-0.98, 0, 'L', fontsize=9, ha='center', va='center', color=text_color, alpha=0.5)
        ax.text(0.98, 0, 'R', fontsize=9, ha='center', va='center', color=text_color, alpha=0.5)
        
        # Draw electrodes
        if show_electrodes:
            for i, (name, pos) in enumerate(zip(self.electrode_names, self.positions)):
                val = values[self.valid_channels[i]]
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = self._colormap(np.clip(norm_val, 0, 1))
                
                # Electrode marker with glow effect
                ax.scatter(pos[0], pos[1], s=120, c='white', alpha=0.3, zorder=9)  # glow
                ax.scatter(pos[0], pos[1], s=70, c=[color], 
                          edgecolors='white', linewidths=1.5, zorder=10)
                
                if show_names:
                    ax.text(pos[0], pos[1] + 0.09, name, fontsize=7,
                           ha='center', color=text_color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='black', 
                                    alpha=0.5, edgecolor='none'))
        
        # Draw region labels
        if show_regions:
            for region_name, region_info in BRAIN_REGIONS.items():
                cx, cy = region_info["center"]
                short_name = region_name.split()[0]
                ax.text(cx, cy, short_name,
                       fontsize=9, ha='center', va='center',
                       color='white', alpha=0.7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', 
                                alpha=0.4, edgecolor='none'))
        
        # Colorbar
        if show_colorbar:
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=30)
            cbar.ax.tick_params(labelsize=9, colors=text_color)
            cbar.outline.set_edgecolor(text_color)
            cbar.ax.set_ylabel('Activity (µV)', fontsize=10, 
                              color=text_color, rotation=270, labelpad=15)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', 
                        color=text_color, pad=10)
        
        plt.tight_layout()
        return fig
    
    def to_image_bytes(self, values: np.ndarray, format: str = 'png', 
                       dpi: int = 100, **kwargs) -> bytes:
        """Return brain map as image bytes."""
        fig = self.create_figure(values, **kwargs)
        buf = BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


class Anatomical3DBrain:
    """
    Creates 3D brain visualization with anatomical structure.
    """
    
    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        resolution: int = 50,
    ):
        self.channel_names = channel_names or STANDARD_CHANNEL_ORDER
        self.resolution = resolution
        self._colormap = create_pet_colormap()
        
        # Get electrode positions (3D)
        from visualization.brain_3d import ELECTRODE_POSITIONS_3D
        
        self.positions_3d = []
        self.valid_channels = []
        self.electrode_names = []
        
        for i, name in enumerate(self.channel_names):
            if name in ELECTRODE_POSITIONS_3D:
                self.positions_3d.append(ELECTRODE_POSITIONS_3D[name])
                self.valid_channels.append(i)
                self.electrode_names.append(name)
        
        self.positions_3d = np.array(self.positions_3d)
        
        # Generate brain mesh
        self._create_brain_mesh()
    
    def _create_brain_mesh(self):
        """Create anatomically-shaped brain mesh."""
        u = np.linspace(0, 2 * np.pi, self.resolution)
        v = np.linspace(0, np.pi, self.resolution)
        
        # Base ellipsoid (brain shape)
        self.x = 0.85 * np.outer(np.cos(u), np.sin(v))
        self.y = 0.90 * np.outer(np.sin(u), np.sin(v))
        self.z = 0.75 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Add sulci (brain folds) - subtle surface variation
        np.random.seed(42)
        sulci = 0.03 * np.sin(8*np.outer(u, np.ones_like(v))) * np.sin(6*np.outer(np.ones_like(u), v))
        self.x += sulci * self.x
        self.y += sulci * self.y
        
        # Shift up to sit on "neck"
        self.z += 0.2
        
        # Only keep top hemisphere (brain visible from above)
        mask = self.z > 0.1
        
    def interpolate_on_surface(self, values: np.ndarray) -> np.ndarray:
        """Interpolate values onto brain surface."""
        valid_values = values[self.valid_channels]
        
        rbf = Rbf(
            self.positions_3d[:, 0],
            self.positions_3d[:, 1],
            self.positions_3d[:, 2],
            valid_values,
            function='thin_plate',
            smooth=0.1
        )
        
        return rbf(self.x, self.y, self.z)
    
    def create_figure(
        self,
        values: np.ndarray,
        title: str = "",
        view: str = "iso",
        show_electrodes: bool = True,
        figsize: Tuple[float, float] = (10, 10),
    ) -> plt.Figure:
        """Create 3D anatomical brain visualization."""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize, facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        # Interpolate activity
        activity = self.interpolate_on_surface(values)
        
        # Normalize
        vmin, vmax = np.nanpercentile(activity, [5, 95])
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = self._colormap(norm(activity))
        
        # Draw brain surface
        ax.plot_surface(
            self.x, self.y, self.z,
            facecolors=colors,
            shade=True,
            alpha=0.95,
            linewidth=0,
            antialiased=True
        )
        
        # Draw electrodes
        if show_electrodes:
            for i, pos in enumerate(self.positions_3d):
                val = values[self.valid_channels[i]]
                color = self._colormap(norm(val))
                ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          s=60, c=[color], edgecolors='white', linewidths=1)
        
        # Set view
        views = {
            'top': (90, 0),
            'front': (0, 90),
            'iso': (30, 45),
        }
        elev, azim = views.get(view, (30, 45))
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-0.5, 1.1)
        ax.set_axis_off()
        
        if title:
            ax.set_title(title, fontsize=14, color='white', pad=10)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=self._colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.ax.tick_params(colors='white')
        cbar.outline.set_edgecolor('white')
        
        plt.tight_layout()
        return fig
    
    def to_image_bytes(self, values: np.ndarray, format: str = 'png',
                       dpi: int = 80, **kwargs) -> bytes:
        """Return as image bytes."""
        fig = self.create_figure(values, **kwargs)
        buf = BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


class FastBrainRenderer:
    """
    Optimized brain renderer for real-time updates.
    
    Uses caching and reduced resolution for fast rendering.
    """
    
    def __init__(self, channel_names: Optional[List[str]] = None):
        self.brain_2d = AnatomicalBrainMap(channel_names, resolution=150)
        self._cache = {}
        self._last_values = None
        
    def render_fast(
        self,
        values: np.ndarray,
        size: Tuple[int, int] = (300, 300),
        colormap: str = "pet",
    ) -> bytes:
        """
        Fast rendering for real-time display.
        
        Uses lower resolution and minimal decorations.
        """
        # Check if values changed significantly
        if self._last_values is not None:
            diff = np.abs(values - self._last_values).mean()
            if diff < 0.1:  # Skip if minimal change
                if 'last_image' in self._cache:
                    return self._cache['last_image']
        
        self._last_values = values.copy()
        
        img_bytes = self.brain_2d.to_image_bytes(
            values,
            show_anatomy=True,
            show_electrodes=True,
            show_names=False,
            show_regions=False,
            show_colorbar=False,
            activity_alpha=0.75,
            figsize=(4, 4),
            dpi=75,
            style="dark"
        )
        
        self._cache['last_image'] = img_bytes
        return img_bytes
