import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pycolmap import SceneManager
from sklearn.cluster import DBSCAN
from typing import List, Dict, Optional, Tuple, Set, Union

from .cache import SharpnessCache

class ColmapPruner:
    """
    Prunes redundant images from a COLMAP reconstruction based on
    spatial proximity, viewing angles, and image sharpness.
    """
    
    def __init__(
        self,
        colmap_dir: Path,
        sharpness_cache: SharpnessCache,
        distance_threshold: float = 0,
        angle_threshold: float = 30.0,
        reduction_ratio: float = 0.5,
    ):
        """
        Initialize the COLMAP pruner.
        
        Args:
            colmap_dir: Path to the COLMAP reconstruction
            sharpness_cache_path: Path to the claritas sharpness cache
            distance_threshold: Distance threshold for clustering. If None,
                                will be automatically determined based on scene scale.
            angle_threshold: Angle threshold in degrees for considering different viewing directions
            reduction_ratio: Target ratio of images to keep (0.0-1.0)
        """
        self.colmap_dir = colmap_dir
        self.sharpness_cache = sharpness_cache
        self.angle_threshold = angle_threshold  # in degrees
        self.reduction_ratio = reduction_ratio
        self.distance_threshold:float = distance_threshold
        
        # Will be loaded later
        self.scene_scale = 0
        self.scene_manager: SceneManager = SceneManager("")

        self.image_names: Dict[int, str] = {}
        self.camera_positions: Dict[int, np.ndarray] = {}
        self.camera_orientations: Dict[int, np.ndarray] = {}

        self.camera_masks: Dict[int, np.ndarray] = {}
        self.camera_world2camera_matrix: Dict[int, np.ndarray] = {}
        self.camera_parameters: Dict[str, Dict[int, Union[np.ndarray, float]]] = {}

        raise NotImplementedError("Not implemented yet")
    
    def load_scene(self):
        """Load the COLMAP scene and extract camera data."""

        if (self.colmap_dir / "sparse" / "0").exists():
            sparse_dir = self.colmap_dir / "sparse" / "0"
        elif (self.colmap_dir / "sparse").exists():
            sparse_dir = self.colmap_dir / "sparse"
        else:
            sparse_dir = self.colmap_dir
            
        raise NotImplementedError("Loading scene from path is not implemented yet")
    
    def _quaternion_angle(self, q1, q2):
        """Calculate angle between two quaternions in degrees."""
        # Normalize quaternions
        q1_norm = q1 / np.linalg.norm(q1)
        q2_norm = q2 / np.linalg.norm(q2)
        
        # Calculate dot product and clamp to [-1, 1] to avoid numerical issues
        dot_product = np.clip(np.abs(np.sum(q1_norm * q2_norm)), -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = 2 * np.arccos(dot_product) * 180 / np.pi
        
        return angle
