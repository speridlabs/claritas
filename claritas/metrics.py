import json
import logging
import pycolmap
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_sfm_metrics(sparse_dir: Path) -> Dict[str, float]:
    """Extract SfM metrics from COLMAP type dataset

    Args:
        sparse_dir: Directory containing sparse reconstruction files

    Returns:
        Dictionary with totalCameras, positionedCameras, pointcloudLength, firstPositionedCamera
    """
    metrics = {
        "totalCameras": 0.0,
        "positionedCameras": 0.0,
        "pointcloudLength": 0.0,
        "firstPositionedCamera": {
            "key": "",
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 0.0]
        }
    }

    try:
        reconstruction = pycolmap.Reconstruction(str(sparse_dir))

        metrics["totalCameras"] = float(reconstruction.num_images())
        metrics["positionedCameras"] = float(reconstruction.num_reg_images())
        metrics["pointcloudLength"] = float(reconstruction.num_points3D())

        if reconstruction.num_reg_images() > 0:
            registered_ids = [img_id for img_id, img in reconstruction.images.items() if img.has_pose]

            if not registered_ids: raise ValueError("No registered images found in reconstruction.")

            image = reconstruction.images[min(registered_ids)]
            cam_from_world = image.cam_from_world()

            # Extract translation (camera position in world coordinates)
            # We need to invert to get camera position: -R^T * t
            rotation_matrix = cam_from_world.rotation.matrix()
            translation = cam_from_world.translation
            camera_position = -rotation_matrix.T @ translation

            # Extract rotation as quaternion (qw, qx, qy, qz)
            quat = cam_from_world.rotation.quat

            metrics["firstPositionedCamera"] = {
                "key": image.name,
                "position": [float(camera_position[0]), float(camera_position[1]), float(camera_position[2])],
                "rotation": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            }


    except Exception as e:
        logger.error(f"Error extracting SfM metrics from {sparse_dir}: {e}")
        logger.warning("Returning zero metrics due to extraction error")

    return metrics


def extract_fornax_metrics(result_dir: Path) -> Dict[str, float]:
    """Extract Fornax training metrics from stats JSON files.

    Args:
        result_dir: Directory containing fornax training results

    Returns:
        Dictionary with psnr, ssim, lpips from the latest evaluation
    """
    metrics = {
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0
    }

    try:
        stats_dir = result_dir / "stats"
        if not stats_dir.exists():
            return metrics

        # Find the latest val_step*.json file
        val_files = list(stats_dir.glob("val_step*.json"))

        if not val_files:
            return metrics

        # Sort by step number and get the latest
        val_files.sort(key=lambda x: int(x.stem.split('step')[1]))
        latest_val_file = val_files[-1]


        with open(latest_val_file, 'r') as f:
            stats = json.load(f)

        # Extract the metrics we need
        if "psnr" in stats:
            metrics["psnr"] = float(stats["psnr"])
        if "ssim" in stats:
            metrics["ssim"] = float(stats["ssim"])
        if "lpips" in stats:
            metrics["lpips"] = float(stats["lpips"])

    except Exception as e:
        logger.error(f"Error extracting Fornax metrics: {e}")

    return metrics
