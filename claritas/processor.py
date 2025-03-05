import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import subprocess
from .cache import SharpnessCache

class ImageProcessor:
    def __init__(self, workers=None, show_progress=True, use_cache=True):
        """
        Initialize the image processor.
        
        Args:

            workers: Number of parallel workers (None for auto)
            show_progress: Show progress bars
            use_cache: Enable computation caching
        """

        self.workers = workers if workers else max(1, os.cpu_count() - 1)
        self.show_progress = show_progress
        self.use_cache = use_cache
        self.cache = SharpnessCache() if use_cache else None
        
    def process_video(self, input_path, output_path):
        """
        Extract frames from video.
        
        Args:
            input_path: Path to input video file
            output_path: Directory to save extracted frames
        """
        video_path = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vsync', 'vfr',
            '-v', 'quiet',
            '-stats',
            '-q:v', '1',
            str(output_dir / 'frame%05d.jpg')
        ]
        subprocess.run(cmd)
        
        return output_dir
        
    def compute_sharpness(self, image_path):
        """Compute image sharpness score."""
        if self.use_cache:
            cached = self.cache.get(image_path)
            if cached is not None:
                return cached
                
        img = cv2.imread(str(image_path))
        if img is None:
            return 0
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try CUDA if available
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(gray)
                laplacian = cv2.cuda.Laplacian(gpu_mat, cv2.CV_64F)
                score = laplacian.download().var()
            else:
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
        if self.use_cache:
            self.cache.set(image_path, score)
            
        return score
        
    def select_sharp_images(self, input_path, output_path=None, target_count=None, groups=None):
        """
        Select sharp images from input directory.
        
        Args:
            input_path: Directory containing input images
            output_path: Directory to save selected images. If None, modifies in-place
            target_count: Number of images to keep
            groups: Number of groups for distribution

        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise ValueError("Invalid input path")


        if self.use_cache:
            cache_path = input_path / '.claritas_cache.json'
            self.cache.cache_file = cache_path
            
        in_place = output_path is None
        if not in_place:
            output_path = Path(output_path)
            
        # Get all images
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [p for p in input_path.rglob('*') if p.suffix.lower() in extensions]
        
        if not images:
            raise ValueError("No images found")
            
        # Compute sharpness scores
        scores = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_path = {executor.submit(self.compute_sharpness, img): img 
                            for img in images}
            
            if self.show_progress:
                futures = tqdm(future_to_path, total=len(images), desc="Computing sharpness")
            else:
                futures = future_to_path
                
            for future in futures:
                path = future_to_path[future]
                try:
                    score = future.result()
                    scores.append((score, path))
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    
        # Handle grouping
        if groups:
            # Split into temporal groups first (maintaining original order)
            total = len(scores)
            group_size = total // groups
            images_per_group = target_count // groups
            selected = []
            
            # Process each temporal group separately
            for i in range(groups):
                start = i * group_size
                end = start + group_size if i < groups - 1 else total
                # Sort sharpness scores WITHIN this group only
                group_scores = scores[start:end]
                group_scores.sort(reverse=True)
                selected.extend(path for _, path in group_scores[:images_per_group])
        else:
            # No grouping - sort all by sharpness
            scores.sort(reverse=True)
            selected = [path for _, path in scores[:target_count]]
            
        # Process files
        if in_place:
            # Remove non-selected files
            for score, path in scores:
                if path not in selected:
                    path.unlink()
        else:
            # Copy selected files
            self.output_path.mkdir(parents=True, exist_ok=True)
            for path in selected:
                shutil.copy2(path, self.output_path / path.name)
                
        if self.use_cache:
            self.cache.save()
            # save in output cache also
            output_cache = output_path / '.claritas_cache.json'
            self.cache.cache_file = output_cache
            self.cache.save()
            
        return selected
