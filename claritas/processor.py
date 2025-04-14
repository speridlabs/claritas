import os
from typing import Optional
import cv2
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from .resize import resize_image
from .colmap import ColmapPruner
from .cache import SharpnessCache

class ColmapProcessor:
    workers: int
    show_progress: bool
    cache: SharpnessCache

    """Class for processing images using COLMAP."""
    def __init__(self, workers=None, show_progress=True, cache=SharpnessCache | None):
        """
        Initialize the COLMAP processor.
        
        Args:
            workers: Number of parallel workers (None for auto)
            show_progress: Show progress bars
        """
        self.workers = workers if workers else max(1, (os.cpu_count() or 2) - 1)
        self.show_progress = show_progress
        self.cache = SharpnessCache()

        if isinstance(cache, SharpnessCache):
            self.cache = cache

    # TODO: add colmap purning options here
    def prune_colmap(self, colmap_dir:str, output_dir:Optional[str]=None):
        """
        Prune images based on a COLMAP reconstruction to remove redundant views.

        Args:
            colmap_dir: Path to the COLMAP reconstruction
            output_path: Directory to save the selected sharp images
                        If None, only returns list of images to keep
            distance_threshold: Distance threshold for clustering (if None, auto-calculated)
            angle_threshold: Angle threshold in degrees for viewing direction similarity
            reduction_ratio: Target ratio of images to keep (0.0-1.0)
        """
        colmap_path = Path(colmap_dir)
        if not colmap_path.exists(): raise ValueError("Invalid COLMAP path")

        images_path = colmap_path / "images"
        if not images_path.exists(): 
            images_path = colmap_path.parent / "images"
            if not images_path.exists(): raise ValueError("Images not found in COLMAP path")

        output_path: Optional[Path] = None

        if output_dir:
            output_path = Path(output_dir)
            if output_path.exists() and not output_path.is_dir():
                raise ValueError("Output path exists but is not a directory")
            output_path.mkdir(parents=True, exist_ok=True)
    
        # TODO: support multiple images folder for one reconstruction
        pruner = ColmapPruner(
            colmap_dir=colmap_path,
            sharpness_cache=self.cache,
        )
    
        raise NotImplementedError("Not implemented yet")
        images_to_keep = pruner.prune(output_dir=output_path)

        # TODO: this does not work in case of not ouput path
        def copy_file(src, dst):
            shutil.copy2(src, dst)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            try:
                list(tqdm(executor.map(copy_file, images_to_keep, [output_path / img.name for img in images_to_keep]), total=len(images_to_keep), desc="Copying selected images"))
            except KeyboardInterrupt:
                print("\nInterrupted by user. Shutting down...")
                executor.shutdown(wait=False)

class ImageProcessor:

    workers: int
    show_progress: bool
    cache: SharpnessCache | None

    def __init__(self, workers=None, show_progress=True, use_cache=True):

        """
        Initialize the image processor.
        
        Args:

            workers: Number of parallel workers (None for auto)
            show_progress: Show progress bars
            use_cache: Enable computation caching
        """
        self.workers = workers if workers else max(1, (os.cpu_count() or 2) - 1)
        self.show_progress = show_progress
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
        image_path = Path(image_path).resolve()  # Get absolute path and resolve symlinks

        if self.cache:
            cached = self.cache.get(image_path)
            if cached is not None:
                return cached

        try:

            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
        except Exception as e:
            raise ValueError(f"Error reading image {image_path}: {str(e)}")
            
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
        except Exception as e:
            if self.show_progress:
                print(f"CUDA processing failed, falling back to CPU: {str(e)}")
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
        if self.cache:
            self.cache.set(image_path, score)
            
        return score
        
    def resize_images(self, input_path, output_path=None, width=None, height=None, max_size=None):
        """
        Resize images using ffmpeg, maintaining aspect ratio.
        
        Args:
            input_path: Path to input image or directory containing images
            output_path: Directory to save resized images. If None, modifies in-place
            width: Target width (height will be calculated to maintain aspect ratio)
            height: Target height (width will be calculated to maintain aspect ratio)
            max_size: Maximum size for either dimension (maintains aspect ratio)

            
        Returns:
            List of paths to resized images
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            if not any(ext in input_path.name.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']):
                raise ValueError(f"Unsupported file type: {input_path}")
            images = [input_path]
            is_dir = False
        elif input_path.is_dir():
            extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            images = [p for p in input_path.rglob('*') if p.suffix.lower() in extensions]
            is_dir = True
        else:
            raise ValueError(f"Invalid input path: {input_path}")
            
        if not images:
            raise ValueError("No images found")
            
        if output_path:
            output_path = Path(output_path)
            if output_path.exists() and not output_path.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)
        
        if not any([width, height, max_size]):
            raise ValueError("Either width, height, or max_size must be specified")
        
        scale_filter = None
        
        if max_size is not None:
            scale_filter = f"scale='if(gt(iw,ih),min(iw,{max_size}),-2)':'if(gt(iw,ih),-2,min(ih,{max_size}))'"
        elif width is not None:
            scale_filter = f"scale={width}:-2"
        elif height is not None:
            scale_filter = f"scale=-2:{height}"
        
        if scale_filter is None:
            raise ValueError("Failed to create a valid scale filter. Please specify width, height, or max_size.")
        
        # Process images in parallel
        resized_images = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:

            future_to_path = {
                executor.submit(
                    resize_image, 
                    img, 
                    output_path if output_path else None,
                    scale_filter,
                    1
                ): img for img in images
            }
            
            try:
                if self.show_progress:
                    futures = tqdm(as_completed(future_to_path), total=len(images), desc="Resizing images")
                else:
                    futures = as_completed(future_to_path)
                    
                for future in futures:
                    result = future.result()
                    if result:
                        resized_images.append(result)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Shutting down...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        
        # Copy cache file if applicable
        if is_dir and output_path and self.cache:
            cache_path = input_path / '.claritas_cache.json'
            if cache_path.exists():
                output_cache = output_path / '.claritas_cache.json'
                shutil.copy2(str(cache_path), str(output_cache))
                
        return resized_images
        
    def select_sharp_images(self, input_path, output_path=None, target_count=None, target_percentage=None, groups=None):
        """
        Select sharp images from input directory.
        
        Args:
            input_path: Directory containing input images
            output_path: Directory to save selected images. If None, modifies in-place
            target_count: Number of images to keep
            target_percentage: The percentage of images to keep
            groups: Number of groups for distribution

        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise ValueError("Invalid input path")

        if self.cache:
            cache_path = input_path / '.claritas_cache.json'
            self.cache.cache_file = str(cache_path)
            self.cache.load()
            
        if output_path:
            output_path = Path(output_path)
            
        # Get all images
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        images = [p for p in input_path.rglob('*') if p.suffix.lower() in extensions]
        
        if not images:
            raise ValueError("No images found")
            
        total = len(images)

        # Validate input parameters
        if target_count is None and target_percentage is None:
            raise ValueError("Either target_count or target_percentage must be specified")
            
        if groups:
            if groups > total:
                groups = total  # Adjust groups if too many
            group_size = max(1, total // groups)
            if target_percentage is not None:
                images_per_group = max(1, int(round((target_percentage / 100.0) * group_size)))
                total_selected = images_per_group * groups
                print(f"Processing {total} images in {groups} groups, selecting {images_per_group} ({target_percentage}%) from each group of ~{group_size} images")
            else:
                if target_count is None:
                    raise ValueError("target_count must be specified when using groups without target_percentage")
                images_per_group = max(1, target_count // groups)
                remainder = target_count % groups  # Handle non-divisible target count
                total_selected = target_count
                print(f"Processing {total} images in {groups} groups, selecting {images_per_group} from each group of ~{group_size} images (plus {remainder} extra)")
        else:
            if target_percentage is not None:
                if target_percentage >= 100 or target_percentage <= 0:
                    raise ValueError("Invalid target percentage")

                total_selected = max(1, int(round((target_percentage / 100.0) * total)))
                print(f"Processing {total} images, selecting {total_selected} images ({target_percentage}%)")
            else:
                if target_count is None:
                    raise ValueError("Either target_count or target_percentage must be specified")
                total_selected = min(target_count, total)
                print(f"Processing {total} images, selecting {total_selected} images")
        
        # Compute sharpness scores
        scores = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_path = {executor.submit(self.compute_sharpness, img): img 
                            for img in images}
            try:
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
            except KeyboardInterrupt:
                print("\nInterrupted by user. Shutting down...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
                    
        # Handle grouping
        if groups:
            selected = []
            # Split into temporal groups first (maintaining original order)
            # Process each temporal group separately
            remainder = 0 if target_percentage is not None else (target_count % groups if target_count is not None else 0)
            for i in range(groups):
                start = i * group_size
                end = start + group_size if i < groups - 1 else total
                # Sort sharpness scores WITHIN this group only
                group_scores = scores[start:end]
                group_scores.sort(reverse=True)
                # Add one extra image to early groups if we have remainder
                this_group_count = images_per_group + (1 if i < remainder else 0)
                selected.extend(path for _, path in group_scores[:this_group_count])
        else:
            # No grouping - sort all by sharpness
            scores.sort(reverse=True)
            # Use total_selected instead of target_count since it's calculated for both percentage and count modes
            selected = [path for _, path in scores[:total_selected]]
            
        if not output_path:
            for score, path in scores:
                if path not in selected:
                    path.unlink()
        else:
            output_path.mkdir(parents=True, exist_ok=True)

            def copy_file(src):
                shutil.copy2(src, output_path / src.name)

            with ThreadPoolExecutor(max_workers=self.workers) as copy_executor:
                # Using executor.map with tqdm for progress display
                try:
                    list(tqdm(copy_executor.map(copy_file, selected), total=len(selected), desc="Copying images"))
                except KeyboardInterrupt:
                    print("\nInterrupted by user. Shutting down...")
                    copy_executor.shutdown(wait=False)

        if self.cache:
            self.cache.save()
            # save in output cache also
            if output_path:
                output_cache = output_path / '.claritas_cache.json'
                self.cache.cache_file = str(output_cache)
                self.cache.save()
            
        return selected
