import os
import subprocess
from pathlib import Path
from typing import Optional, Union

def resize_image(img_path: Union[Path, str], output_path: Optional[Union[Path, str]] = None, 
                scale_filter: str = "shouldthrowerror", quality: int = 1) -> Optional[Path]:
    """
    Resize an image using ffmpeg with the specified scale filter and quality.
    
    Args:
        img_path: Path to the input image
        output_path: Directory to save the resized image, or None to modify in place
        scale_filter: ffmpeg scale filter (e.g., "scale=iw/2:-1" for half width)
        quality: ffmpeg quality value (lower is better quality, 1-31 range)
        
    Returns:
        Path to the resized image or None if operation failed
    """
    img_path = Path(img_path)
    
    if not img_path.exists():
        print(f"Input file does not exist: {img_path}")
        return None

    try:
        in_place = output_path is None
        target_file = img_path.with_suffix('.tmp' + img_path.suffix) if in_place else Path(output_path) / img_path.name
        
        if not in_place:
            os.makedirs(Path(output_path), exist_ok=True)
            
        cmd = [
            'ffmpeg', 
            '-hide_banner', 
            '-loglevel', 'error',
            '-i', str(img_path), 
            '-map_metadata 0'
            '-vf', scale_filter, 
            '-q:v', str(quality),
            str(target_file)
        ]
                
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
        if target_file.exists():
            if in_place:
                target_file.replace(img_path)
                return img_path
            return target_file
        
        if result.stderr:
            print(f"Error processing {img_path}: {result.stderr}")
        return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error processing {img_path}: {e}")
        if e.stderr:
            print(f"ffmpeg error: {e.stderr}")
        return None
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

