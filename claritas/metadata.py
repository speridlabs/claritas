import os
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

def copy_video_metadata(src_video: Path, dst_frames: list[Path],
                        workers: Optional[int] = None, show_progress: bool = False):
    """Parallel copy of key QuickTime tags from video into JPEG frames."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    def copy_one(frame: Path):
        subprocess.run([
            'exiftool', '-q', '-q', '-overwrite_original',
            '-extractEmbedded',                 # load embedded tags if present
            '-tagsfromfile', str(src_video),
            '-QuickTime:Make>EXIF:Make',
            '-QuickTime:Model>EXIF:Model',
            '-QuickTime:LensModel-eng-ES>EXIF:LensModel',
            '-QuickTime:FocalLengthIn35mmFormat>EXIF:FocalLength',
            str(frame)
        ], check=True)

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(copy_one, f) for f in dst_frames]
        iterator = (
            tqdm(as_completed(futures), total=len(futures), desc="Copying video metadata")
            if show_progress else
            as_completed(futures)
        )
        for future in iterator:
            try:
                future.result()
            except Exception as e:
                print(f"Warning: video metadata copy failed: {e}")

def copy_metadata(src: Union[Path, str], dst: Union[Path, str]):
    """Copy all tags from src to dst in place."""
    subprocess.run([
        'exiftool', 
        '-q',          # quiet: no banner
        '-q',          # really quiet: no "n files updated"
        '-overwrite_original',
        '-tagsfromfile', str(src),
        '-all:all', str(dst)
    ], check=True)


def copy_metadata_bulk(
    src: Union[Path, str],
    dst_list: List[Path],
    workers: Optional[int] = None,
    show_progress: bool = False
):
    """Parallel copy metadata from src onto each path in dst_list."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(copy_metadata, src, dst): dst for dst in dst_list}

        iterator = (
            tqdm(as_completed(futures), total=len(dst_list), desc="Copying metadata")
            if show_progress
            else as_completed(futures)
        )

        for future in iterator:
            dst = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Warning: metadata copy failed for {dst}: {e}")

