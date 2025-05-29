import os
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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

