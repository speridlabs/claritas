# Claritas

A Python library for selecting and managing sharp images from videos or image collections.

## Features

- Extract frames from videos
- Select sharpest images from a collection
- Cache sharpness computations
- Parallel processing support
- Progress visualization
- In-place or copy mode operations
- CUDA support native

## Installation

```bash
pip install claritas
```

## Usage

```python
from claritas import ImageProcessor

# Create processor instance
processor = ImageProcessor(
    workers=4,  # Number of parallel workers
    show_progress=True,  # Show progress bars
    use_cache=True  # Enable computation caching
)

# Process video
processor.process_video("video.mp4", "frames/")

# Select sharp images
processor.select_sharp_images(
    input_path="input/folder",
    output_path="output/folder",
    target_count=100,  # Number of images to keep
    groups=10  # Number of groups for distribution
)
```

## CLI Usage

For processing a video file:
```bash
claritas --input video.mp4 --output frames/ --count 100 --workers 4
```

For selecting sharp images from a directory:
```bash
claritas --input input/folder --output output/folder --count 100 --groups 10
```

Or to modify in-place:
```bash
claritas --input input/folder --count 100
```

Or to resize:
```bash
claritas --input input/folder --resize 1000
```

For more information, see the [documentation](https://github.com/yourusername/claritas).
