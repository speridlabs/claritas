import argparse
from pathlib import Path
from .processor import ImageProcessor

def main():
    parser = argparse.ArgumentParser(
        description="Select and retain only the sharpest frames from a video or folder of images."
    )
    parser.add_argument('--input', required=True, help="Path to the input video or folder of images")
    parser.add_argument('--output', help="Directory to save the preserved images. Will modify in-place if not specified.")
    parser.add_argument('--count', type=int, required=True, help="Target number of images to retain")
    parser.add_argument('--groups', type=int, help="Number of groups for distribution")
    parser.add_argument('--workers', type=int, help="Number of parallel workers")
    parser.add_argument('--no-cache', action='store_true', help="Disable caching of sharpness calculations")
    parser.add_argument('--no-progress', action='store_true', help="Hide progress bars")
    parser.add_argument('--in-place', action='store_true', help="Modify input directory instead of copying")
    
    args = parser.parse_args()

    
    processor = ImageProcessor(
        workers=args.workers,
        show_progress=not args.no_progress,
        use_cache=not args.no_cache
    )
    
    try:
        if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            if not args.output:
                raise ValueError("Output directory required for video processing")
            processor.process_video(args.input, args.output)
            input_path = args.output
        else:
            input_path = args.input
            
        selected = processor.select_sharp_images(
            input_path=input_path,
            output_path=None if args.in_place else args.output,
            target_count=args.count,
            groups=args.groups
        )
        print(f"Successfully processed {len(selected)} images")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main()
