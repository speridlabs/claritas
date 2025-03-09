import argparse
from pathlib import Path
from .processor import ImageProcessor

def main():
    parser = argparse.ArgumentParser(
        description="Select and retain only the sharpest frames from a video or folder of images."
    )
    parser.add_argument('--input', required=True, help="Path to the input video or folder of images")
    parser.add_argument('--output', help="Directory to save the preserved images. Will modify in-place if not specified.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--count', type=int, help="Target number of images to retain")
    group.add_argument('--percentage', type=float, help="Percentage of images to retain (0-100)")
    parser.add_argument('--groups', type=int, help="Number of groups for distribution. Will apply percentage/count per group")
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
            # For video input, just extract frames if no count/percentage specified
            if not (args.count or args.percentage):
                processor.process_video(args.input, args.output)
                print(f"Successfully extracted frames to {args.output}")
                return 0
            else:
                processor.process_video(args.input, args.output)
                input_path = args.output
        else:
            input_path = args.input
            if not (args.count or args.percentage):
                raise ValueError("Either --count or --percentage must be specified for image folder processing")
            
        # Validate percentage if provided
        if args.percentage is not None:
            if not 0 <= args.percentage <= 100:
                raise ValueError("Percentage must be between 0 and 100")
            
        selected = processor.select_sharp_images(
            input_path=input_path,
            output_path=None if args.in_place else args.output,
            target_count=args.count,
            target_percentage=args.percentage if args.percentage is not None else None,
            groups=args.groups
        )
        print(f"Successfully processed {len(selected)} images")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 130  # Standard Unix practice for Ctrl+C
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main()
