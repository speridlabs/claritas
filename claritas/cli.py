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

    resize_group = parser.add_argument_group('Resize options')
    resize_group.add_argument('--resize', type=int, nargs='?', const=True, 
                            help="Resize images before processing. Can be used as flag or with a value representing max dimension size")
    size_group = resize_group.add_mutually_exclusive_group()
    size_group.add_argument('--max-width', type=int, help="Target width for resizing (height will be calculated to maintain aspect ratio)")
    size_group.add_argument('--max-height', type=int, help="Target height for resizing (width will be calculated to maintain aspect ratio)")
    
    args = parser.parse_args()

    
    processor = ImageProcessor(
        workers=args.workers,
        show_progress=not args.no_progress,
        use_cache=not args.no_cache
    )

    if args.input == args.output:
        raise ValueError("Input and output paths cannot be the same")
    
    try:
        input_path = args.input

        if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            if not args.output:
                raise ValueError("Output directory required for video processing")

            processor.process_video(args.input, args.output)
            print(f"Successfully extracted frames to {args.output}")

            if not (args.count or args.percentage or args.resize):
                return 0

            input_path = args.output

        if args.resize:
            width = args.max_width
            height = args.max_height
            max_size = args.resize if isinstance(args.resize, int) else None
            
            if not any([width, height, max_size]):
                raise ValueError("For resizing, either provide a number with --resize or use --max-width/--max-height")
            
            # if comming from video, output of video is input that is output
            resize_output = args.output if args.output != input_path else None
            
            print(f"Resizing images in {input_path}...")
            resized_images = processor.resize_images(
                input_path=input_path,
                output_path=resize_output,
                width=width,
                height=height,
                max_size=max_size
            )
            
            # If we resized to a new directory, update input_path for next steps
            if resize_output:
                input_path = Path(resize_output)
                print(f"Resized {len(resized_images)} images to {input_path}")

            if not (args.count or args.percentage):
                return 0
        
        if not (args.count or args.percentage):
                raise ValueError("Either --count or --percentage must be specified for image folder processing")

        if args.percentage is not None:
            if not 0 <= args.percentage <= 100:
                raise ValueError("Percentage must be between 0 and 100")

        # if coming from video, output of video is input that is output
        sharp_output = args.output if args.output != input_path else None

        selected = processor.select_sharp_images(
            input_path=input_path,
            output_path=sharp_output,
            target_count=args.count,
            target_percentage=args.percentage,
            groups=args.groups
        )

        print(f"Successfully processed {len(selected)} images")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main()
