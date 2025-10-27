import argparse
from pathlib import Path
from .metadata import copy_video_metadata
from .processor import ImageProcessor, ColmapProcessor

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
    parser.add_argument('--only-copy-metadata', action='store_true', help="Only copy metadata from input video to output folder of images.")

    resize_group = parser.add_argument_group('Resize options')
    resize_group.add_argument('--resize', type=int, nargs='?', const=True,
                            help="Resize images before processing. Can be used as flag or with a value representing max dimension size")
    size_group = resize_group.add_mutually_exclusive_group()
    size_group.add_argument('--max-width', type=int, help="Target width for resizing (height will be calculated to maintain aspect ratio)")
    size_group.add_argument('--max-height', type=int, help="Target height for resizing (width will be calculated to maintain aspect ratio)")

    colmap_group = parser.add_argument_group('COLMAP pruning options')
    colmap_group.add_argument('--colmap-dir', help="Path to COLMAP reconstruction for camera pruning")

    args = parser.parse_args()


    processor = ImageProcessor(
        workers=args.workers,
        show_progress=not args.no_progress,
        use_cache=not args.no_cache
    )

    colmap_processor = ColmapProcessor(
        workers=args.workers,
        show_progress=not args.no_progress,
    )

    if args.input == args.output:
        raise ValueError("Input and output paths cannot be the same")

    try:
        input_path = args.input

        if args.colmap_dir:
            print(f"Pruning based on COLMAP reconstruction...")
            colmap_processor.prune_colmap(
                colmap_dir=args.colmap_dir,
                output_dir=args.output,
            )
            return 0

        if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            if not args.output:
                raise ValueError("Output directory required for video processing")

            if args.only_copy_metadata:
                video_path = Path(args.input)
                output_folder = Path(args.output)
                if not output_folder.is_dir():
                    raise ValueError("--output must be a directory of images when using --only-copy-metadata")

                extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                images = [p for p in output_folder.rglob('*') if p.suffix.lower() in extensions]

                if not images:
                    print("No images found in the output directory.")
                    return 0

                print(f"Copying metadata from video {video_path} to {len(images)} images in {output_folder}...")
                copy_video_metadata(
                    src_video=video_path,
                    dst_frames=images,
                    workers=args.workers,
                    show_progress=not args.no_progress
                )
                print("Successfully copied metadata.")
                return 0

            output_dir, num_frames, fps = processor.process_video(args.input, args.output)
            print(f"Successfully extracted frames to {args.output} (output_dir: {output_dir}, num_frames: {num_frames}, fps: {fps})")

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
