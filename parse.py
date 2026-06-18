import argparse
import datetime
import logging
import os
import pathlib
import tempfile
import subprocess
import shutil
import json
import cv2
import numpy as np

def dir_path(path: str) -> pathlib.Path:
    """Ensure argument is a directory

    Args:
        path (str): The string that should represent a path

    Returns:
        pathlib.Path: The resolved directory path

    Raises:
        argparse.ArgumentTypeError: This will be raised if the passed argument is not a valid directory
    """
    check_path = pathlib.Path(path)

    if check_path.is_dir():
        return check_path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def parse_args() -> argparse.Namespace:
    """Helper function to define and parse arguments

    Returns:
        argparse.Namespace: A namespace with all of the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract set of videos from Wyze cameras based on time filter"
    )

    parser.add_argument(
        '-d', '--source-dir',
        help="The root of the Wyze Camera's drive",
        type=dir_path,
        required=True
    )

    parser.add_argument(
        '-o', '--out-dir',
        help="The output directory.",
        type=dir_path,
        required=True
    )

    parser.add_argument(
        '-n', '--file-name',
        help="The output filename or prefix. Defaults to timestamp range or merged_timestamp.mkv.",
        type=str
    )

    parser.add_argument(
        '-v', '--verbose',
        help="Increase verbosity. Can be used more than once.",
        action='count', default=0
    )

    parser.add_argument(
        '-b', '--begin',
        help="The lower end of the time period to search for logs, in ISO 8601 format (optional)",
        type=datetime.datetime.fromisoformat,
        default=None
    )

    parser.add_argument(
        '-e', '--end',
        type=datetime.datetime.fromisoformat,
        help="The upper end of the time period to search for logs, in ISO 8601 format (optional). Defaults to now if begin is specified.",
        default=None
    )

    # Motion detection arguments
    parser.add_argument(
        '--motion-only',
        help="Only extract segments of video with significant motion",
        action='store_true'
    )

    parser.add_argument(
        '--motion-threshold',
        help="Percent threshold of changed pixels between frames to register motion (default: 0.5)",
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--min-motion-duration',
        help="Minimum duration in seconds of a motion segment to keep (default: 1.0)",
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--min-static-duration',
        help="Maximum gap in seconds between motion segments to consider them contiguous (default: 65.0)",
        type=float,
        default=65.0
    )

    parser.add_argument(
        '--frame-skip',
        help="Process every N-th frame to speed up analysis (default: 5)",
        type=int,
        default=5
    )

    parser.add_argument(
        '--crop-bottom',
        help="Fraction of frame height at the bottom to crop out (e.g. ignoring timestamps) (default: 0.1)",
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--buffer-local',
        help="Buffer segments to local temp storage to optimize SD card reads (default: False)",
        action='store_true'
    )

    parser.add_argument(
        '--history-file',
        help="Path to the history file to avoid rescanning (default: .wyze_fetch_history.json in output directory)",
        type=str
    )

    return parser.parse_args()


def setup_logger(verbosity: int):
    """Configure the logger

    Args:
        verbosity (int): An integer to define how much to increase verbosity. Each step is an additional level of verbosity, maxing at debug.
    """
    level = max([logging.WARNING - verbosity*10, 0])
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def enumerate_video_segments(begin: datetime.datetime | None, end: datetime.datetime | None, path: pathlib.Path) -> list[pathlib.Path]:
    """Given a path, return all video segments within the specified time range

    Args:
        begin (datetime.datetime | None): A datetime object to represent the beginning of the period
        end (datetime.datetime | None): A datetime object to represent the end of the period
        path (pathlib.Path): A path-like object of where to look

    Returns:
        list[pathlib.Path]: A list of path-like objects corresponding to video segments within the range
    """
    all_segments = path.rglob('*.mp4')
    matching_segments: set[pathlib.Path] = set()
    for segment in all_segments:
        if begin is not None or end is not None:
            try:
                # Expecting path structure like .../YYYYMMDD/HH/MM.mp4
                segment_date = datetime.datetime.combine(
                    datetime.date.fromisoformat(segment.parts[-3]),
                    datetime.time(
                        int(segment.parts[-2]),
                        int(segment.stem),
                        0
                    )
                )
            except (IndexError, ValueError) as e:
                # Skip the segment if we are filtering by date but cannot parse the date from its path
                logging.warning(f"Skipping segment {segment} - could not parse date from path: {e}")
                continue

            if begin is not None and segment_date < begin:
                continue
            if end is not None and segment_date > end:
                continue

        logging.debug(f"Adding {segment}")
        matching_segments.add(segment)
        
    logging.info(f"Found {len(matching_segments)} segments")

    return matching_segments


def check_video_has_motion(
    video_path: pathlib.Path,
    threshold_percent: float,
    min_motion_duration: float,
    frame_skip: int,
    crop_bottom: float
) -> bool:
    """Scan a video file for motion and return True as soon as a significant 
    motion segment is found.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Could not open video file {video_path} for motion check")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0  # fallback

    motion_start_frame = None
    prev_gray = None
    frame_idx = 0
    has_motion = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Crop bottom of the frame (e.g. to ignore timestamp overlays)
            if crop_bottom > 0:
                h = gray.shape[0]
                crop_h = int(h * (1.0 - crop_bottom))
                if crop_h > 0:
                    gray = gray[:crop_h, :]

            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is not None:
                delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                non_zero = cv2.countNonZero(thresh)
                total_pixels = gray.shape[0] * gray.shape[1]
                percent = (non_zero / total_pixels) * 100.0

                if percent >= threshold_percent:
                    if motion_start_frame is None:
                        motion_start_frame = frame_idx
                    else:
                        dur = (frame_idx - motion_start_frame) / fps
                        if dur >= min_motion_duration:
                            has_motion = True
                            break
                else:
                    # Reset motion sequence
                    motion_start_frame = None
            
            prev_gray = gray

        frame_idx += 1

    cap.release()
    return has_motion


def get_segment_timestamp(segment: pathlib.Path) -> datetime.datetime:
    """Extract a datetime timestamp for a video segment.
    Attempts to parse Wyze directory structure first, otherwise falls back to file modification time.
    """
    try:
        # Expecting path structure like .../YYYYMMDD/HH/MM.mp4
        return datetime.datetime.combine(
            datetime.date.fromisoformat(segment.parts[-3]),
            datetime.time(
                int(segment.parts[-2]),
                int(segment.stem),
                0
            )
        )
    except (IndexError, ValueError):
        # Fallback to modification time
        mtime = segment.stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime)


def get_unique_path(base_path: pathlib.Path) -> pathlib.Path:
    """Resolve naming collisions by appending a numeric counter.
    """
    if not base_path.exists():
        return base_path
    suffix = base_path.suffix
    parent = base_path.parent
    stem = base_path.stem
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def make_group_filename(start_dt: datetime.datetime, end_dt: datetime.datetime, is_single: bool, file_name_arg: str | None) -> str:
    """Construct a descriptive filename for a video group.
    """
    start_str = start_dt.strftime("%Y%m%d_%H%M%S")
    prefix = "motion"
    
    if file_name_arg:
        prefix = pathlib.Path(file_name_arg).stem

    if is_single:
        return f"{prefix}_{start_str}.mkv"
    else:
        if start_dt.date() == end_dt.date():
            return f"{prefix}_{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%H%M')}.mkv"
        else:
            return f"{prefix}_{start_str}_to_{end_dt.strftime('%Y%m%d_%H%M%S')}.mkv"


def load_history(history_path: pathlib.Path) -> dict:
    """Load processed history log to avoid duplicate scanning.
    """
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load history file {history_path}: {e}")
    return {"processed_files": {}}


def save_history(history_path: pathlib.Path, history: dict):
    """Save processed history log to file.
    """
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save history file {history_path}: {e}")


def main():
    """Do the needful
    """
    args = parse_args()
    setup_logger(args.verbose)

    # If begin is specified but end is not, default end to now.
    begin = args.begin
    end = args.end
    if begin is not None and end is None:
        end = datetime.datetime.now()

    segments = enumerate_video_segments(begin, end, args.source_dir)

    if not segments:
        logging.warning("No video segments found matching criteria.")
        return

    sorted_segments = sorted(segments)

    if not args.motion_only:
        # Normal behavior (no motion filtering)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as fh:
            for line in sorted_segments:
                fh.write(f"file '{line.resolve()}'\n".encode())
        
        try:
            if args.file_name:
                outfile = args.file_name
            else:
                if begin is not None and end is not None:
                    outfile = f"{begin.isoformat(timespec='minutes')}-{end.isoformat(timespec='minutes')}.mkv"
                else:
                    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    outfile = f"merged_{now_str}.mkv"

            out_path = pathlib.Path(args.out_dir) / outfile
            logging.info(f"Beginning file merge into {out_path}...")
            merge_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', fh.name, '-c', 'copy', str(out_path)]
            result = subprocess.run(merge_cmd, capture_output=True)
            if result.returncode != 0:
                logging.error(f"Failed to run ffmpeg merge: {result.stderr.decode()} {result.stdout.decode()}")
            else:
                logging.info("Complete")
        finally:
            pathlib.Path(fh.name).unlink(missing_ok=True)
        return

    # Load history database
    if args.history_file:
        history_path = pathlib.Path(args.history_file)
    else:
        history_path = args.out_dir / ".wyze_fetch_history.json"
    history = load_history(history_path)
    processed_dict = history.setdefault("processed_files", {})

    # State variables for grouping and clean up
    current_group = []
    temp_files_to_clean = []

    def merge_current_group():
        nonlocal current_group
        if not current_group:
            return
        
        start_dt = current_group[0][1]
        end_dt = current_group[-1][1]
        filename = make_group_filename(start_dt, end_dt, len(current_group) == 1, args.file_name)
        out_path = get_unique_path(pathlib.Path(args.out_dir) / filename)

        logging.info(f"Merging group ({len(current_group)} files) into {out_path}...")
        
        # Write temporary concat list inside system temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as fh:
            for clip_path, clip_dt in current_group:
                fh.write(f"file '{clip_path.resolve()}'\n".encode())

        try:
            merge_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', fh.name, '-c', 'copy', str(out_path)]
            result = subprocess.run(merge_cmd, capture_output=True)
            if result.returncode != 0:
                logging.error(f"Failed to run ffmpeg merge: {result.stderr.decode()} {result.stdout.decode()}")
            else:
                logging.info(f"  Group merge complete: {out_path.name}")
        finally:
            pathlib.Path(fh.name).unlink(missing_ok=True)
            # Clean up buffered files from system temp immediately after merge
            if args.buffer_local:
                for temp_path, dt in current_group:
                    temp_path.unlink(missing_ok=True)
                    if temp_path in temp_files_to_clean:
                        temp_files_to_clean.remove(temp_path)
            
            # Reset current group
            current_group = []

    # Processing loop with signal interruption support
    try:
        for segment in sorted_segments:
            dt = get_segment_timestamp(segment)
            segment_key = str(segment.resolve())
            local_path_created = None

            # Check history cache
            if segment_key in processed_dict:
                has_motion = processed_dict[segment_key]["has_motion"]
                logging.info(f"Using history for {segment.name} (has_motion={has_motion})")
            else:
                # Need to scan this segment
                if args.buffer_local:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as local_temp:
                        local_temp_path = pathlib.Path(local_temp.name)
                        temp_files_to_clean.append(local_temp_path)
                        local_path_created = local_temp_path
                    
                    try:
                        logging.info(f"Buffering {segment.name} locally...")
                        shutil.copy(segment, local_temp_path)
                        
                        has_motion = check_video_has_motion(
                            video_path=local_temp_path,
                            threshold_percent=args.motion_threshold,
                            min_motion_duration=args.min_motion_duration,
                            frame_skip=args.frame_skip,
                            crop_bottom=args.crop_bottom
                        )
                    except Exception as e:
                        logging.error(f"Error scanning {segment}: {e}")
                        has_motion = False
                        local_temp_path.unlink(missing_ok=True)
                        if local_temp_path in temp_files_to_clean:
                            temp_files_to_clean.remove(local_temp_path)
                        continue
                else:
                    # Direct scanning
                    try:
                        logging.info(f"Scanning {segment.name} directly...")
                        has_motion = check_video_has_motion(
                            video_path=segment,
                            threshold_percent=args.motion_threshold,
                            min_motion_duration=args.min_motion_duration,
                            frame_skip=args.frame_skip,
                            crop_bottom=args.crop_bottom
                        )
                    except Exception as e:
                        logging.error(f"Error scanning {segment}: {e}")
                        continue

                # Record in history cache and save state
                processed_dict[segment_key] = {
                    "has_motion": has_motion,
                    "scanned_at": datetime.datetime.now().isoformat()
                }
                save_history(history_path, history)

            if has_motion:
                # Resolve the path to use for the concatenation step
                if args.buffer_local:
                    if local_path_created is None:
                        # Copy locally now since it was bypassed by using history
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as local_temp:
                            local_temp_path = pathlib.Path(local_temp.name)
                            temp_files_to_clean.append(local_temp_path)
                            local_path_created = local_temp_path
                        logging.info(f"Buffering motion segment {segment.name} locally for merge...")
                        shutil.copy(segment, local_temp_path)
                    
                    path_to_use = local_path_created
                else:
                    path_to_use = segment

                # Check if this clip belongs to the current contiguous block
                if current_group:
                    prev_path, prev_dt = current_group[-1]
                    gap = (dt - prev_dt).total_seconds()
                    if gap > args.min_static_duration:
                        # Gap detected! Merge the current group immediately.
                        merge_current_group()

                current_group.append((path_to_use, dt))
            else:
                # Segment has no motion. Clean up the temp file if created
                if local_path_created is not None:
                    local_path_created.unlink(missing_ok=True)
                    if local_path_created in temp_files_to_clean:
                        temp_files_to_clean.remove(local_path_created)

        # End of loop: merge any remaining active group
        merge_current_group()

    except (KeyboardInterrupt, SystemExit):
        logging.warning("Script interrupted. Performing graceful shutdown and merging completed segments...")
        save_history(history_path, history)
        if current_group:
            try:
                merge_current_group()
            except Exception as e:
                logging.error(f"Failed to merge remaining group during shutdown: {e}")
        raise
    finally:
        # Clean up any remaining temp files in case of errors/interruption
        for path in temp_files_to_clean:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()