import argparse
import datetime
import logging
import os
import pathlib
import tempfile
import subprocess

def dir_path(path: str) -> None:
    """Ensure argument is a directory

    Args:
        path (str): The string that should represent a path

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
        help="The output filename.  Defaults to timestamp.",
        type=str
    )

    parser.add_argument(
        '-v', '--verbose',
        help="Increase verbosity.  Can be used more than once.",
        action='count', default=0
    )

    parser.add_argument(
        '-b', '--begin',
        help="The lower end of the time period to search for logs, in ISO 8601 format",
        type=datetime.datetime.fromisoformat,
        required=True
    )

    parser.add_argument(
        '-e', '--end',
        type=datetime.datetime.fromisoformat,
        help="The upper end of the time period to search for logs, in ISO 8601 format.  Defaults to now.",
        default=datetime.datetime.now()
    )

    return parser.parse_args()


def setup_logger(verbosity: int):
    """Configure the logger

    Args:
        verbosity (int): An integer to define how much to increase verbosity.  Each step is an additional level of verbosity, maxing at debug.
    """
    level = max([logging.WARNING - verbosity*10, 0])
    # log_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    # root_logger = logging.getLogger()
    # root_logger.setLevel(level)
    # root_logger.handlers[0].setFormatter(log_fmt)


def enumerate_video_segments(begin: datetime.datetime, end: datetime.datetime, path: pathlib.Path) -> list[pathlib.Path]:
    """Given a path, return all video segments within the specified time range

    Note, this is currently very fragile and depends on Wyze not changing their naming pattern.  If they do, this will fail spectacularly.

    Args:
        begin (datetime.datetime): A datetime object to represent the beginning of the period
        end (datetime.datetime): A datetime object to represent the end of the period
        path (pathlib.Path): A path-like object of where to look

    Returns:
        list[pathlib.Path]: A list of path-like objects corresponding to video segments within the range
    """
    all_segments = path.rglob('*.mp4')
    matching_segments: set[pathlib.Path] = set()
    for segment in all_segments:
        segment_date = datetime.datetime.combine(
            datetime.date.fromisoformat(segment.parts[-3]),
            datetime.time(
                int(segment.parts[-2]),
                int(segment.stem),
                0
                )
        )

        if begin <= segment_date <= end:
            logging.debug(f"Adding {segment}")
            matching_segments.add(segment)
        
    logging.info(f"Found {len(matching_segments)} segments")

    return matching_segments

def main():
    """Do the needful
    """
    args = parse_args()
    setup_logger(args.verbose)

    segments = enumerate_video_segments(args.begin, args.end, args.source_dir)

    fh = tempfile.NamedTemporaryFile(delete=False)
    for line in sorted(segments):
        fh.write(f"file '{line}'\n".encode())
    fh.close()

    outfile = args.file_name if args.file_name else f"{args.begin.isoformat(timespec='minutes')}-{args.end.isoformat(timespec='minutes')}.mkv"

    logging.info("Beginning file merge.")
    result = subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', fh.name, '-c', 'copy', pathlib.Path(args.out_dir, outfile)], capture_output=True)
    if result.returncode != 0:
        logging.error(f"Failed to run ffmpeg: {result.stderr} {result.stdout}")
    logging.info("Complete")

    os.unlink(fh.name)

if __name__ == "__main__":
    main()