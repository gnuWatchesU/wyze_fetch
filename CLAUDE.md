# wyze_fetch — orientation notes (2026-07-28)

Single-script Python CLI (`parse.py`, no package layout — `[tool.uv] package = false`) that extracts and merges video segments from a Wyze camera's SD card into consolidated MKV files, optionally filtering down to only the segments containing motion. Unrelated to the `fleet`/`k8s` Ohana home lab — personal utility project. GPLv3. Very early stage (5 commits total; latest `6c05e64 claudified`, 2026-07-27); no tests directory.

## How it works

1. **`enumerate_video_segments`** recursively globs `*.mp4` under the source dir, expecting Wyze's on-camera layout `.../YYYYMMDD/HH/MM.mp4`, filtered by optional `--begin`/`--end` ISO-8601 range.
2. **`get_segment_timestamp`** sorts segments chronologically, falling back to file mtime if a path doesn't match the expected structure.
3. Two modes:
   - **Default**: all matching segments go into an ffmpeg concat list, merged via `ffmpeg -f concat -c copy` into one output file.
   - **`--motion-only`**: each segment is opened with OpenCV (`check_video_has_motion`), scanned frame-by-frame (`--frame-skip`) for pixel-difference motion above `--motion-threshold`, ignoring frame-to-frame brightness swings above `--brightness-change-threshold` (avoids false positives from IR-cut switching, exposure jumps, clouds) and optionally cropping a bottom strip (`--crop-bottom`) to ignore timestamp overlays. Motion segments within `--min-static-duration` seconds of each other are grouped (`merge_current_group`) and merged separately per group, named by timestamp range (`make_group_filename`, collision-safe via `get_unique_path`).
4. **History/caching** (`--history-file`, default `.wyze_fetch_history.json` in the output dir) records per-file motion-scan results as JSON so repeated runs skip re-scanning.
5. **`--buffer-local`** copies segments to local temp storage before scanning/merging (helps on slow SD cards); cleaned up after use.
6. Handles `KeyboardInterrupt`/`SystemExit` gracefully — saves history and merges any in-progress group before re-raising.

## CLI

```
python parse.py -d <source-dir> -o <out-dir> [options]
```

Required: `-d/--source-dir`, `-o/--out-dir`. Notable flags: `-n/--file-name`, `-v/--verbose` (repeatable), `-b/--begin`, `-e/--end`, `--motion-only`, `--motion-threshold`, `--min-motion-duration`, `--min-static-duration`, `--frame-skip`, `--crop-bottom`, `--brightness-change-threshold`, `--buffer-local`, `--history-file`.

## Dependencies

- `opencv-python-headless>=4.8.0`, `numpy>=1.20.0` (managed via `uv`, `uv.lock`, requires Python >=3.11).
- External binary: `ffmpeg` (invoked via `subprocess`, not a Python dependency).
