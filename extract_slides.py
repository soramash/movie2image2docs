#!/usr/bin/env python3
"""
Extract slide images from a video by detecting slide changes using ffmpeg's scene detection.

Requirements:
  - ffmpeg must be installed and available in PATH

Usage examples:
  # Basic usage (vfr mode: recommended)
  python3 extract_slides.py input.mov

  # Customize threshold and spacing between detections (seek mode only uses --min-gap)
  python3 extract_slides.py input.mov --threshold 0.30 --min-gap 1.0 --mode seek

  # Save to a specific directory with a custom filename pattern and scale to max width 1920
  python3 extract_slides.py input.mov --out slides --pattern slide-%03d.png --max-width 1920

  # Force vfr mode
  python3 extract_slides.py input.mov --mode vfr --threshold 0.08 --include-first

Notes:
  - threshold: 0..1 (higher means fewer detections; slides usually work well around 0.25-0.40)
  - min-gap: minimum seconds between saved frames to avoid duplicates caused by transitions
  - pattern: must include a printf-style integer placeholder like %03d (default: slide-%03d.png)
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def ensure_ffmpeg() -> str:
    """Ensure ffmpeg is available and return its path."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit(
            "Error: ffmpeg is not installed or not found in PATH.\n"
            "Install on macOS with: brew install ffmpeg"
        )
    return ffmpeg


def detect_scene_times(video: Path, threshold: float) -> List[float]:
    """Use ffmpeg select+showinfo to detect scene-change times (in seconds)."""
    # Build filter chain: select frames with scene score over threshold, then show info
    # Quote the expression so commas are not parsed as filter separators
    vf = f"select='gt(scene,{threshold})',showinfo"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        vf,
        "-f",
        "null",
        "-",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    times: List[float] = []
    # showinfo outputs lines containing: "pts_time:123.456"
    # Some ffmpeg builds print a space after 'pts_time:'; allow optional whitespace
    pts_re = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")
    for line in proc.stderr.splitlines():
        m = pts_re.search(line)
        if m:
            try:
                times.append(float(m.group(1)))
            except ValueError:
                pass

    # Deduplicate and sort
    times = sorted(set(times))
    return times


def filter_by_min_gap(times: List[float], min_gap: float) -> List[float]:
    if min_gap <= 0:
        return times
    filtered: List[float] = []
    last = -1e9
    for t in times:
        if t - last >= min_gap:
            filtered.append(t)
            last = t
    return filtered


def extract_frame(video: Path, t: float, out_path: Path, max_width: int | None) -> None:
    # Use accurate seek by placing -ss before -i and relying on decoding
    # -y to overwrite without prompt
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
    ]
    if max_width:
        # Keep aspect ratio by setting height to -2 (even). Limit width to max_width.
        cmd += ["-vf", f"scale=min({max_width},iw):-2"]
    cmd += [str(out_path)]

    subprocess.run(cmd, check=True)


def extract_vfr(video: Path, out_pattern_path: Path, threshold: float, max_width: int | None, include_first: bool) -> None:
    # Direct ffmpeg extraction using select(scene) and vfr output.
    expr = f"eq(n,0)+gt(scene,{threshold})" if include_first else f"gt(scene,{threshold})"
    vf_filters = [f"select='{expr}'"]
    if max_width:
        vf_filters.append(f"scale=min({max_width},iw):-2")
    vf = ",".join(vf_filters)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        str(out_pattern_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract one image per slide from a video using ffmpeg scene detection.")
    p.add_argument("video", help="Path to the input video (e.g., .mov)")
    p.add_argument("--out", default="slides", help="Output directory (default: slides)")
    p.add_argument("--threshold", type=float, default=0.30, help="Scene threshold 0..1 (default: 0.30)")
    p.add_argument("--min-gap", type=float, default=1.0, help="Minimum seconds between saved frames (seek mode only; default: 1.0)")
    p.add_argument("--mode", choices=["seek", "vfr"], default="vfr", help="Extraction mode: 'vfr' uses ffmpeg select+vfr (robust), 'seek' detects times then seeks (default: vfr)")
    try:
        from argparse import BooleanOptionalAction  # Python 3.9+
        p.add_argument("--include-first", action=BooleanOptionalAction, default=True,
                       help="Include the very first frame (n=0) as a slide (default: enabled)")
    except Exception:
        # Fallback for very old Python: enable via presence of flag only
        p.add_argument("--include-first", action="store_true", default=True,
                       help="Include the very first frame (n=0) as a slide (default: enabled)")
    p.add_argument(
        "--pattern",
        default="slide-%03d.png",
        help="Filename pattern with printf-style integer (default: slide-%03d.png)",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for filenames (default: 1)",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Optional max width to scale output images; keeps aspect ratio",
    )

    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"Input video not found: {video}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_ffmpeg()

    # Prepare filename pattern
    pattern = args.pattern
    # Accept %d or %0Nd patterns (e.g., %03d)
    if not re.search(r"%\d*d", pattern):
        print("Warning: --pattern lacks %d placeholder. Using default slide-%03d.png")
        pattern = "slide-%03d.png"

    if args.mode == "vfr":
        out_pattern_path = out_dir / pattern
        print(f"Extracting with ffmpeg select+vfr (threshold={args.threshold}, include_first={args.include_first}) -> {out_dir}")
        extract_vfr(video, out_pattern_path, args.threshold, args.max_width, args.include_first)
        # Count saved files (replace printf-style %d or %0Nd with *)
        count_glob = re.sub(r"%\d*d", "*", pattern)
        saved = len(list(out_dir.glob(count_glob)))
        print(f"Done. Saved approximately {saved} slides to {out_dir}")
        return

    # 'seek' mode: detect times then extract frames one by one
    print(f"Detecting slide changes in: {video}")
    times = detect_scene_times(video, args.threshold)
    # Optionally include the very first frame
    if args.include_first:
        if not times or times[0] > 0.0:
            times = [0.0] + times
    if not times:
        print("No scene changes detected. Capturing the first frame only.")
        times = [0.0]

    times = filter_by_min_gap(times, args.min_gap)

    count = 0
    idx = args.start_index
    for t in times:
        filename = pattern % idx
        out_path = out_dir / filename
        extract_frame(video, t, out_path, args.max_width)
        print(f"Saved {out_path} at t={t:.2f}s")
        idx += 1
        count += 1

    print(f"Done. Saved {count} slides to {out_dir}")


if __name__ == "__main__":
    main()
