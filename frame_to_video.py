"""
frames_to_video.py - Convert frame PNGs from a run into an MP4 video.

Usage:
    python frames_to_video.py                                    # latest run
    python frames_to_video.py --run runs/baseline_20260304_143022
    python frames_to_video.py --run runs/baseline_20260304_143022 --fps 4
    python frames_to_video.py --run runs/baseline_20260304_143022 --out my_video.mp4
"""

import os
import argparse
import glob
import subprocess


def find_latest_run():
    """Auto-pick the latest run folder."""
    runs_dir = "runs"
    if not os.path.isdir(runs_dir):
        return None
    folders = sorted([
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ])
    return folders[-1] if folders else None


def frames_to_video(frames_dir: str, out_path: str, fps: int = 3):
    """
    Read frame_NNNN.png files and write an MP4.
    Uses ffmpeg's native image sequence reader — no piping, no PIL.
    """
    pattern = os.path.join(frames_dir, "frame_*.png")
    frame_files = sorted(glob.glob(pattern))

    if not frame_files:
        print(f"  No frames found in {frames_dir}")
        return False

    print(f"  Found {len(frame_files)} frames")
    print(f"  FPS: {fps}  →  duration: {len(frame_files)/fps:.1f}s")

    # ffmpeg reads the sequence directly from disk: frame_%04d.png
    input_pattern = os.path.join(frames_dir, "frame_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vcodec", "mpeg4",
        "-pix_fmt", "yuv420p",
        "-q:v", "5",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        out_path,
    ]

    print(f"  Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Saved → {out_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ffmpeg error (code {result.returncode}):")
        err_lines = result.stderr.strip().split('\n')
        for line in err_lines[-5:]:
            print(f"    {line}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert frames to MP4 video")
    parser.add_argument("--run", type=str, default=None,
                        help="Path to run folder (default: latest)")
    parser.add_argument("--fps", type=int, default=3,
                        help="Frames per second (default: 3)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output MP4 path (default: <run>/episode.mp4)")
    args = parser.parse_args()

    if args.run:
        run_dir = args.run
    else:
        run_dir = find_latest_run()
        if run_dir:
            print(f"  Auto-selected: {run_dir}")
        else:
            print("  No runs found. Specify --run path.")
            exit(1)

    frames_dir = os.path.join(run_dir, "frames")
    if not os.path.isdir(frames_dir):
        print(f"  No frames/ folder in {run_dir}")
        exit(1)

    out_path = args.out or os.path.join(run_dir, "episode.mp4")

    print(f"  Run     : {run_dir}")
    print(f"  Output  : {out_path}")
    print()

    success = frames_to_video(frames_dir, out_path, fps=args.fps)
    if not success:
        print("\n  If ffmpeg is not installed: sudo apt install ffmpeg")