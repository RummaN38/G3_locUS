from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Smoke-test SEKAI_Real_Walking_Dataset")
    parser.add_argument(
        "--csv",
        default=str(root / "data" / "sekai-real-walking.csv"),
        help="Path to sekai-real-walking CSV",
    )
    parser.add_argument(
        "--clips",
        default=str(root / "data" / "clips"),
        help="Directory containing clip MP4s (and optionally yamnet/yamnet.h5)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index for __getitem__",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    clips_dir = Path(args.clips).expanduser().resolve()

    if not csv_path.is_file():
        print(f"Missing CSV: {csv_path}", file=sys.stderr)
        return 1
    if not clips_dir.is_dir():
        print(f"Missing clips directory: {clips_dir}", file=sys.stderr)
        return 1

    os.chdir(root)

    sys.path.insert(0, str(root))
    from sekai_dataset import SEKAI_Real_Walking_Dataset

    mp4s = sorted(clips_dir.glob("*.mp4"))
    print(f"CSV: {csv_path}")
    print(f"Clips dir: {clips_dir} ({len(mp4s)} .mp4 files)")

    ds = SEKAI_Real_Walking_Dataset(csv_path=str(csv_path), features_path=str(clips_dir))

    n_csv = len(ds.csv)
    n_ds = len(ds)
    print(f"dataset.__len__ = {n_ds}  |  CSV rows = {n_csv}")
    if n_ds != n_csv:
        print(
            "Warning: __len__ counts .mp4 files in clips dir, not CSV rows; "
            "indices can misalign if these differ.",
            file=sys.stderr,
        )

    idx = args.index
    if idx < 0 or idx >= n_ds:
        print(f"Index {idx} out of range [0, {n_ds})", file=sys.stderr)
        return 1

    text, video, video_mask, lat, lon = ds[idx]
    print(f"__getitem__({idx}):")
    print(f"  text: {text[:120]!r}{'...' if len(text) > 120 else ''}")
    print(f"  video: shape={video.shape} dtype={video.dtype}")
    print(f"  video_mask: shape={video_mask.shape} dtype={video_mask.dtype}")
    print(f"  lat, lon: {lat}, {lon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
