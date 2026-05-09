"""
SEKAI dataset + full clip pipeline (ffmpeg audio, optional JPEG frames, YAMNet top-3, geocode).

YAMNet weights live in repo `video_processor/`. Requires: ffmpeg, tensorflow, soundfile, opencv,
pandas; geopy optional for lat/lon.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

from rawvideo_util import RawVideoExtractor

# --- repo paths: YAMNet code + weights in ../video_processor ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
_VIDEO_PROC = _REPO_ROOT / "video_processor"
if str(_VIDEO_PROC) not in sys.path:
    sys.path.insert(0, str(_VIDEO_PROC))

import params as yamnet_params  # noqa: E402
import yamnet as yamnet_model  # noqa: E402


def _ffmpeg_bin() -> str:
    local = _VIDEO_PROC / "ffmpeg"
    return str(local) if local.is_file() else "ffmpeg"


def extract_audio_from_video(video_path: str, out_wav: str) -> None:
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        out_wav,
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed extracting audio. Install ffmpeg and ensure the video has an audio track."
        )


def extract_frames_uniform(video_path: str, frames_dir: Path, max_frames: int) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return []
    take = min(max_frames, n)
    indices = np.linspace(0, n - 1, num=take, dtype=int)
    saved: list[str] = []
    fi = 0
    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = frames_dir / f"frame_{fi:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(str(out_path.resolve()))
        fi += 1
    cap.release()
    return saved


_yamnet_bundle: tuple | None = None
_yamnet_bundle_hop: float | None = None


def _get_yamnet_bundle(patch_hop_seconds: float = 0.48):
    global _yamnet_bundle, _yamnet_bundle_hop
    params = yamnet_params.Params(sample_rate=16000.0, patch_hop_seconds=patch_hop_seconds)
    class_map = _VIDEO_PROC / "yamnet_class_map.csv"
    weights = _VIDEO_PROC / "yamnet.h5"
    if not class_map.is_file() or not weights.is_file():
        raise FileNotFoundError(
            f"Missing YAMNet assets under {_VIDEO_PROC} (yamnet_class_map.csv, yamnet.h5)."
        )
    if _yamnet_bundle is None or _yamnet_bundle_hop != patch_hop_seconds:
        class_names = yamnet_model.class_names(str(class_map))
        model = yamnet_model.yamnet_frames_model(params)
        model.load_weights(str(weights))
        _yamnet_bundle = (model, class_names, params)
        _yamnet_bundle_hop = patch_hop_seconds
    return _yamnet_bundle


def yamnet_top3_mean(
    wav_path: str, patch_hop_seconds: float = 0.48
) -> tuple[list[dict[str, float]], list[str]]:
    model, class_names, _params = _get_yamnet_bundle(patch_hop_seconds)
    wav_data, _sr = sf.read(wav_path, dtype=np.int16)
    waveform = (wav_data / 32768.0).astype(np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    scores, _emb, _spec = model(waveform)
    scores = scores.numpy()
    mean_scores = np.mean(scores, axis=0)
    top_idx = np.argsort(mean_scores)[::-1][:3]
    ranked: list[dict[str, float]] = []
    names: list[str] = []
    for i in top_idx:
        ranked.append({"class": class_names[i], "score": float(mean_scores[i])})
        names.append(class_names[i])
    return ranked, names


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def geocode_location(address: str, user_agent: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        return None, None, "geopy not installed (pip install geopy)"
    geolocator = Nominatim(user_agent=user_agent)
    try:
        loc = geolocator.geocode(address, timeout=15)
    except Exception as e:
        return None, None, str(e)
    if loc is None:
        return None, None, "no geocoding result"
    return float(loc.latitude), float(loc.longitude), None


def load_csv_row(csv_path: Path, video_path: Path, location_override: Optional[str] = None):
    df = pd.read_csv(csv_path)
    base = video_path.name
    col = "videoFile" if "videoFile" in df.columns else None
    if col is None:
        row = df.iloc[0]
    else:
        m = df[df[col].astype(str) == base]
        if len(m) == 0:
            m = df[df[col].astype(str).str.contains(base, regex=False, na=False)]
        if len(m) == 0:
            if location_override:
                row_dict = {
                    "videoFile": base,
                    "location": location_override,
                    "_note": "No CSV row matched filename; location from --location",
                }
                return _json_safe(row_dict), location_override.strip()
            raise ValueError(
                f"No CSV row with videoFile matching {base!r}. "
                f"Rename the file, fix the CSV, or pass --location \"...\" . Columns: {list(df.columns)}"
            )
        row = m.iloc[0]
    loc_col = "location" if "location" in row.index else None
    if loc_col is None:
        raise ValueError(f"CSV must have a 'location' column. Got: {list(row.index)}")
    location_text = str(row[loc_col]).strip()
    row_dict = _json_safe(row.where(pd.notnull(row), None).to_dict())
    return row_dict, location_text


def run_clip_pipeline(
    video_path: Path,
    csv_row: dict,
    location_text: str,
    work_dir: Path,
    csv_file: Optional[str] = None,
    max_frames: int = 24,
    yamnet_hop: float = 0.48,
    geocode_user_agent: str = "LocUsVideoPipeline/1.0",
    geocode_cache: Optional[dict] = None,
    run_geocode: bool = True,
) -> dict:
    video_path = Path(video_path).resolve()
    stem = video_path.stem
    work_dir = Path(work_dir)
    audio_dir = work_dir / "audio"
    frames_dir = work_dir / "frames"
    audio_dir.mkdir(parents=True, exist_ok=True)

    wav_path = audio_dir / f"{stem}.wav"
    extract_audio_from_video(str(video_path), str(wav_path))
    frame_paths = extract_frames_uniform(str(video_path), frames_dir, max_frames)
    top3_detail, top3_names = yamnet_top3_mean(str(wav_path), patch_hop_seconds=yamnet_hop)

    if run_geocode:
        if geocode_cache is not None and location_text in geocode_cache:
            lat, lon, geo_err = geocode_cache[location_text]
        else:
            lat, lon, geo_err = geocode_location(location_text, geocode_user_agent)
            if geocode_cache is not None:
                geocode_cache[location_text] = (lat, lon, geo_err)
    else:
        lat, lon, geo_err = None, None, "geocoding disabled"

    combined_text = f"{location_text}. Audio: {top3_names[0]}, {top3_names[1]}, {top3_names[2]}."

    meta = {
        "video_file": str(video_path),
        "csv_file": csv_file,
        "csv_row": csv_row,
        "location_text": location_text,
        "latitude": lat,
        "longitude": lon,
        "geocode_error": geo_err,
        "combined_text": combined_text,
        "audio_top3": top3_detail,
        "audio_wav": str(wav_path.resolve()),
        "frame_paths": frame_paths,
        "frame_count_saved": len(frame_paths),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    meta_path = work_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


class SEKAI_Real_Walking_DataLoader(Dataset):
    """
    CSV columns: videoFile, location. Clips under features_path.

    Caches pipeline output under features_path/.sekai_cache/<stem>/ (metadata.json, audio/, frames/).
    Call prewarm_metadata_cache() before DataLoader(num_workers>0).

    __getitem__: text, video, video_mask  OR  + latitude, longitude if return_location_coords.
    """

    def __init__(
        self,
        csv_path="sekai-real-walking.csv",
        features_path="./data/sekai-real-walking",
        metadata_cache_dir=None,
        feature_framerate=1.0,
        max_frames=100,
        image_resolution=224,
        frame_order=0,
        slice_framepos=0,
        use_pipeline_text=True,
        pipeline_max_saved_frames=24,
        yamnet_hop=0.48,
        run_geocoding=True,
        geocode_user_agent="SEKAI_Real_Walking_DataLoader/1.0",
        refresh_metadata=False,
        return_location_coords=False,
    ):
        self.csv_file_path = os.path.abspath(str(csv_path))
        self.csv = pd.read_csv(csv_path)
        if "videoFile" not in self.csv.columns or "location" not in self.csv.columns:
            raise ValueError("CSV must include columns 'videoFile' and 'location'.")

        self.features_path = os.path.abspath(str(features_path))
        self.sample_len = len(self.csv)

        self.metadata_cache_dir = metadata_cache_dir or os.path.join(
            self.features_path, ".sekai_cache"
        )
        os.makedirs(self.metadata_cache_dir, exist_ok=True)

        self.pipeline_max_saved_frames = pipeline_max_saved_frames
        self.yamnet_hop = yamnet_hop
        self.run_geocoding = run_geocoding
        self.geocode_user_agent = geocode_user_agent
        self.refresh_metadata = refresh_metadata
        self._geocode_cache = {}

        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )

    def __len__(self):
        return self.sample_len

    def _resolve_clip_path(self, video_file) -> Optional[str]:
        name = str(video_file).strip()
        direct = os.path.join(self.features_path, name)
        if os.path.isfile(direct):
            return os.path.abspath(direct)
        stem = Path(name).stem
        for ext in (".mp4", ".MP4", ".mov", ".MOV", ".webm", ".WEBM", ".mkv", ".MKV"):
            p = os.path.join(self.features_path, stem + ext)
            if os.path.isfile(p):
                return os.path.abspath(p)
        base_only = Path(name).name
        direct2 = os.path.join(self.features_path, base_only)
        if os.path.isfile(direct2):
            return os.path.abspath(direct2)
        return None

    def _ensure_metadata(self, row, video_path: str) -> dict:
        stem = Path(video_path).stem
        cache_sub = os.path.join(self.metadata_cache_dir, stem)
        meta_path = os.path.join(cache_sub, "metadata.json")
        if os.path.isfile(meta_path) and not self.refresh_metadata:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)

        csv_row = _json_safe(row.where(pd.notnull(row), None).to_dict())
        location_text = str(row["location"]).strip()
        return run_clip_pipeline(
            Path(video_path),
            csv_row,
            location_text,
            Path(cache_sub),
            csv_file=self.csv_file_path,
            max_frames=self.pipeline_max_saved_frames,
            yamnet_hop=self.yamnet_hop,
            geocode_user_agent=self.geocode_user_agent,
            geocode_cache=self._geocode_cache,
            run_geocode=self.run_geocoding,
        )

    def prewarm_metadata_cache(self):
        for idx in range(len(self)):
            row = self.csv.iloc[idx]
            path = self._resolve_clip_path(row["videoFile"])
            if path is None:
                raise FileNotFoundError(
                    f"Video not found for row {idx} videoFile={row['videoFile']!r} in {self.features_path}"
                )
            self._ensure_metadata(row, path)

    def _load_video(self, video_path: str):
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        video = np.zeros(
            (1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float,
        )
        if not os.path.isfile(video_path):
            return video, video_mask

        raw_video_data = self.rawVideoExtractor.get_video_data(video_path)["video"]
        if len(raw_video_data.shape) <= 3:
            return video, video_mask

        raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)
        if self.max_frames < raw_video_slice.shape[0]:
            if self.slice_framepos == 0:
                video_slice = raw_video_slice[: self.max_frames, ...]
            elif self.slice_framepos == 1:
                video_slice = raw_video_slice[-self.max_frames :, ...]
            else:
                idx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_slice[idx, ...]
        else:
            video_slice = raw_video_slice

        video_slice = self.rawVideoExtractor.process_frame_order(
            video_slice, frame_order=self.frame_order
        )
        n = video_slice.shape[0]
        if n >= 1:
            video[0][:n, ...] = video_slice
            video_mask[0][:n] = 1
        return video, video_mask

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        video_path = self._resolve_clip_path(row["videoFile"])
        if video_path is None:
            raise FileNotFoundError(
                f"No video file matching {row['videoFile']!r} under {self.features_path}"
            )
        meta = self._ensure_metadata(row, video_path)

        text = meta["combined_text"]
        video, video_mask = self._load_video(video_path)

        lat = meta.get("latitude")
        lon = meta.get("longitude")
        if lat is None or lon is None:
            raise ValueError(f"Latitude or longitude is None for video {video_path}")
        lat_f = float(lat)
        lon_f = float(lon)

        return text, video, video_mask, lat_f, lon_f
