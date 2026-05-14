import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset
from geopy.geocoders import Nominatim

from rawvideo_util import RawVideoExtractor
from yamnet import yamnet as yamnet_model, params as yamnet_params


def extract_audio_from_video(video_path: str, out_wav: str) -> None:
    cmd = [
        "ffmpeg", # change this to the path of the ffmpeg binary
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


def json_safe(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class SEKAI_Real_Walking_Dataset(Dataset):
    def __init__(
        self,
        csv_path="sekai-real-walking.csv",
        features_path="./data/clips",
        feature_framerate=1.0,
        max_frames=100,
        image_resolution=224,
        pipeline_max_saved_frames=24,
        save_frames=False,
        yamnet_hop=0.48,
        refresh_metadata=False
    ):
        self.csv_file_path = os.path.abspath(str(csv_path))
        self.csv = pd.read_csv(csv_path)

        self.features_path = os.path.abspath(str(features_path))
        self.sample_len = len([f for f in os.listdir(self.features_path) if f.endswith('.mp4')])

        self.metadata_cache_dir = "./data/sekai-real-walking.sekai_cache"
        os.makedirs(self.metadata_cache_dir, exist_ok=True)

        self.pipeline_max_saved_frames = pipeline_max_saved_frames
        self.refresh_metadata = refresh_metadata
        self.geocode_cache = {}
        self.save_frames = save_frames

        self.feature_framerate = feature_framerate
        self.max_frames = max_frames

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )

        self.yamnet_hop = yamnet_hop
        self.yamnet_model_path, self.yamnet_class_names_path, self.yamnet_params_path =  os.path.abspath(os.path.join(self.features_path, "yamnet/yamnet.h5")), os.path.abspath(os.path.join(self.features_path, "yamnet/yamnet_class_map.csv")), os.path.abspath(os.path.join(self.features_path, "yamnet/yamnet.h5"))
        self.yamnet_class_names = yamnet_model.class_names(self.yamnet_class_names_path)
        self.yamnet_model = yamnet_model.yamnet_frames_model(self.yamnet_params_path)
        self.yamnet_model.load_weights(self.yamnet_model_path)

    def __len__(self):
        return self.sample_len
    
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        video_path = os.path.join(self.features_path, row["videoFile"])
        if video_path is None:
            raise FileNotFoundError(
                f"No video file matching {row['videoFile']!r} under {self.features_path}"
            )
        meta, video, video_mask = self._ensure_metadata(row, video_path)

        text = meta["combined_text"]

        lat = float(meta.get("latitude"))
        lon = float(meta.get("longitude"))

        return text, video, video_mask, lat, lon

    def create_metadata_cache(self): # before training, create the metadata cache
        for idx in range(self.sample_len):
            row = self.csv.iloc[idx]
            path = os.path.join(self.features_path, row["videoFile"])
            if path is None:
                raise FileNotFoundError(
                    f"Video not found for row {idx} videoFile={row['videoFile']!r} in {self.features_path}"
                )
            self._ensure_metadata(row, path)

    def _ensure_metadata(self, row, video_path: str) -> tuple[dict, np.ndarray, np.ndarray]:
        stem = Path(video_path).stem
        cache_sub = os.path.join(self.metadata_cache_dir, stem)
        meta_path = os.path.join(cache_sub, "metadata.json")
        if os.path.isfile(meta_path) and not self.refresh_metadata:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                video, video_mask = self._load_video(video_path, save_frames=False, frames_dir=None)
                return meta, video, video_mask

        csv_row = json_safe(row.where(pd.notnull(row), None).to_dict())
        location_text = str(row["location"]).strip()
        return self._run_clip_pipeline(
            Path(video_path),
            csv_row,
            location_text,
            Path(cache_sub),
        )

    def _clip_unnormalized(self, frame_chw: np.ndarray) -> np.ndarray:
        x = np.asarray(frame_chw, dtype=np.float32)[0]
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
        x = x * std + mean
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
        x = np.transpose(x, (1, 2, 0))
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    def _load_video(
        self,
        video_path: str | Path,
        save_frames: bool,
        frames_dir: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros(
            (1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float,
        ) # [1, T, 1, 3, H, W]
        frame_paths: list[str] = []

        raw_video_data = self.rawVideoExtractor.get_video_data(video_path)["video"] 
        if len(raw_video_data.shape) <= 3:
            print(f"Video {video_path} has less than 3 dimensions")

        raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data) # [T, 1, 3, H, W]
        slice_framepos = 0
        if self.max_frames < raw_video_slice.shape[0]: # if more frames
            if slice_framepos == 0:
                video_slice = raw_video_slice[: self.max_frames, ...] # keep the first max_frames frames
            elif slice_framepos == 1:
                video_slice = raw_video_slice[-self.max_frames :, ...] # keep the last max_frames frames
            else:
                idx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_slice[idx, ...] # keep the frames at the random indices
        else:
            video_slice = raw_video_slice

        frame_order = 0
        video_slice = self.rawVideoExtractor.process_frame_order(
            video_slice, frame_order=frame_order
        ) # keep video in ordinary order
        n = video_slice.shape[0] # number of frames
        if n >= 1:
            vid_np = (
                video_slice.detach().cpu().numpy()
                if hasattr(video_slice, "detach")
                else np.asarray(video_slice, dtype=np.float32)
            )
            video[0][:n, ...] = vid_np
            video_mask[0][:n] = 1 # make the frames valid
            if save_frames and frames_dir is not None:
                stem = Path(video_path).stem
                out_sub = Path(frames_dir) / stem
                out_sub.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    bgr = self._clip_normalized_chw_to_bgr_uint8(vid_np[i])
                    fp = out_sub / f"{stem}_{i:05d}.png"
                    cv2.imwrite(str(fp), bgr)
                    frame_paths.append(str(fp.resolve()))
        return video, video_mask, frame_paths
    
    def _geocode_location(self, address: str) -> tuple[float, float]:
        geolocator = Nominatim(user_agent="LocUs")
        loc = geolocator.geocode(address, timeout=15)
        if loc is None:
            raise ValueError("No geocoding result")
        return float(loc.latitude), float(loc.longitude)
    
    def _yamnet_top3(self, wav_path: str) -> tuple[list[dict[str, float]], list[str]]:
        wav_data, _sr = sf.read(wav_path, dtype=np.int16)
        waveform = (wav_data / 32768.0).astype(np.float32)
        scores, _emb, _spec = self.yamnet_model(waveform)
        scores = scores.numpy()
        top_idx = np.argsort(scores)[::-1][:3]
        ranked, names = [], []
        for i in top_idx:
            ranked.append({"class": self.yamnet_class_names[i], "score": float(scores[i])})
            names.append(self.yamnet_class_names[i])
        return ranked, names

    def _run_clip_pipeline(
        self,
        video_path: Path,
        csv_row: dict,
        location_text: str,
        work_dir: Path
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        video_path = Path(video_path).resolve()
        stem = video_path.stem
        work_dir = Path(work_dir)
        audio_dir = work_dir / "audio"
        frames_dir = work_dir / "frames"
        audio_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        wav_path = audio_dir / f"{stem}.wav"
        extract_audio_from_video(str(video_path), str(wav_path))
        video, video_mask, frame_paths = self._load_video(
            video_path,
            save_frames=self.save_frames,
            frames_dir=frames_dir,
        )
        top3_detail, top3_names = self._yamnet_top3(str(wav_path))

        if self.geocode_cache is not None and location_text in self.geocode_cache:
            lat, lon = self.geocode_cache[location_text]
        else:
            lat, lon = self._geocode_location(location_text)
            if self.geocode_cache is not None:
                self.geocode_cache[location_text] = (lat, lon)

        combined_text = f"{location_text}. Audio: {top3_names[0]}, {top3_names[1]}, {top3_names[2]}."

        meta = {
            "video_file": str(video_path),
            "csv_file": self.csv_file_path,
            "csv_row": csv_row,
            "location_text": location_text,
            "latitude": lat,
            "longitude": lon,
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
        return meta, video, video_mask
