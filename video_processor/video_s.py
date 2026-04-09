import os
import sys
import subprocess
import numpy as np
import cv2
import soundfile as sf
import matplotlib.pyplot as plt

import params as yamnet_params
import yamnet as yamnet_model

# SETTINGS  — change these as needed
TOP_K             = 3      # how many top classes to show per time frame
PATCH_HOP_SECONDS = 0.1    # time resolution of predictions (seconds)
OUTPUT_IMAGE      = "average_frame.png"
OUTPUT_TEXT       = "audio_classes.txt"
TEMP_AUDIO        = "_temp_audio.wav"

# ffmpeg binary path:
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_FFMPEG = os.path.join(_SCRIPT_DIR, "ffmpeg")
FFMPEG_BIN = _LOCAL_FFMPEG if os.path.isfile(_LOCAL_FFMPEG) else "ffmpeg"

def extract_audio_from_video(video_path: str, out_wav: str):
    """Use ffmpeg to extract audio track from the video as a WAV file."""
    print("[1/4] Extracting audio from video ...")
    # print(f"    Using ffmpeg at: {FFMPEG_BIN}")
    cmd = [
        FFMPEG_BIN, "-y",           # -y = overwrite without asking
        "-i", video_path,           # input video
        "-ac", "1",                 # audio channels = 1 (convert stereo to mono)
        "-ar", "16000",             # resample to audio rate = 16 kHz (required by YAMNet)
        "-vn",                      # video: none (discard the video track, keep only audio)
        out_wav                     # output WAV file
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed. Make sure ffmpeg is available."
        )
    print(f"    Audio saved to: {out_wav}")

def compute_average_frame(video_path: str) -> np.ndarray:
    """Read every frame and compute the pixel-level average."""
    print("[2/4] Computing average frame ...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"    Video: {width}x{height}, {total_frames} frames")

    # float64 accumulator to avoid overflow when summing many frames
    accumulator = np.zeros((height, width, 3), dtype=np.float64)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads as BGR — convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        accumulator += frame_rgb.astype(np.float64)
        count += 1
    cap.release()

    if count == 0:
        raise RuntimeError("No frames could be read from the video.")
    average = (accumulator / count).astype(np.uint8)
    print(f"    Averaged {count} frames.")
    return average

def save_and_show_average_frame(avg_frame: np.ndarray, out_path: str):
    """Save the average frame as PNG and display it with matplotlib."""
    print(f"[3/4] Saving average frame to: {out_path}")

    # Save to disk (convert RGB → BGR for OpenCV imwrite)
    cv2.imwrite(out_path, cv2.cvtColor(avg_frame, cv2.COLOR_RGB2BGR))

    # Display with matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(avg_frame)
    plt.axis('off')
    plt.title("Average Frame (pixel-level)")
    plt.tight_layout()
    # plt.show()

def _format_timestamp(sec: float) -> str:
    """Convert seconds to MM:SS.mmm string."""
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:06.3f}"

def run_yamnet(wav_path: str, out_txt: str):
    """Load YAMNet, run on extracted audio, print + save predictions."""
    print("[4/4] Running YAMNet audio classification ...")

    # -- Build the model -- ffmpeg already resampled to 16000 Hz, so we match that here
    params = yamnet_params.Params(sample_rate=16000.0,
                                  patch_hop_seconds=PATCH_HOP_SECONDS)
    class_names = yamnet_model.class_names("yamnet_class_map.csv")
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights("yamnet.h5")

    # -- Load audio --
    wav_data, sr = sf.read(wav_path, dtype=np.int16)
    waveform = (wav_data / 32768.0).astype(np.float32)

    # Handle stereo just in case (ffmpeg already makes it mono, but be safe)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # -- Run model --
    scores, embeddings, spectrogram = yamnet(waveform)
    scores = scores.numpy()

    # -- Build text output --
    hop      = params.patch_hop_seconds
    half_win = params.patch_window_seconds / 2.0
    lines    = []

    for i in range(scores.shape[0]):
        t_center = i * hop + half_win
        row      = scores[i]
        top_idx  = np.argsort(row)[::-1][:TOP_K]
        parts    = [f"{class_names[j]} ({row[j]:.3f})" for j in top_idx]
        line     = f"{_format_timestamp(t_center)}    " + "  |  ".join(parts)
        lines.append(line)

    # Overall clip summary (average across all frames)
    mean_scores   = np.mean(scores, axis=0)
    top5_idx      = np.argsort(mean_scores)[::-1][:5]
    summary_lines = [
        "",
        "OVERALL CLIP SUMMARY (top 5 averaged across all frames)",
    ]
    for rank, idx in enumerate(top5_idx, start=1):
        summary_lines.append(f"  #{rank}  {class_names[idx]:<30}  score: {mean_scores[idx]:.4f}")

    all_lines = lines + summary_lines

    # -- Save to .txt --
    with open(out_txt, "w") as f:
        f.write("\n".join(all_lines))

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <path_to_video>")
        print("Example: python video_processor.py myvideo.mp4")
        sys.exit(1)

    video_path = sys.argv[1] # Reads the video filename from the command line and checks it actually exists on disk.
    if not os.path.isfile(video_path):
        print(f"Error: file not found — {video_path}")
        sys.exit(1)

    print(f"\n=== Processing: {video_path} ===\n")

    # Step 1 — extract audio
    extract_audio_from_video(video_path, TEMP_AUDIO)

    # Step 2 — compute average frame
    avg_frame = compute_average_frame(video_path)

    # Step 3 — save + show average frame
    save_and_show_average_frame(avg_frame, OUTPUT_IMAGE)

    # Step 4 — run YAMNet on audio
    run_yamnet(TEMP_AUDIO, OUTPUT_TEXT)

if __name__ == "__main__":
    main()
