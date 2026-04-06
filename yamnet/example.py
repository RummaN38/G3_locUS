# Imports.
from pathlib import Path

import numpy as np
import soundfile as sf

import matplotlib.pyplot as plt

import params as yamnet_params
import yamnet as yamnet_model
import tensorflow as tf

_ROOT = Path(__file__).resolve().parent

TOP_K = 1

def _format_timestamp(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:06.3f}"

def topk_timeline(scores: np.ndarray, class_names: list, params: yamnet_params.Params) -> str:
    """Plain text: one line per frame, timestamp (patch center) and top-K class labels with scores."""
    hop = params.patch_hop_seconds
    half_win = params.patch_window_seconds / 2.0
    text = ""
    for i in range(scores.shape[0]):
        t_center = i * hop + half_win
        row = scores[i]
        idx = np.argsort(row)[::-1][:TOP_K]
        parts = [f"{class_names[j]} ({float(row[j]):.3f})" for j in idx]
        text += f"{_format_timestamp(t_center)}  " + " | ".join(parts) + "; "
    return text

# Read in the audio.
wav_file_name = _ROOT / 'speech_whistling2.wav'
wav_data, sr = sf.read(wav_file_name, dtype=np.int16)
waveform = wav_data / 32768.0

# The graph is designed for a sampling rate of 16 kHz, but higher rates should work too.
# We also generate scores at a 10 Hz frame rate.
params = yamnet_params.Params(sample_rate=sr, patch_hop_seconds=0.1)
print("Sample rate =", params.sample_rate)

# Set up the YAMNet model.
class_names = yamnet_model.class_names(str(_ROOT / 'yamnet_class_map.csv'))
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights(str(_ROOT / 'yamnet.h5'))

# Run the model.
scores, embeddings, spectrogram = yamnet(waveform)
scores = scores.numpy()
spectrogram = spectrogram.numpy()

# Use this as audio -> text mapping, you can make it a bit more enriched
print("--- Top-%d labels per frame (time = patch center) ---" % TOP_K)
text = topk_timeline(scores, class_names, params)
print(text)

# Visualize the results.
plt.figure(figsize=(10, 8))

# Plot the waveform.
plt.subplot(3, 1, 1)
plt.plot(waveform)
plt.xlim([0, len(waveform)])
# Plot the log-mel spectrogram (returned by the model).
plt.subplot(3, 1, 2)
plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='lower')

# Plot and label the model output scores for the top-scoring classes.
mean_scores = np.mean(scores, axis=0)
top_N = 10
top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
plt.subplot(3, 1, 3)
plt.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
# Compensate for the patch_window_seconds (0.96s) context window to align with spectrogram.
patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds
plt.xlim([-patch_padding, scores.shape[0] + patch_padding])
# Label the top_N classes.
yticks = range(0, top_N, 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
_ = plt.ylim(-0.5 + np.array([top_N, 0]))
plt.show()
