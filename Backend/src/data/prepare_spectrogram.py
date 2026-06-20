import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from huggingface_hub import snapshot_download

def save_spectrogram_image(file_path, output_path):
    y, sr = librosa.load(file_path, sr=22050)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(
        spec_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        fmax=8000,
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title=f"Mel spectrogram: {Path(file_path).stem}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def process_split(split_dir, output_dir):
    generated = 0
    # Each subfolder is a chord label (A, A#, G, etc.)
    for chord_name in os.listdir(split_dir):
        chord_dir = os.path.join(split_dir, chord_name)
        if not os.path.isdir(chord_dir):
            continue
        chord_output_dir = os.path.join(output_dir, chord_name)
        os.makedirs(chord_output_dir, exist_ok=True)
        for wav_file in os.listdir(chord_dir):
            if not wav_file.endswith(".wav"):
                continue
            file_path = os.path.join(chord_dir, wav_file)
            try:
                image_name = os.path.splitext(wav_file)[0] + ".png"
                output_path = os.path.join(chord_output_dir, image_name)
                save_spectrogram_image(file_path, output_path)
                generated += 1
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
    return generated

def main():
    # Download raw files only — no audio decoding by HuggingFace
    print("Downloading dataset files...")
    repo_dir = snapshot_download(
        repo_id="rodriler/isolated-guitar-chords",
        repo_type="dataset"
    )
    print(f"Downloaded to: {repo_dir}")

    train_dir = os.path.join(repo_dir, "data", "Train")
    test_dir  = os.path.join(repo_dir, "data", "Test")
    output_root = os.path.join("data", "processed", "spectrogram_images")

    print("Processing train split...")
    train_count = process_split(train_dir, os.path.join(output_root, "train"))

    print("Processing test split...")
    test_count = process_split(test_dir, os.path.join(output_root, "test"))

    print(f"Done. Train images: {train_count}, Test images: {test_count}")
    print(f"Saved under: {output_root}")

if __name__ == "__main__":
    main()