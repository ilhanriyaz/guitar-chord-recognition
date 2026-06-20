import os
import numpy as np
import librosa
from huggingface_hub import snapshot_download

N_MELS = 128
TIME_STEPS = 128  # fixed number of time frames for both representations


def fix_length(feature, time_steps=TIME_STEPS):
    """Pad or truncate the time axis (last axis) to a fixed length so every
    sample has the same shape, regardless of the source clip's duration."""
    if feature.shape[1] < time_steps:
        pad_width = time_steps - feature.shape[1]
        feature = np.pad(feature, ((0, 0), (0, pad_width)), mode="constant")
    else:
        feature = feature[:, :time_steps]
    return feature


def extract_spectrogram(y, sr):
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return fix_length(spec_db).astype(np.float32)


def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return fix_length(chroma).astype(np.float32)


def process_split(split_dir):
    spectrograms, chromas, labels = [], [], []
    # Each subfolder is a chord label (A, A#, G, etc.)
    for chord_name in sorted(os.listdir(split_dir)):
        chord_dir = os.path.join(split_dir, chord_name)
        if not os.path.isdir(chord_dir):
            continue
        for wav_file in os.listdir(chord_dir):
            if not wav_file.endswith(".wav"):
                continue
            file_path = os.path.join(chord_dir, wav_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                spectrograms.append(extract_spectrogram(y, sr))
                chromas.append(extract_chroma(y, sr))
                labels.append(chord_name)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
    return (
        np.stack(spectrograms),
        np.stack(chromas),
        np.array(labels),
    )


def main():
    # Download raw files only — no audio decoding by HuggingFace
    print("Downloading dataset files...")
    repo_dir = snapshot_download(
        repo_id="rodriler/isolated-guitar-chords",
        repo_type="dataset"
    )
    print(f"Downloaded to: {repo_dir}")

    train_dir = os.path.join(repo_dir, "data", "Train")
    test_dir = os.path.join(repo_dir, "data", "Test")

    print("Processing train split...")
    train_spec, train_chroma, train_labels = process_split(train_dir)

    print("Processing test split...")
    test_spec, test_chroma, test_labels = process_split(test_dir)

    os.makedirs("data/processed/cnn", exist_ok=True)
    np.savez(
        "data/processed/cnn/train.npz",
        spectrogram=train_spec,
        chroma=train_chroma,
        label=train_labels,
    )
    np.savez(
        "data/processed/cnn/test.npz",
        spectrogram=test_spec,
        chroma=test_chroma,
        label=test_labels,
    )

    print(f"Done. Train: {len(train_labels)} samples, Test: {len(test_labels)} samples")
    print(f"Spectrogram shape: {train_spec.shape[1:]}, Chroma shape: {train_chroma.shape[1:]}")


if __name__ == "__main__":
    main()
