import os
import numpy as np
import pandas as pd
import librosa
from huggingface_hub import snapshot_download

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std  = np.std(chroma, axis=1)
    return np.concatenate([chroma_mean, chroma_std])

def process_split(split_dir):
    rows = []
    # Each subfolder is a chord label (A, A#, G, etc.)
    for chord_name in os.listdir(split_dir):
        chord_dir = os.path.join(split_dir, chord_name)
        if not os.path.isdir(chord_dir):
            continue
        for wav_file in os.listdir(chord_dir):
            if not wav_file.endswith(".wav"):
                continue
            file_path = os.path.join(chord_dir, wav_file)
            try:
                features = extract_features(file_path)
                rows.append({
                    "label": chord_name,
                    **{f"feature_{i}": v for i, v in enumerate(features)}
                })
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
    return pd.DataFrame(rows)

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

    print("Processing train split...")
    train_df = process_split(train_dir)

    print("Processing test split...")
    test_df = process_split(test_dir)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"Done. Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(f"Chords found: {sorted(train_df['label'].unique())}")

if __name__ == "__main__":
    main()