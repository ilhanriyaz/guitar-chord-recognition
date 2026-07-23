"""
demo_wav2vec2.py

wav2vec2 demo: self-supervised pretraining + CTC fine-tuning.

WHAT TO SHOW AND WHY:
The core contrast with GMM-HMM is that wav2vec2 never sees a hand-designed
feature like an MFCC -- it learns its own representation directly from raw
waveform samples via contrastive self-supervised pretraining, then a thin
CTC head maps those learned representations to characters/subwords.

Two things worth putting on screen:
  1. The transcript itself (works on out-of-vocabulary words / made-up words
     far better than the GMM-HMM, because there's no fixed lexicon).
  2. A 2D projection (PCA) of the model's internal hidden-state embeddings,
     colored by the character each frame decodes to. This is the visual
     payoff: you can literally see the model has organized similar sounds
     into clusters *without anyone telling it what a phoneme is*.

USAGE:
  # Transcribe an audio file
  python wav2vec_demo.py --wav test_clip.wav

  # Record live from the microphone (5 seconds by default)
  python wav2vec_demo.py --live
  python wav2vec_demo.py --live --duration 8
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from librosa import load

MODEL_NAME = "facebook/wav2vec2-base-960h"  # swap for a larger checkpoint if you want stronger noise/accent robustness
SAMPLE_RATE = 16000  # wav2vec2 expects 16kHz audio


def load_audio(wav_path: str):
    """Load an audio file resampled to 16kHz mono."""
    y, sr = load(wav_path, sr=SAMPLE_RATE)
    return y, sr


def record_audio(duration: float = 5.0):
    """Record `duration` seconds of mono audio from the default microphone."""
    import sounddevice as sd

    print(f"\nRecording for {duration:.0f}s -- speak now...")
    y = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")
    return y.flatten(), SAMPLE_RATE


def transcribe(y: np.ndarray, sr: int = SAMPLE_RATE):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()

    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, T, vocab_size)
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", help="Audio clip to transcribe")
    source.add_argument("--live", action="store_true", help="Record live from the microphone")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording length in seconds (used with --live)")
    args = parser.parse_args()

    if args.live:
        y, sr = record_audio(args.duration)
    else:
        y, sr = load_audio(args.wav)

    transcription = transcribe(y, sr)

    print(f"\nTranscription: {transcription}")
