import argparse
from functools import lru_cache

import numpy as np


from librosa import load

MODEL_NAME = "facebook/wav2vec2-base-960h"  # swap for a larger checkpoint if you want stronger noise/accent robustness
SAMPLE_RATE = 16000  # wav2vec2 expects 16kHz audio

def load_audio(wav_path: str):
    """Load an audio file resampled to 16kHz mono."""
    y, sr = load(wav_path, sr=SAMPLE_RATE)
    return y, sr


def record_audio():
    """Record mono audio from the default microphone until Enter is pressed."""
    import sounddevice as sd

    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    print("\nRecording -- speak now, press Enter to stop...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input()
    print("Recording finished.")

    y = np.concatenate(frames, axis=0).flatten() if frames else np.zeros(0, dtype="float32")
    return y, SAMPLE_RATE


@lru_cache(maxsize=None)
def load_model():
    """Load the wav2vec2 processor/model once (cached) onto the GPU if available."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, logging
    logging.set_verbosity_error()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_NAME} on {device} ({dtype})...")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()

    return processor, model, device, dtype


def transcribe(y: np.ndarray, sr: int = SAMPLE_RATE):
    import torch

    processor, model, device, dtype = load_model()

    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_values"] = inputs["input_values"].to(dtype)

    with torch.inference_mode():
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
    args = parser.parse_args()

    if args.live:
        load_model()  # load once, before the loop
        while True:
            print("\nPress Enter to start recording (or 'q' + Enter to quit)...")
            if input().strip().lower() == "q":
                break
            y, sr = record_audio()
            transcription = transcribe(y, sr)
            print(f"\nTranscription: {transcription}")
    else:
        y, sr = load_audio(args.wav)
        transcription = transcribe(y, sr)
        print(f"\nTranscription: {transcription}")
