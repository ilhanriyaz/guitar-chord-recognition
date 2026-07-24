import argparse
from functools import lru_cache

import numpy as np


from librosa import load

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio


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
def load_model(model_size: str = "small"):
    """Load the Whisper processor/model once (cached) onto the GPU if available."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, logging
    logging.set_verbosity_error()  # suppress warnings about missing tokenizer files

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading whisper-{model_size} on {device} ({dtype})...")

    model_name = f"openai/whisper-{model_size}"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
    model = model.to(device)
    model.eval()

    return processor, model, device, dtype


def transcribe_with_attention(y: np.ndarray, sr: int = SAMPLE_RATE, model_size: str = "small"):
    import torch

    processor, model, device, dtype = load_model(model_size)

    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to(device=device, dtype=dtype)

    with torch.inference_mode():
        generated = model.generate(
            input_features,
            output_attentions=True,
            return_dict_in_generate=True,
            max_new_tokens=128,
        )

    transcription = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]

    return transcription


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", help="Audio clip to transcribe")
    source.add_argument("--live", action="store_true", help="Record live from the microphone")
    parser.add_argument("--model", default="small",
                         choices=["tiny", "base", "small", "medium", "large-v3"])
    args = parser.parse_args()

    if args.live:
        load_model(args.model)  # load once, before the loop
        while True:
            print("\nPress Enter to start recording (or 'q' + Enter to quit)...")
            if input().strip().lower() == "q":
                break
            y, sr = record_audio()
            transcription = transcribe_with_attention(y, sr, args.model)
            print(f"\nTranscription: {transcription}")
    else:
        y, sr = load_audio(args.wav)
        transcription = transcribe_with_attention(y, sr, args.model)
        print(f"\nTranscription: {transcription}")
