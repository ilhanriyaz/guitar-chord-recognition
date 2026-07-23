"""
demo_whisper.py

Whisper demo: encoder-decoder + cross-attention, trained multitask on
680k hours of weakly-labeled, noisy, multilingual web audio.

WHAT TO SHOW AND WHY:
The contrast with wav2vec2 isn't "better feature learning" -- it's
architecture and training philosophy:
  - wav2vec2: encoder only, self-supervised pretraining, needs a CTC/LM
    head fine-tuned per task.
  - Whisper: full encoder-DECODER, trained end-to-end and supervised
    (albeit weakly) directly on (audio, text) pairs at massive scale, so
    it can transcribe, translate, or detect language with ONE model and
    no fine-tuning.

The visual payoff here is the cross-attention map: for each output token
the decoder generates, which encoder audio frames did it attend to? Unlike
the GMM-HMM's rigid left-to-right state machine, Whisper's decoder is free
to attend anywhere in the audio -- and in practice you'll usually still see
a roughly monotonic diagonal, which is itself a great talking point: "it
wasn't told to go in order, it learned that speech and text are usually
monotonically aligned."

USAGE:
  # Transcribe an audio file
  python whisper_demo.py --wav test_clip.wav --model small

  # Record live from the microphone -- press Enter to stop each recording,
  # then 'q' + Enter to quit the session
  python whisper_demo.py --live --model small

  (model options: tiny, base, small, medium, large-v3)
"""

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
