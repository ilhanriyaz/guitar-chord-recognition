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

  # Record live from the microphone (5 seconds by default)
  python whisper_demo.py --live --model small
  python whisper_demo.py --live --duration 8

  (model options: tiny, base, small, medium, large-v3)
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from librosa import load

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio


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


def transcribe_with_attention(y: np.ndarray, sr: int = SAMPLE_RATE, model_size: str = "small"):
    model_name = f"openai/whisper-{model_size}"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.float()
    model.eval()

    inputs = processor(y, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        generated = model.generate(
            inputs["input_features"],
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
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording length in seconds (used with --live)")
    parser.add_argument("--model", default="small",
                         choices=["tiny", "base", "small", "medium", "large-v3"])
    args = parser.parse_args()

    if args.live:
        y, sr = record_audio(args.duration)
    else:
        y, sr = load_audio(args.wav)

    transcription = transcribe_with_attention(y, sr, args.model)

    print(f"\nTranscription: {transcription}")
