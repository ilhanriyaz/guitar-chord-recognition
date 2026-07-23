'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

// ---------- music theory helpers ----------

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

type ChordTemplate = { name: string; vec: number[]; tones: number[] };

function buildTemplates(): ChordTemplate[] {
  const majorIntervals = [0, 4, 7];
  const minorIntervals = [0, 3, 7];
  const templates: ChordTemplate[] = [];

  for (let root = 0; root < 12; root++) {
    const maj = new Array(12).fill(0);
    majorIntervals.forEach((iv) => (maj[(root + iv) % 12] = 1));
    templates.push({ name: NOTE_NAMES[root], vec: maj, tones: majorIntervals.map((iv) => (root + iv) % 12) });

    const min = new Array(12).fill(0);
    minorIntervals.forEach((iv) => (min[(root + iv) % 12] = 1));
    templates.push({ name: `${NOTE_NAMES[root]}m`, vec: min, tones: minorIntervals.map((iv) => (root + iv) % 12) });
  }
  return templates;
}

const TEMPLATES = buildTemplates();

function cosineSim(a: number[], b: number[]): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Given a 12-bin pitch-class energy vector, find the closest matching chord template.
function classifyChroma(chroma: number[]): { name: string; score: number } {
  const total = chroma.reduce((sum, value) => sum + value, 0);
  if (total < 1e-6) return { name: '—', score: 0 };
  const norm = chroma.map((v) => v / total);

  let best: ChordTemplate | null = null;
  let bestScore = -1;
  let secondScore = -1;
  for (const t of TEMPLATES) {
    const toneEnergy = t.tones.reduce((sum, pc) => sum + norm[pc], 0);
    const weakestTone = Math.min(...t.tones.map((pc) => norm[pc]));
    const strongestTone = Math.max(...t.tones.map((pc) => norm[pc]));
    const balance = strongestTone > 0 ? weakestTone / strongestTone : 0;
    const score = 0.55 * cosineSim(norm, t.vec) + 0.3 * toneEnergy + 0.15 * balance;
    if (score > bestScore) {
      secondScore = bestScore;
      bestScore = score;
      best = t;
    } else if (score > secondScore) {
      secondScore = score;
    }
  }
  const margin = Math.max(0, bestScore - secondScore);
  const confidence = Math.min(1, Math.max(0, bestScore * 0.82 + margin * 2.5));
  return { name: best ? best.name : '—', score: confidence };
}

// Accumulate FFT bin magnitudes into a 12-bin pitch-class (chroma) vector.
//
// Only a small fraction of the spectrum carries chord information: the note
// fundamentals and their lowest harmonics. Everything else — the noise floor
// and the dense clutter of upper harmonics between notes — just smears energy
// across all 12 pitch classes and pulls the template match off the true chord.
// So we (1) cap the analysis band at CHROMA_MAX_FREQ, (2) gate out any bin that
// isn't loud relative to this frame's peak, and (3) weight by power (magnitude
// squared) so genuine peaks dominate the diffuse background.
function percentile(values: number[], fraction: number): number {
  if (values.length === 0) return -Infinity;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

function chromaFromFreqData(freqData: Float32Array, binHz: number): number[] {
  const chroma = new Array(12).fill(0);
  const startBin = Math.max(1, Math.floor(MIN_FREQ / binHz));
  const endBin = Math.min(freqData.length - 2, Math.ceil(CHROMA_MAX_FREQ / binHz));
  const band = Array.from(freqData.slice(startBin, endBin + 1)).filter(Number.isFinite);
  const peakDb = Math.max(...band);
  if (peakDb < CHROMA_SILENCE_DB) return chroma;

  // Use the lower part of this frame's spectrum as its noise estimate. A local
  // peak must clear both this adaptive floor and the frame-relative gate.
  const noiseFloorDb = percentile(band, 0.35);
  const gateDb = Math.max(CHROMA_FLOOR_DB, noiseFloorDb + NOISE_MARGIN_DB, peakDb - CHROMA_GATE_DB);
  const peaks: { midi: number; weight: number }[] = [];

  for (let i = startBin + 1; i < endBin; i++) {
    const db = freqData[i];
    if (db < gateDb || db <= freqData[i - 1] || db < freqData[i + 1]) continue;

    // Parabolic interpolation improves the frequency estimate between FFT bins,
    // especially for the low E/A strings where bins are several cents wide.
    const left = freqData[i - 1];
    const right = freqData[i + 1];
    const denominator = left - 2 * db + right;
    const offset =
      denominator === 0 ? 0 : Math.max(-0.5, Math.min(0.5, (0.5 * (left - right)) / denominator));
    const freq = (i + offset) * binHz;
    const midi = 69 + 12 * Math.log2(freq / 440);
    const lowFrequencyWeight = 1 / Math.sqrt(Math.max(1, freq / 220));
    peaks.push({ midi, weight: Math.sqrt(db - gateDb) * lowFrequencyWeight });
  }

  // Estimate a per-frame tuning offset. Harmonics share their fundamental's
  // cents offset, so a slightly flat or sharp guitar still lands on note centres.
  let sin = 0;
  let cos = 0;
  for (const peak of peaks) {
    const phase = (peak.midi - Math.round(peak.midi)) * Math.PI * 2;
    sin += Math.sin(phase) * peak.weight;
    cos += Math.cos(phase) * peak.weight;
  }
  const tuningOffset = peaks.length >= 3 ? Math.atan2(sin, cos) / (Math.PI * 2) : 0;

  for (const peak of peaks) {
    const tunedMidi = peak.midi - tuningOffset;
    const nearest = Math.round(tunedMidi);
    const distance = Math.abs(tunedMidi - nearest);
    if (distance > PEAK_NOTE_WIDTH) continue;
    const pc = ((nearest % 12) + 12) % 12;
    const pitchWeight = 0.5 + 0.5 * Math.cos((Math.PI * distance) / PEAK_NOTE_WIDTH);
    chroma[pc] += peak.weight * pitchWeight;
  }

  // Compression prevents one ringing string from drowning out the other tones.
  for (let pc = 0; pc < 12; pc++) chroma[pc] = Math.sqrt(chroma[pc]);
  return chroma;
}

const MIN_FREQ = 55; // ~A1, chroma detection floor
const CHROMA_MAX_FREQ = 1600; // fundamentals + low harmonics; above this is mostly clutter
const CHROMA_GATE_DB = 30; // keep bins within this many dB of the frame peak
const CHROMA_FLOOR_DB = -78; // absolute noise floor; bins quieter than this never count
const CHROMA_SILENCE_DB = -70; // if the peak is below this, the frame is silence
const NOISE_MARGIN_DB = 9;
const PEAK_NOTE_WIDTH = 0.45; // semitones either side of the tuned note centre

// Chord display latch: once a chord is detected confidently, keep showing it until a
// different chord clears the threshold. This stops the readout from flickering away on
// brief quiet spots or strum transitions.
const CHORD_HOLD_THRESHOLD = 0.48; // confidence (including runner-up margin) needed to replace display
const CHORD_CLEAR_MS = 2000; // after this long with no confident chord, clear to —

// Spectrum display range, aligned to octave boundaries (C2 .. C7) so the
// log-frequency axis lands cleanly on note gridlines.
const SPEC_MIN_FREQ = 65.41; // C2
const SPEC_MAX_FREQ = 2093.0; // C7
const SPEC_MIN_MIDI = 36; // C2
const SPEC_MAX_MIDI = 96; // C7
const HISTORY_LEN = 8;
const DETECTION_INTERVAL_MS = 80;
const CHROMA_EMA_ALPHA = 0.35;

const AXIS_H = 20; // reserved height at bottom of spectrum for the note axis

function freqToX(freq: number, w: number): number {
  const logMin = Math.log2(SPEC_MIN_FREQ);
  const logMax = Math.log2(SPEC_MAX_FREQ);
  const t = (Math.log2(freq) - logMin) / (logMax - logMin);
  return t * w;
}

function midiToFreq(m: number): number {
  return 440 * Math.pow(2, (m - 69) / 12);
}

// ---------- component ----------

export default function ChordDetectorPage() {
  const [mode, setMode] = useState<'mic' | 'file'>('file');

  // --- live microphone state ---
  const waveCanvasRef = useRef<HTMLCanvasElement>(null);
  const specCanvasRef = useRef<HTMLCanvasElement>(null);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const historyRef = useRef<{ name: string; score: number }[]>([]);
  const smoothedChromaRef = useRef<number[]>(new Array(12).fill(0));
  const lastDetectionAtRef = useRef(0);
  const lastConfidentAtRef = useRef(0); // timestamp of the last chord that cleared the hold threshold
  const streamRef = useRef<MediaStream | null>(null);

  const [listening, setListening] = useState(false);
  const [statusText, setStatusText] = useState('Microphone idle');
  const [chordName, setChordName] = useState('—');
  const [confidence, setConfidence] = useState(0);

  const drawWaveform = useCallback((analyser: AnalyserNode) => {
    const canvas = waveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);

    const w = canvas.width;
    const h = canvas.height;

    // white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    // center reference line
    ctx.strokeStyle = 'rgba(0,0,0,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // blue waveform
    ctx.strokeStyle = '#007aff';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    const slice = w / data.length;
    for (let i = 0; i < data.length; i++) {
      const v = data[i] / 128.0;
      const y = (v * h) / 2;
      const x = i * slice;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, []);

  const detectChord = useCallback((chroma: number[]) => {
    const now = performance.now();
    const { name, score } = classifyChroma(chroma);

    // Silent frame: don't overwrite the display. Hold the last chord through brief gaps,
    // and only clear back to — after a sustained silence.
    if (name === '—') {
      if (now - lastConfidentAtRef.current > CHORD_CLEAR_MS) {
        setChordName('—');
        setConfidence(0);
        historyRef.current = [];
      }
      return;
    }

    const history = historyRef.current;
    history.push({ name, score });
    if (history.length > HISTORY_LEN) history.shift();

    const counts: Record<string, number> = {};
    history.forEach((h) => (counts[h.name] = (counts[h.name] || 0) + 1));
    let winner = history[history.length - 1].name;
    let winnerCount = 0;
    for (const name in counts) {
      if (counts[name] > winnerCount) {
        winnerCount = counts[name];
        winner = name;
      }
    }
    const avgScore =
      history.filter((h) => h.name === winner).reduce((a, b) => a + b.score, 0) / winnerCount;

    // Latch: only swap the displayed chord once a candidate clears the threshold. Below it,
    // keep showing the last confident chord until the silence timeout elapses.
    if (avgScore >= CHORD_HOLD_THRESHOLD) {
      lastConfidentAtRef.current = now;
      setChordName(winner);
      setConfidence(Math.round(Math.min(1, Math.max(0, avgScore)) * 100));
    } else if (now - lastConfidentAtRef.current > CHORD_CLEAR_MS) {
      setChordName('—');
      setConfidence(0);
    }
  }, []);

  const drawSpectrumAndDetect = useCallback(
    (analyser: AnalyserNode, audioCtx: AudioContext) => {
      const canvas = specCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Short-Time Fourier Transform snapshot: the analyser applies a windowed
      // FFT to the most recent block of samples, so reading it every animation
      // frame gives a spectrum that tracks the signal in real time.
      const freqData = new Float32Array(analyser.frequencyBinCount);
      analyser.getFloatFrequencyData(freqData);

      const w = canvas.width;
      const h = canvas.height;
      const plotH = h - AXIS_H; // magnitude plot area (axis sits below)

      // white background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, w, h);

      const minDb = -100;
      const maxDb = -20;

      // horizontal magnitude gridlines + dB scale labels (y-axis: magnitude)
      ctx.textBaseline = 'middle';
      ctx.font = '10px -apple-system, system-ui, sans-serif';
      for (let db = maxDb; db >= minDb; db -= 20) {
        const norm = (db - minDb) / (maxDb - minDb);
        const y = plotH - norm * plotH;
        ctx.strokeStyle = 'rgba(0,0,0,0.05)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
        ctx.fillStyle = '#aeaeb2';
        ctx.fillText(`${db}`, 3, y - 6);
      }

      // vertical note gridlines + note-name axis (x-axis: log-scaled frequency)
      ctx.textBaseline = 'alphabetic';
      for (let m = SPEC_MIN_MIDI; m <= SPEC_MAX_MIDI; m++) {
        const freq = midiToFreq(m);
        const x = freqToX(freq, w);
        const pc = ((m % 12) + 12) % 12;
        const isC = pc === 0;

        ctx.strokeStyle = isC ? 'rgba(0,0,0,0.12)' : 'rgba(0,0,0,0.045)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, plotH);
        ctx.stroke();

        // label every natural note (sharps get a gridline only, to stay legible)
        const isNatural = ![1, 3, 6, 8, 10].includes(pc);
        if (isNatural) {
          const octave = Math.floor(m / 12) - 1;
          const label = isC ? `C${octave}` : NOTE_NAMES[pc];
          ctx.fillStyle = isC ? '#1d1d1f' : '#8e8e93';
          ctx.font = isC
            ? '600 10px -apple-system, system-ui, sans-serif'
            : '10px -apple-system, system-ui, sans-serif';
          const tw = ctx.measureText(label).width;
          ctx.fillText(label, x - tw / 2, h - 6);
        }
      }

      const sampleRate = audioCtx.sampleRate;
      const binHz = sampleRate / analyser.fftSize;

      // spectrum trace with a soft blue fill beneath it
      const points: { x: number; y: number }[] = [];
      for (let i = 1; i < freqData.length; i++) {
        const freq = i * binHz;
        if (freq < SPEC_MIN_FREQ || freq > SPEC_MAX_FREQ) continue;
        const db = freqData[i];
        const norm = Math.min(1, Math.max(0, (db - minDb) / (maxDb - minDb)));
        points.push({ x: freqToX(freq, w), y: plotH - norm * plotH });
      }

      if (points.length > 1) {
        const grad = ctx.createLinearGradient(0, 0, 0, plotH);
        grad.addColorStop(0, 'rgba(0,122,255,0.28)');
        grad.addColorStop(1, 'rgba(0,122,255,0.02)');
        ctx.beginPath();
        ctx.moveTo(points[0].x, plotH);
        points.forEach((p) => ctx.lineTo(p.x, p.y));
        ctx.lineTo(points[points.length - 1].x, plotH);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.beginPath();
        points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.strokeStyle = '#007aff';
        ctx.lineWidth = 1.75;
        ctx.lineJoin = 'round';
        ctx.stroke();
      }

      const now = performance.now();
      if (now - lastDetectionAtRef.current >= DETECTION_INTERVAL_MS) {
        const chroma = chromaFromFreqData(freqData, binHz);
        const smoothed = smoothedChromaRef.current;
        const hasSignal = chroma.some((value) => value > 0);
        for (let pc = 0; pc < 12; pc++) {
          smoothed[pc] = hasSignal
            ? CHROMA_EMA_ALPHA * chroma[pc] + (1 - CHROMA_EMA_ALPHA) * smoothed[pc]
            : 0;
        }
        lastDetectionAtRef.current = now;
        detectChord(smoothed);
      }
    },
    [detectChord],
  );

  const loopRef = useRef<() => void>(() => {});

  const loop = useCallback(() => {
    const analyser = analyserRef.current;
    const audioCtx = audioCtxRef.current;
    if (!analyser || !audioCtx) return;
    drawWaveform(analyser);
    drawSpectrumAndDetect(analyser, audioCtx);
    rafRef.current = requestAnimationFrame(() => loopRef.current());
  }, [drawWaveform, drawSpectrumAndDetect]);

  useEffect(() => {
    loopRef.current = loop;
  }, [loop]);

  const start = useCallback(async () => {
    try {
      setStatusText('requesting microphone…');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      streamRef.current = stream;

      const AudioContextCtor =
        window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioContextCtor();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 16384;
      analyser.smoothingTimeConstant = 0.35;
      source.connect(analyser);

      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      setListening(true);
      setStatusText('listening…');
      rafRef.current = requestAnimationFrame(() => loopRef.current());
    } catch (err) {
      setStatusText(`microphone access denied or unavailable: ${(err as Error).message}`);
    }
  }, []);

  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    audioCtxRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current = null;
    analyserRef.current = null;
    streamRef.current = null;
    historyRef.current = [];
    smoothedChromaRef.current.fill(0);
    lastDetectionAtRef.current = 0;
    lastConfidentAtRef.current = 0;
    setListening(false);
    setStatusText('Microphone idle');
    setChordName('—');
    setConfidence(0);
  }, []);

  // cleanup mic on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      audioCtxRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // --- uploaded-file state ---
  // An uploaded audio/video file is played through the same AnalyserNode pipeline
  // as the microphone, so its waveform, spectrum, and chord readout update live and
  // stay in sync with playback (and with the video frame, for video files).
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState(false);
  const [filePlaying, setFilePlaying] = useState(false);

  const mediaElRef = useRef<HTMLMediaElement | null>(null);
  const fileCtxRef = useRef<AudioContext | null>(null);
  const fileAnalyserRef = useRef<AnalyserNode | null>(null);
  const fileSourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const fileRafRef = useRef<number | null>(null);
  const fileLoopRef = useRef<() => void>(() => {});

  const stopFileLoop = useCallback(() => {
    if (fileRafRef.current) cancelAnimationFrame(fileRafRef.current);
    fileRafRef.current = null;
  }, []);

  const teardownFileGraph = useCallback(() => {
    stopFileLoop();
    fileCtxRef.current?.close();
    fileCtxRef.current = null;
    fileAnalyserRef.current = null;
    fileSourceRef.current = null;
  }, [stopFileLoop]);

  const fileLoop = useCallback(() => {
    const analyser = fileAnalyserRef.current;
    const ctx = fileCtxRef.current;
    if (!analyser || !ctx) return;
    drawWaveform(analyser);
    drawSpectrumAndDetect(analyser, ctx);
    fileRafRef.current = requestAnimationFrame(() => fileLoopRef.current());
  }, [drawWaveform, drawSpectrumAndDetect]);

  useEffect(() => {
    fileLoopRef.current = fileLoop;
  }, [fileLoop]);

  // Lazily wire the media element into a WebAudio graph. createMediaElementSource can
  // only run once per element, so the element is remounted (keyed on fileUrl) whenever
  // a new file is chosen and the previous graph is torn down first.
  const ensureFileGraph = useCallback(() => {
    const el = mediaElRef.current;
    if (!el || fileCtxRef.current) return;
    const AudioContextCtor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AudioContextCtor();
    const source = ctx.createMediaElementSource(el);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 16384;
    analyser.smoothingTimeConstant = 0.35;
    source.connect(analyser);
    analyser.connect(ctx.destination); // keep the file audible during analysis
    fileCtxRef.current = ctx;
    fileSourceRef.current = source;
    fileAnalyserRef.current = analyser;
  }, []);

  const handleMediaPlay = useCallback(() => {
    ensureFileGraph();
    fileCtxRef.current?.resume();
    setFilePlaying(true);
    stopFileLoop();
    fileRafRef.current = requestAnimationFrame(() => fileLoopRef.current());
  }, [ensureFileGraph, stopFileLoop]);

  const handleMediaPause = useCallback(() => {
    stopFileLoop();
    setFilePlaying(false);
  }, [stopFileLoop]);

  useEffect(() => {
    return () => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
    };
  }, [fileUrl]);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      teardownFileGraph();
      setSelectedFile(file);
      setIsVideo(file.type.startsWith('video'));
      setFilePlaying(false);
      setChordName('—');
      setConfidence(0);
      historyRef.current = [];
      smoothedChromaRef.current.fill(0);
      lastDetectionAtRef.current = 0;
      lastConfidentAtRef.current = 0;
      setFileUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
    },
    [teardownFileGraph],
  );

  // Stop live file analysis when leaving file mode, and clean it up on unmount.
  useEffect(() => {
    if (mode !== 'file') {
      // pause() fires the media element's 'pause' event, which clears filePlaying.
      mediaElRef.current?.pause();
      stopFileLoop();
    }
  }, [mode, stopFileLoop]);

  useEffect(() => {
    return () => teardownFileGraph();
  }, [teardownFileGraph]);

  return (
    <main className="min-h-screen bg-[#f5f5f7] text-[#1d1d1f] flex flex-col items-center px-6 py-12">
      <header className="w-full max-w-3xl mb-8 text-center">
        <div className="text-[13px] font-medium tracking-wide uppercase text-[#0071e3]">
          Chord Detector
        </div>
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight mt-2">
          Chord Scope
        </h1>
        <p className="text-[16px] text-[#6e6e73] mt-3">
          Real-time waveform, STFT spectrum, and chroma-based chord detection.
        </p>
      </header>

      {/* segmented mode toggle */}
      <div className="w-full max-w-3xl flex justify-center mb-8">
        <div className="inline-flex gap-1 p-1 rounded-full bg-black/5">
          <button
            onClick={() => {
              if (listening) stop();
              setMode('file');
            }}
            className={`text-[14px] font-medium px-6 py-2 rounded-full transition-all ${
              mode === 'file'
                ? 'bg-white text-[#1d1d1f] shadow-[0_1px_4px_rgba(0,0,0,0.12)]'
                : 'text-[#6e6e73] hover:text-[#1d1d1f]'
            }`}
          >
            Upload file
          </button>
          <button
            onClick={() => setMode('mic')}
            className={`text-[14px] font-medium px-6 py-2 rounded-full transition-all ${
              mode === 'mic'
                ? 'bg-white text-[#1d1d1f] shadow-[0_1px_4px_rgba(0,0,0,0.12)]'
                : 'text-[#6e6e73] hover:text-[#1d1d1f]'
            }`}
          >
            Microphone
          </button>
        </div>
      </div>

      {mode === 'file' && (
        <section className="w-full max-w-3xl bg-white rounded-3xl p-7 shadow-[0_2px_16px_rgba(0,0,0,0.06)] ring-1 ring-black/5">
          <div className="text-[13px] font-semibold uppercase tracking-wide text-[#6e6e73] mb-4">
            Upload an audio or video recording
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            <input
              type="file"
              accept="audio/*,video/*"
              onChange={handleFileChange}
              className="text-[14px] text-[#6e6e73] file:mr-4 file:border-0 file:bg-black/5 file:text-[#1d1d1f] file:rounded-full file:px-5 file:py-2.5 file:text-[14px] file:font-medium file:cursor-pointer file:transition-colors hover:file:bg-black/10"
            />
            <span className="text-[14px] text-[#6e6e73]">
              {selectedFile ? (filePlaying ? 'Analyzing playback…' : 'Press play to analyze') : 'No file selected'}
            </span>
          </div>

          {fileUrl && isVideo && (
            <video
              key={fileUrl}
              ref={mediaElRef as React.RefObject<HTMLVideoElement>}
              controls
              src={fileUrl}
              onPlay={handleMediaPlay}
              onPause={handleMediaPause}
              onEnded={handleMediaPause}
              className="w-full mt-6 rounded-2xl bg-black ring-1 ring-black/5 max-h-[420px]"
            >
              Your browser does not support video playback.
            </video>
          )}

          {fileUrl && !isVideo && (
            <audio
              key={fileUrl}
              ref={mediaElRef as React.RefObject<HTMLAudioElement>}
              controls
              src={fileUrl}
              onPlay={handleMediaPlay}
              onPause={handleMediaPause}
              onEnded={handleMediaPause}
              className="w-full mt-6 h-10"
            >
              Your browser does not support audio playback.
            </audio>
          )}

          {fileUrl && (
            <>
              <div className="text-[13px] font-semibold uppercase tracking-wide text-[#6e6e73] mt-6 mb-2">
                Waveform — time domain
              </div>
              <canvas
                ref={waveCanvasRef}
                width={900}
                height={140}
                className="w-full rounded-2xl ring-1 ring-black/5"
              />

              <div className="text-[13px] font-semibold uppercase tracking-wide text-[#6e6e73] mt-6 mb-2">
                Spectrum — log frequency (note) &times; magnitude
              </div>
              <canvas
                ref={specCanvasRef}
                width={900}
                height={280}
                className="w-full rounded-2xl ring-1 ring-black/5"
              />

              <div className="flex items-center gap-8 mt-8 flex-wrap">
                <div className="text-7xl font-semibold text-[#0071e3] min-w-[160px] tracking-tight">
                  {chordName}
                </div>
                <div className="text-[13px] text-[#6e6e73]">
                  Confidence
                  <div className="w-44 h-1.5 bg-black/8 rounded-full overflow-hidden mt-1.5">
                    <div
                      className="h-full bg-[#0071e3] transition-all duration-100"
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                  <span className="tabular-nums">{confidence}%</span>
                </div>
              </div>
            </>
          )}
        </section>
      )}

      {mode === 'mic' && (
        <section className="w-full max-w-3xl bg-white rounded-3xl p-7 shadow-[0_2px_16px_rgba(0,0,0,0.06)] ring-1 ring-black/5">
          <div className="flex items-center gap-4 mb-6 flex-wrap">
            <button
              onClick={listening ? stop : start}
              className={`text-[15px] font-medium px-6 py-2.5 rounded-full shadow-[0_2px_8px_rgba(0,0,0,0.15)] transition-all active:scale-[0.98] ${
                listening
                  ? 'bg-[#ff3b30] text-white shadow-[0_2px_8px_rgba(255,59,48,0.35)] hover:bg-[#ff453a]'
                  : 'bg-[#0071e3] text-white shadow-[0_2px_8px_rgba(0,113,227,0.35)] hover:bg-[#0077ed]'
              }`}
            >
              {listening ? 'Stop' : 'Start listening'}
            </button>
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                listening ? 'bg-[#34c759] shadow-[0_0_8px_#34c759]' : 'bg-[#c7c7cc]'
              }`}
            />
            <span className="text-[14px] text-[#6e6e73]">{statusText}</span>
          </div>

          <div className="text-[13px] font-semibold uppercase tracking-wide text-[#6e6e73] mb-2">
            Waveform — time domain
          </div>
          <canvas
            ref={waveCanvasRef}
            width={900}
            height={140}
            className="w-full rounded-2xl ring-1 ring-black/5"
          />

          <div className="text-[13px] font-semibold uppercase tracking-wide text-[#6e6e73] mt-6 mb-2">
            Spectrum — log frequency (note) &times; magnitude
          </div>
          <canvas
            ref={specCanvasRef}
            width={900}
            height={280}
            className="w-full rounded-2xl ring-1 ring-black/5"
          />

          <div className="flex items-center gap-8 mt-8 flex-wrap">
            <div className="text-7xl font-semibold text-[#0071e3] min-w-[160px] tracking-tight">
              {chordName}
            </div>
            <div className="text-[13px] text-[#6e6e73]">
              Confidence
              <div className="w-44 h-1.5 bg-black/8 rounded-full overflow-hidden mt-1.5">
                <div
                  className="h-full bg-[#0071e3] transition-all duration-100"
                  style={{ width: `${confidence}%` }}
                />
              </div>
              <span className="tabular-nums">{confidence}%</span>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
