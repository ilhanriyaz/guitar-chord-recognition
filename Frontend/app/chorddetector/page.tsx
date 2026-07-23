'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

// ---------- music theory helpers ----------

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

type ChordTemplate = { name: string; vec: number[] };

function buildTemplates(): ChordTemplate[] {
  const majorIntervals = [0, 4, 7];
  const minorIntervals = [0, 3, 7];
  const templates: ChordTemplate[] = [];

  for (let root = 0; root < 12; root++) {
    const maj = new Array(12).fill(0);
    majorIntervals.forEach((iv) => (maj[(root + iv) % 12] = 1));
    templates.push({ name: NOTE_NAMES[root], vec: maj });

    const min = new Array(12).fill(0);
    minorIntervals.forEach((iv) => (min[(root + iv) % 12] = 1));
    templates.push({ name: `${NOTE_NAMES[root]}m`, vec: min });
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
  const maxC = Math.max(...chroma);
  if (maxC < 1e-6) return { name: '—', score: 0 };
  const norm = chroma.map((v) => v / maxC);

  let best: ChordTemplate | null = null;
  let bestScore = -1;
  for (const t of TEMPLATES) {
    const score = cosineSim(norm, t.vec);
    if (score > bestScore) {
      bestScore = score;
      best = t;
    }
  }
  return { name: best ? best.name : '—', score: Math.min(1, Math.max(0, bestScore)) };
}

// Accumulate FFT bin magnitudes into a 12-bin pitch-class (chroma) vector.
function chromaFromFreqData(freqData: Float32Array, binHz: number): number[] {
  const chroma = new Array(12).fill(0);
  for (let i = 1; i < freqData.length; i++) {
    const freq = i * binHz;
    if (freq < MIN_FREQ || freq > 5000) continue;
    const db = freqData[i];
    const magnitude = Math.pow(10, db / 20);
    const midi = 69 + 12 * Math.log2(freq / 440);
    const pc = ((Math.round(midi) % 12) + 12) % 12;
    chroma[pc] += magnitude;
  }
  return chroma;
}

const MIN_FREQ = 55; // ~A1, chroma detection floor

// Spectrum display range, aligned to octave boundaries (C2 .. C7) so the
// log-frequency axis lands cleanly on note gridlines.
const SPEC_MIN_FREQ = 65.41; // C2
const SPEC_MAX_FREQ = 2093.0; // C7
const SPEC_MIN_MIDI = 36; // C2
const SPEC_MAX_MIDI = 96; // C7
const HISTORY_LEN = 8;

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

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// ---------- file-based chord sequence detection ----------

type ChordSegment = { start: number; end: number; name: string; confidence: number };

const FILE_HOP = 0.25; // seconds between analysis windows

// Smooth per-window raw detections with a majority-vote sliding window (matches the
// live-mic smoothing), then collapse consecutive identical chords into segments.
function smoothAndSegment(
  raw: { time: number; name: string; score: number }[],
  hop: number,
): ChordSegment[] {
  const smoothed = raw.map((_, i) => {
    const windowSlice = raw.slice(Math.max(0, i - HISTORY_LEN + 1), i + 1);
    const counts: Record<string, number> = {};
    windowSlice.forEach((r) => (counts[r.name] = (counts[r.name] || 0) + 1));

    let winner = windowSlice[windowSlice.length - 1].name;
    let winnerCount = 0;
    for (const name in counts) {
      if (counts[name] > winnerCount) {
        winnerCount = counts[name];
        winner = name;
      }
    }
    const avgScore =
      windowSlice.filter((r) => r.name === winner).reduce((a, b) => a + b.score, 0) / winnerCount;

    return { time: raw[i].time, name: winner, score: avgScore };
  });

  const segments: ChordSegment[] = [];
  for (const point of smoothed) {
    const last = segments[segments.length - 1];
    if (last && last.name === point.name) {
      last.end = point.time + hop;
      last.confidence = (last.confidence + point.score) / 2;
    } else {
      segments.push({ start: point.time, end: point.time + hop, name: point.name, confidence: point.score });
    }
  }
  return segments;
}

async function detectChordSequence(
  file: File,
  onProgress: (t: number) => void,
): Promise<ChordSegment[]> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextCtor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

  const decodeCtx = new AudioContextCtor();
  let audioBuffer: AudioBuffer;
  try {
    audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
  } finally {
    await decodeCtx.close();
  }

  const offlineCtx = new OfflineAudioContext(
    audioBuffer.numberOfChannels,
    audioBuffer.length,
    audioBuffer.sampleRate,
  );
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  const analyser = offlineCtx.createAnalyser();
  analyser.fftSize = 8192;
  analyser.smoothingTimeConstant = 0.2;
  source.connect(analyser);
  analyser.connect(offlineCtx.destination);
  source.start(0);

  const duration = audioBuffer.duration;
  const binHz = audioBuffer.sampleRate / analyser.fftSize;
  const raw: { time: number; name: string; score: number }[] = [];

  const suspendTimes: number[] = [];
  for (let t = FILE_HOP; t < duration; t += FILE_HOP) suspendTimes.push(t);

  const analyzeAt = (time: number) => {
    const freqData = new Float32Array(analyser.frequencyBinCount);
    analyser.getFloatFrequencyData(freqData);
    const chroma = chromaFromFreqData(freqData, binHz);
    const { name, score } = classifyChroma(chroma);
    raw.push({ time, name, score });
    onProgress(time / duration);
  };

  // Step the offline render forward one hop at a time, sampling the analyser at
  // each stop. Chained so each suspend point is registered before rendering reaches it.
  const stepping = suspendTimes.reduce(
    (promise, t) => promise.then(() => offlineCtx.suspend(t)).then(() => {
      analyzeAt(t);
      return offlineCtx.resume();
    }),
    Promise.resolve(),
  );

  await Promise.all([stepping, offlineCtx.startRendering()]);

  onProgress(1);
  return smoothAndSegment(raw, FILE_HOP);
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
    const { name, score } = classifyChroma(chroma);
    if (name === '—') {
      setChordName('—');
      setConfidence(0);
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

    setChordName(winner);
    setConfidence(Math.round(Math.min(1, Math.max(0, avgScore)) * 100));
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

      const chroma = chromaFromFreqData(freqData, binHz);
      detectChord(chroma);
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
      analyser.fftSize = 8192;
      analyser.smoothingTimeConstant = 0.55;
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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeProgress, setAnalyzeProgress] = useState(0);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [chordSequence, setChordSequence] = useState<ChordSegment[]>([]);

  useEffect(() => {
    return () => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
    };
  }, [fileUrl]);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setSelectedFile(file);
      setChordSequence([]);
      setAnalyzeError(null);
      setAnalyzeProgress(0);
      setFileUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
    },
    [],
  );

  const analyzeFile = useCallback(async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setAnalyzeError(null);
    setAnalyzeProgress(0);
    setChordSequence([]);
    try {
      const segments = await detectChordSequence(selectedFile, setAnalyzeProgress);
      setChordSequence(segments);
    } catch (err) {
      setAnalyzeError((err as Error).message);
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile]);

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
            Upload an audio recording
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="text-[14px] text-[#6e6e73] file:mr-4 file:border-0 file:bg-black/5 file:text-[#1d1d1f] file:rounded-full file:px-5 file:py-2.5 file:text-[14px] file:font-medium file:cursor-pointer file:transition-colors hover:file:bg-black/10"
            />
            <button
              onClick={analyzeFile}
              disabled={!selectedFile || isAnalyzing}
              className="bg-[#0071e3] text-white text-[15px] font-medium px-6 py-2.5 rounded-full shadow-[0_2px_8px_rgba(0,113,227,0.35)] transition-all hover:bg-[#0077ed] hover:shadow-[0_4px_14px_rgba(0,113,227,0.45)] active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
            >
              {isAnalyzing ? 'Analyzing…' : 'Detect chords'}
            </button>
          </div>

          {fileUrl && (
            <audio controls src={fileUrl} className="w-full mt-6 h-10">
              Your browser does not support audio playback.
            </audio>
          )}

          {isAnalyzing && (
            <div className="mt-5">
              <div className="w-full h-1.5 bg-black/8 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#0071e3] transition-all duration-150"
                  style={{ width: `${Math.round(analyzeProgress * 100)}%` }}
                />
              </div>
              <div className="text-[13px] text-[#6e6e73] mt-2">
                {Math.round(analyzeProgress * 100)}%
              </div>
            </div>
          )}

          {analyzeError && (
            <div className="text-[13px] text-[#ff3b30] mt-4">Error: {analyzeError}</div>
          )}

          {!isAnalyzing && chordSequence.length > 0 && (
            <div className="mt-7">
              <div className="text-[13px] font-semibold uppercase tracking-wide text-[#6e6e73] mb-3">
                Detected chord sequence
              </div>
              <div className="max-h-96 overflow-y-auto rounded-2xl bg-[#f5f5f7] divide-y divide-black/5">
                {chordSequence.map((seg, i) => (
                  <div key={i} className="flex items-center gap-4 px-4 py-3">
                    <span className="text-[13px] text-[#6e6e73] w-24 shrink-0 tabular-nums">
                      {formatTime(seg.start)} – {formatTime(seg.end)}
                    </span>
                    <span className="text-xl font-semibold text-[#0071e3] w-14 shrink-0">
                      {seg.name}
                    </span>
                    <div className="flex-1 h-1.5 bg-black/8 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#0071e3]"
                        style={{ width: `${Math.round(seg.confidence * 100)}%` }}
                      />
                    </div>
                    <span className="text-[13px] text-[#6e6e73] w-10 text-right shrink-0 tabular-nums">
                      {Math.round(seg.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
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
