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

const MIN_FREQ = 55; // ~A1
const MAX_FREQ = 1000;
const HISTORY_LEN = 8;

function freqToX(freq: number, w: number): number {
  const logMin = Math.log2(MIN_FREQ);
  const logMax = Math.log2(MAX_FREQ);
  const t = (Math.log2(freq) - logMin) / (logMax - logMin);
  return t * w;
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
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = '#173028';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.strokeStyle = '#4ff2a3';
    ctx.lineWidth = 1.5;
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

      const freqData = new Float32Array(analyser.frequencyBinCount);
      analyser.getFloatFrequencyData(freqData);

      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // pitch-class background shading + labels
      for (let m = 24; m < 108; m++) {
        const f0 = 440 * Math.pow(2, (m - 69) / 12);
        const f1 = 440 * Math.pow(2, (m + 1 - 69) / 12);
        if (f1 < MIN_FREQ || f0 > MAX_FREQ) continue;
        const x0 = Math.max(0, freqToX(Math.max(f0, MIN_FREQ), w));
        const x1 = Math.min(w, freqToX(Math.min(f1, MAX_FREQ), w));
        const pc = ((m % 12) + 12) % 12;
        ctx.fillStyle = pc === 0 ? 'rgba(79,242,163,0.10)' : 'rgba(79,242,163,0.03)';
        ctx.fillRect(x0, 0, x1 - x0, h - 24);
        if (x1 - x0 > 10) {
          ctx.fillStyle = '#3a5b50';
          ctx.font = '9px monospace';
          ctx.fillText(NOTE_NAMES[pc], x0 + 2, h - 30);
        }
      }

      const minDb = -100;
      const maxDb = -20;
      const sampleRate = audioCtx.sampleRate;
      const binHz = sampleRate / analyser.fftSize;

      ctx.beginPath();
      ctx.strokeStyle = '#4ff2a3';
      ctx.lineWidth = 1.5;
      let first = true;

      for (let i = 1; i < freqData.length; i++) {
        const freq = i * binHz;
        if (freq < MIN_FREQ || freq > MAX_FREQ) continue;

        const db = freqData[i];
        const norm = Math.min(1, Math.max(0, (db - minDb) / (maxDb - minDb)));

        const x = freqToX(freq, w);
        const y = h - 24 - norm * (h - 24);
        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      ctx.fillStyle = '#6e8d82';
      ctx.font = '10px monospace';
      [110, 220, 440, 880].forEach((f) => {
        if (f >= MIN_FREQ && f <= MAX_FREQ) {
          const x = freqToX(f, w);
          ctx.fillText(`${f}Hz`, x - 14, h - 8);
        }
      });

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
    <main className="min-h-screen bg-[#0b1210] text-[#d7e6df] font-mono flex flex-col items-center px-4 py-8">
      <header className="w-full max-w-3xl mb-5 border-b border-[#1c2e29] pb-3">
        <div className="text-[11px] tracking-[3px] uppercase text-[#4ff2a3]/80">
          DSP / instrument
        </div>
        <h1 className="text-2xl font-bold mt-1">
          Chord <span className="text-[#4ff2a3]">Scope</span>
        </h1>
        <p className="text-[13px] text-[#6e8d82] mt-1.5">
          Real-time waveform + FFT spectrum + chroma-based chord detection.
        </p>
      </header>

      <div className="w-full max-w-3xl flex gap-2 mb-4">
        <button
          onClick={() => {
            if (listening) stop();
            setMode('file');
          }}
          className={`text-[12px] tracking-[2px] uppercase px-4 py-2 rounded border transition-colors ${
            mode === 'file'
              ? 'border-[#4ff2a3] text-[#4ff2a3] bg-[#4ff2a3]/10'
              : 'border-[#1c2e29] text-[#6e8d82] hover:border-[#2c433c]'
          }`}
        >
          Upload file
        </button>
        <button
          onClick={() => setMode('mic')}
          className={`text-[12px] tracking-[2px] uppercase px-4 py-2 rounded border transition-colors ${
            mode === 'mic'
              ? 'border-[#4ff2a3] text-[#4ff2a3] bg-[#4ff2a3]/10'
              : 'border-[#1c2e29] text-[#6e8d82] hover:border-[#2c433c]'
          }`}
        >
          Microphone
        </button>
      </div>

      {mode === 'file' && (
        <section className="w-full max-w-3xl bg-[#101c19] border border-[#1c2e29] rounded-md p-4">
          <div className="text-[11px] tracking-[2px] uppercase text-[#6e8d82] mb-2">
            Upload an audio recording
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="text-[13px] text-[#6e8d82] file:mr-3 file:border file:border-[#1c2e29] file:bg-[#0b1210] file:text-[#4ff2a3] file:rounded file:px-3 file:py-1.5 file:text-[12px] file:uppercase file:tracking-[1px] file:cursor-pointer"
            />
            <button
              onClick={analyzeFile}
              disabled={!selectedFile || isAnalyzing}
              className="border border-[#4ff2a3] text-[#4ff2a3] text-[13px] tracking-[2px] uppercase px-5 py-2.5 rounded hover:bg-[#4ff2a3] hover:text-[#08110e] transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-[#4ff2a3]"
            >
              {isAnalyzing ? 'Analyzing…' : 'Detect chords'}
            </button>
          </div>

          {fileUrl && (
            <audio controls src={fileUrl} className="w-full mt-4 h-10">
              Your browser does not support audio playback.
            </audio>
          )}

          {isAnalyzing && (
            <div className="mt-4">
              <div className="w-full h-1.5 bg-[#1c2e29] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#4ff2a3] transition-all duration-150"
                  style={{ width: `${Math.round(analyzeProgress * 100)}%` }}
                />
              </div>
              <div className="text-xs text-[#6e8d82] mt-1">
                {Math.round(analyzeProgress * 100)}%
              </div>
            </div>
          )}

          {analyzeError && (
            <div className="text-xs text-[#ff6b6b] mt-3">Error: {analyzeError}</div>
          )}

          {!isAnalyzing && chordSequence.length > 0 && (
            <div className="mt-5">
              <div className="text-[11px] tracking-[2px] uppercase text-[#6e8d82] mb-2">
                Detected chord sequence
              </div>
              <div className="max-h-96 overflow-y-auto rounded border border-[#1c2e29] divide-y divide-[#1c2e29]">
                {chordSequence.map((seg, i) => (
                  <div key={i} className="flex items-center gap-4 px-3 py-2">
                    <span className="text-xs text-[#6e8d82] w-24 shrink-0">
                      {formatTime(seg.start)} – {formatTime(seg.end)}
                    </span>
                    <span className="text-lg font-bold text-[#ffb454] w-14 shrink-0">
                      {seg.name}
                    </span>
                    <div className="flex-1 h-1.5 bg-[#1c2e29] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#4ff2a3]"
                        style={{ width: `${Math.round(seg.confidence * 100)}%` }}
                      />
                    </div>
                    <span className="text-xs text-[#6e8d82] w-10 text-right shrink-0">
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
        <section className="w-full max-w-3xl bg-[#101c19] border border-[#1c2e29] rounded-md p-4">
          <div className="flex items-center gap-3 mb-4 flex-wrap">
            <button
              onClick={listening ? stop : start}
              className="border border-[#4ff2a3] text-[#4ff2a3] text-[13px] tracking-[2px] uppercase px-5 py-2.5 rounded hover:bg-[#4ff2a3] hover:text-[#08110e] transition-colors"
            >
              {listening ? 'Stop' : 'Start listening'}
            </button>
            <span
              className={`w-2 h-2 rounded-full ${
                listening ? 'bg-[#4ff2a3] shadow-[0_0_8px_#4ff2a3]' : 'bg-[#6e8d82]'
              }`}
            />
            <span className="text-xs text-[#6e8d82]">{statusText}</span>
          </div>

          <div className="text-[11px] tracking-[2px] uppercase text-[#6e8d82] mb-2">
            Waveform — time domain
          </div>
          <canvas
            ref={waveCanvasRef}
            width={900}
            height={140}
            className="w-full bg-[#08110e] rounded-sm"
          />

          <div className="text-[11px] tracking-[2px] uppercase text-[#6e8d82] mt-5 mb-2">
            Spectrum — frequency domain, log scale
          </div>
          <canvas
            ref={specCanvasRef}
            width={900}
            height={260}
            className="w-full bg-[#08110e] rounded-sm"
          />

          <div className="flex items-baseline gap-6 mt-6 flex-wrap">
            <div className="text-6xl font-bold text-[#ffb454] min-w-[160px] drop-shadow-[0_0_18px_rgba(255,180,84,0.35)]">
              {chordName}
            </div>
            <div className="text-xs text-[#6e8d82] leading-relaxed">
              confidence
              <div className="w-40 h-1.5 bg-[#1c2e29] rounded-full overflow-hidden mt-1">
                <div
                  className="h-full bg-[#4ff2a3] transition-all duration-100"
                  style={{ width: `${confidence}%` }}
                />
              </div>
              <span>{confidence}%</span>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
