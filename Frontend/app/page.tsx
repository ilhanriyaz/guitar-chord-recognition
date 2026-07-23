import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Guitar Chord Recognition",
  description:
    "Explore the diatonic chords of any key, or detect chords live or from an uploaded recording.",
};

export default function Home() {
  return (
    <main className="min-h-screen bg-[#f5f5f7] text-[#1d1d1f] flex flex-col items-center justify-center px-6 py-20">
      <div className="w-full max-w-3xl text-center">
        <div className="text-[13px] font-medium tracking-wide uppercase text-[#0071e3]">
          Guitar Toolkit
        </div>
        <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight mt-3">
          Chord Recognition
        </h1>
        <p className="text-[17px] leading-relaxed text-[#6e6e73] mt-4 max-w-xl mx-auto">
          Learn the diatonic chords of any key and mode, or detect the chords
          being played from your microphone or an uploaded recording.
        </p>

        <div className="grid sm:grid-cols-2 gap-5 mt-12 text-left">
          <Link
            href="/chordexplorer"
            className="group block rounded-3xl bg-white p-7 shadow-[0_2px_16px_rgba(0,0,0,0.06)] ring-1 ring-black/5 transition-all duration-200 hover:-translate-y-1 hover:shadow-[0_12px_32px_rgba(0,0,0,0.10)]"
          >
            <div className="text-[12px] font-semibold tracking-wide uppercase text-[#0071e3]">
              Explore
            </div>
            <div className="text-xl font-semibold mt-2">Chord Explorer</div>
            <p className="text-[15px] leading-relaxed text-[#6e6e73] mt-2">
              Browse the diatonic chords of any key and mode.
            </p>
            <div className="mt-5 inline-flex items-center gap-1.5 text-[15px] font-medium text-[#0071e3]">
              Open
              <span className="transition-transform duration-200 group-hover:translate-x-1">
                &rarr;
              </span>
            </div>
          </Link>

          <Link
            href="/chorddetector"
            className="group block rounded-3xl bg-white p-7 shadow-[0_2px_16px_rgba(0,0,0,0.06)] ring-1 ring-black/5 transition-all duration-200 hover:-translate-y-1 hover:shadow-[0_12px_32px_rgba(0,0,0,0.10)]"
          >
            <div className="text-[12px] font-semibold tracking-wide uppercase text-[#0071e3]">
              Detect
            </div>
            <div className="text-xl font-semibold mt-2">Chord Detector</div>
            <p className="text-[15px] leading-relaxed text-[#6e6e73] mt-2">
              Identify chords live from your microphone, or upload an audio file
              to detect the full chord sequence.
            </p>
            <div className="mt-5 inline-flex items-center gap-1.5 text-[15px] font-medium text-[#0071e3]">
              Open
              <span className="transition-transform duration-200 group-hover:translate-x-1">
                &rarr;
              </span>
            </div>
          </Link>
        </div>
      </div>
    </main>
  );
}
