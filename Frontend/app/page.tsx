import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Guitar Chord Recognition",
  description:
    "Explore the diatonic chords of any key, or detect chords live or from an uploaded recording.",
};

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0b1210] text-[#d7e6df] font-mono flex flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-2xl text-center">
        <div className="text-[11px] tracking-[3px] uppercase text-[#4ff2a3]/80">
          guitar toolkit
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold mt-2">
          Chord <span className="text-[#4ff2a3]">Recognition</span>
        </h1>
        <p className="text-[13px] text-[#6e8d82] mt-3 max-w-md mx-auto">
          Learn the diatonic chords of any key and mode, or detect the chords
          being played from your microphone or an uploaded recording.
        </p>

        <div className="grid sm:grid-cols-2 gap-4 mt-10 text-left">
          <Link
            href="/chordexplorer"
            className="group block border border-[#1c2e29] bg-[#101c19] rounded-md p-5 hover:border-[#4ff2a3] transition-colors"
          >
            <div className="text-[11px] tracking-[2px] uppercase text-[#6e8d82] group-hover:text-[#4ff2a3]/80">
              Explore
            </div>
            <div className="text-lg font-bold mt-1">Chord Explorer</div>
            <p className="text-[13px] text-[#6e8d82] mt-2">
              Browse the diatonic chords of any key and mode.
            </p>
          </Link>

          <Link
            href="/chorddetector"
            className="group block border border-[#1c2e29] bg-[#101c19] rounded-md p-5 hover:border-[#4ff2a3] transition-colors"
          >
            <div className="text-[11px] tracking-[2px] uppercase text-[#6e8d82] group-hover:text-[#4ff2a3]/80">
              Detect
            </div>
            <div className="text-lg font-bold mt-1">Chord Detector</div>
            <p className="text-[13px] text-[#6e8d82] mt-2">
              Identify chords live from your microphone, or upload an audio
              file to detect the full chord sequence.
            </p>
          </Link>
        </div>
      </div>
    </main>
  );
}
