import type { Metadata } from "next";
import KeyChordExplorer from "@/components/KeyChordExplorer";

export const metadata: Metadata = {
  title: "Chord Explorer",
  description: "Explore the diatonic chords of any key and mode.",
};

export default function ChordExplorerPage() {
  return <KeyChordExplorer />;
}
