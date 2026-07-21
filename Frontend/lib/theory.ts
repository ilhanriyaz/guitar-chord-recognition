// ---------- music theory core ----------

const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

const MODES = [
  { name: 'Ionian (major)', intervals: [0, 2, 4, 5, 7, 9, 11] },
  { name: 'Dorian', intervals: [0, 2, 3, 5, 7, 9, 10] },
  { name: 'Phrygian', intervals: [0, 1, 3, 5, 7, 8, 10] },
  { name: 'Lydian', intervals: [0, 2, 4, 6, 7, 9, 11] },
  { name: 'Mixolydian', intervals: [0, 2, 4, 5, 7, 9, 10] },
  { name: 'Aeolian (natural minor)', intervals: [0, 2, 3, 5, 7, 8, 10] },
  { name: 'Locrian', intervals: [0, 1, 3, 5, 6, 8, 10] },
] as const;

type Quality = 'maj' | 'min' | 'dim' | 'aug';

interface ChordInfo {
  degree: number;
  roman: string;
  root: string;
  quality: Quality;
  qualityLabel: string;
  seventhLabel: string;
  triadNotes: string[];
  seventhNotes: string[];
}

const ROMANS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'];

function noteName(rootIndex: number, semitones: number): string {
  return NOTES[(rootIndex + semitones) % 12];
}

function buildScale(rootIndex: number, intervals: readonly number[]): string[] {
  return intervals.map((iv) => noteName(rootIndex, iv));
}

function semitoneDistance(a: string, b: string): number {
  const ai = NOTES.indexOf(a as (typeof NOTES)[number]);
  const bi = NOTES.indexOf(b as (typeof NOTES)[number]);
  return (bi - ai + 12) % 12;
}

function qualityFromIntervals(i1: number, i2: number): Quality {
  if (i1 === 4 && i2 === 3) return 'maj';
  if (i1 === 3 && i2 === 4) return 'min';
  if (i1 === 3 && i2 === 3) return 'dim';
  return 'aug';
}

function getDiatonicChords(rootIndex: number, modeIdx: number): ChordInfo[] {
  const scale = buildScale(rootIndex, MODES[modeIdx].intervals);

  return scale.map((root, i) => {
    const third = scale[(i + 2) % 7];
    const fifth = scale[(i + 4) % 7];
    const seventh = scale[(i + 6) % 7];

    const i1 = semitoneDistance(root, third);
    const i2 = semitoneDistance(third, fifth);
    const i3 = semitoneDistance(fifth, seventh);
    const quality = qualityFromIntervals(i1, i2);

    let roman = ROMANS[i];
    if (quality === 'min' || quality === 'dim') roman = roman.toLowerCase();
    if (quality === 'dim') roman += '\u00B0';
    if (quality === 'aug') roman += '+';

    const qualityLabel =
      quality === 'maj' ? 'major' : quality === 'min' ? 'minor' : quality === 'dim' ? 'diminished' : 'augmented';

    let seventhLabel = '';
    if (quality === 'maj') seventhLabel = i3 === 4 ? 'maj7' : 'dom7';
    else if (quality === 'min') seventhLabel = i3 === 3 ? 'min7' : 'minMaj7';
    else if (quality === 'dim') seventhLabel = i3 === 3 ? 'm7\u266D5' : 'dim7';
    else seventhLabel = 'aug7';

    return {
      degree: i + 1,
      roman,
      root,
      quality,
      qualityLabel,
      seventhLabel,
      triadNotes: [root, third, fifth],
      seventhNotes: [root, third, fifth, seventh],
    };
  });
}

export { NOTES, MODES, getDiatonicChords };
export type { Quality };