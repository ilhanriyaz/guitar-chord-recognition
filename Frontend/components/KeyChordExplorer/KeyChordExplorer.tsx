'use client';

import { useMemo, useState } from 'react';
import { getDiatonicChords, MODES, NOTES, Quality } from '@/lib/theory';

// ---------- component ----------

const QUALITY_COLOR: Record<Quality, string> = {
  maj: '#e8a655',
  min: '#4fb3ac',
  dim: '#c9605c',
  aug: '#8f7fd9',
};

export default function KeyChordExplorer() {
  const [rootIndex, setRootIndex] = useState(0);
  const [modeIdx, setModeIdx] = useState(0);
  const [showSevenths, setShowSevenths] = useState(false);

  const chords = useMemo(() => getDiatonicChords(rootIndex, modeIdx), [rootIndex, modeIdx]);
  const keyLabel = `${NOTES[rootIndex]} ${MODES[modeIdx].name}`;

  return (
    <div
      style={{
        background: '#141210',
        color: '#f2ede4',
        fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        minHeight: '100vh',
        width: '100%',
        padding: '3rem 1.5rem 4rem',
      }}
    >
      <div style={{ maxWidth: 960, margin: '0 auto' }}>
        <div style={{ marginBottom: '2.5rem', textAlign: 'center' }}>
          <p style={{ fontSize: 13, letterSpacing: 1.5, textTransform: 'uppercase', color: '#8a8378', margin: 0 }}>
            Diatonic chord finder
          </p>
          <h1 style={{ fontSize: 32, fontWeight: 500, margin: '8px 0 0' }}>{keyLabel}</h1>
        </div>

        {/* chromatic key wheel */}
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
          <svg width="220" height="220" viewBox="0 0 220 220">
            <circle cx="110" cy="110" r="95" fill="none" stroke="#332e28" strokeWidth="1" />
            {NOTES.map((note, i) => {
              const angle = (i / 12) * 2 * Math.PI - Math.PI / 2;
              const x = 110 + 95 * Math.cos(angle);
              const y = 110 + 95 * Math.sin(angle);
              const active = i === rootIndex;
              return (
                <g key={note} onClick={() => setRootIndex(i)} style={{ cursor: 'pointer' }}>
                  <circle cx={x} cy={y} r={active ? 18 : 15} fill={active ? '#e8a655' : '#211e1a'} stroke="#3d372f" strokeWidth="1" />
                  <text
                    x={x}
                    y={y}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize="12"
                    fontWeight={active ? 600 : 400}
                    fill={active ? '#141210' : '#cfc7ba'}
                  >
                    {note}
                  </text>
                </g>
              );
            })}
            <text x="110" y="115" textAnchor="middle" fontSize="13" fill="#8a8378">
              key
            </text>
          </svg>
        </div>

        {/* mode selector */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', marginBottom: '0.75rem' }}>
          {MODES.map((m, i) => (
            <button
              key={m.name}
              onClick={() => setModeIdx(i)}
              style={{
                padding: '6px 12px',
                borderRadius: 8,
                fontSize: 12,
                border: '1px solid #3d372f',
                background: i === modeIdx ? '#e8a655' : 'transparent',
                color: i === modeIdx ? '#141210' : '#cfc7ba',
                cursor: 'pointer',
              }}
            >
              {m.name}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <label style={{ fontSize: 13, color: '#a89f92', display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" checked={showSevenths} onChange={(e) => setShowSevenths(e.target.checked)} />
            show seventh chords
          </label>
        </div>

        {/* chord grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
          {chords.map((c) => (
            <div
              key={c.degree}
              style={{
                background: '#1c1a16',
                border: `1px solid ${QUALITY_COLOR[c.quality]}44`,
                borderRadius: 10,
                padding: '14px 16px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: 18, fontWeight: 600, color: QUALITY_COLOR[c.quality] }}>{c.roman}</span>
                <span style={{ fontSize: 11, color: '#8a8378' }}>{c.qualityLabel}</span>
              </div>
              <p style={{ fontSize: 16, fontWeight: 500, margin: '4px 0 2px' }}>
                {c.root}
                {c.quality === 'min' ? 'm' : c.quality === 'dim' ? '\u00B0' : c.quality === 'aug' ? '+' : ''}
                {showSevenths ? c.seventhLabel.replace(/^(maj|min|dom)/, '') : ''}
              </p>
              <p style={{ fontSize: 11, color: '#8a8378', margin: 0, fontFamily: 'ui-monospace, monospace' }}>
                {(showSevenths ? c.seventhNotes : c.triadNotes).join(' \u2013 ')}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
