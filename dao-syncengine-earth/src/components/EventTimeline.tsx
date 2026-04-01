import { useState, type CSSProperties } from 'react';
import { EventKind, type SimEvent, DRIVING_VARS, STATE_VARS } from '../lib/event-engine';

interface EventTimelineProps {
  events: SimEvent[];
  onChange: (events: SimEvent[]) => void;
  tEnd: number;
}

// --- Variable range config ---

const DRIVING_RANGES: Record<string, { min: number; max: number; step: number }> = {
  H:      { min: 0, max: 1, step: 0.05 },
  E:      { min: 0, max: 1, step: 0.05 },
  S:      { min: 0, max: 3, step: 0.1 },
  N:      { min: 0, max: 1, step: 0.05 },
  G:      { min: 0, max: 1, step: 0.05 },
  Rc:     { min: 0, max: 1, step: 0.05 },
  D_jing: { min: 0, max: 2, step: 0.05 },
  R_qi:   { min: 0, max: 1, step: 0.05 },
  D_qi:   { min: 0, max: 1, step: 0.05 },
};

const DRIVING_LABELS: Record<string, string> = {
  H: 'Heart Engagement', E: 'Embodiment', S: 'Stimulation',
  N: 'Novelty', G: 'Generative Fraction', Rc: 'Reciprocity',
  D_jing: 'Discharge Rate', R_qi: 'Env. Qi Support', D_qi: 'Qi Drain',
};

const KIND_COLORS: Record<string, string> = {
  [EventKind.SET]: '#4488ff',
  [EventKind.PULSE]: '#ff4444',
  [EventKind.RAMP]: '#44cc88',
};

// --- Styles ---

const c: Record<string, CSSProperties> = {
  wrapper: { marginBottom: 20 },
  sectionLabel: {
    fontSize: 13, fontWeight: 600, color: '#999',
    textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8,
  },
  timelineBar: {
    position: 'relative', height: 48, background: '#14141e',
    borderRadius: 6, border: '1px solid #2a2a35', marginBottom: 16, overflow: 'visible',
  },
  marker: {
    position: 'absolute', top: '50%', transform: 'translate(-50%, -50%)',
    cursor: 'pointer', zIndex: 2, fontSize: 11, fontWeight: 700,
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
  },
  markerDot: {
    width: 12, height: 12, borderRadius: '50%',
  },
  markerDiamond: {
    width: 10, height: 10, transform: 'rotate(45deg)',
  },
  rampBar: {
    position: 'absolute', top: '50%', height: 8, borderRadius: 4,
    transform: 'translateY(-50%)', opacity: 0.6,
  },
  markerLabel: {
    position: 'absolute', top: 38, fontSize: 9, color: '#888',
    whiteSpace: 'nowrap', transform: 'translateX(-50%)',
  },
  tickLabels: {
    display: 'flex', justifyContent: 'space-between',
    fontSize: 10, color: '#555', marginTop: -10, marginBottom: 12, padding: '0 2px',
  },
  eventList: { display: 'flex', flexDirection: 'column', gap: 6 },
  eventRow: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '8px 12px', background: '#14141e', borderRadius: 6,
    border: '1px solid #2a2a35', flexWrap: 'wrap',
  },
  dayInput: {
    width: 50, background: '#0d0d14', border: '1px solid #2a2a35',
    borderRadius: 4, color: '#e8ddd0', padding: '3px 6px', fontSize: 13,
    fontFamily: 'monospace', textAlign: 'center',
  },
  badge: {
    fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 10,
    textTransform: 'uppercase', letterSpacing: '0.05em',
  },
  labelInput: {
    flex: 1, minWidth: 120, background: '#0d0d14', border: '1px solid #2a2a35',
    borderRadius: 4, color: '#e8ddd0', padding: '4px 8px', fontSize: 13,
  },
  summary: { fontSize: 12, color: '#888', fontFamily: 'monospace' },
  deleteBtn: {
    background: 'transparent', border: 'none', color: '#666', cursor: 'pointer',
    fontSize: 18, padding: '0 4px', lineHeight: 1,
  },
  addBtn: {
    background: 'transparent', border: '1px dashed #2a2a35', color: '#888',
    padding: '8px 16px', borderRadius: 6, cursor: 'pointer', fontSize: 13,
    marginTop: 8, width: '100%', textAlign: 'center',
  },
  // Form styles
  form: {
    background: '#14141e', border: '1px solid #2a2a35', borderRadius: 8,
    padding: 16, marginTop: 8,
  },
  formRow: { display: 'flex', gap: 10, marginBottom: 10, flexWrap: 'wrap', alignItems: 'center' },
  formLabel: { fontSize: 12, color: '#999', width: 70, flexShrink: 0 },
  formInput: {
    background: '#0d0d14', border: '1px solid #2a2a35', borderRadius: 4,
    color: '#e8ddd0', padding: '4px 8px', fontSize: 13,
  },
  tabBar: { display: 'flex', gap: 4, marginBottom: 12 },
  tab: {
    padding: '6px 14px', borderRadius: 4, cursor: 'pointer', fontSize: 12,
    fontWeight: 600, border: '1px solid #2a2a35', background: 'transparent', color: '#888',
  },
  tabActive: {
    padding: '6px 14px', borderRadius: 4, cursor: 'pointer', fontSize: 12,
    fontWeight: 600, border: '1px solid #4169E1', background: '#4169E120', color: '#e8ddd0',
  },
  varRow: {
    display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexWrap: 'wrap',
  },
  checkbox: { accentColor: '#4169E1' },
  varLabel: { fontSize: 12, color: '#c8c0b8', width: 55, fontFamily: 'monospace' },
  slider: { flex: 1, minWidth: 80, accentColor: '#4169E1', height: 4 },
  numInput: {
    width: 60, background: '#0d0d14', border: '1px solid #2a2a35', borderRadius: 4,
    color: '#e8ddd0', padding: '3px 6px', fontSize: 12, fontFamily: 'monospace', textAlign: 'right',
  },
  presetBtn: {
    background: '#ff444420', border: '1px solid #ff444466', color: '#ff8888',
    padding: '4px 10px', borderRadius: 4, cursor: 'pointer', fontSize: 11, fontWeight: 600,
  },
  confirmBtn: {
    background: '#4169E1', border: 'none', color: '#fff', padding: '8px 20px',
    borderRadius: 4, cursor: 'pointer', fontSize: 13, fontWeight: 600, marginTop: 8,
  },
  cancelBtn: {
    background: 'transparent', border: '1px solid #2a2a35', color: '#888',
    padding: '8px 16px', borderRadius: 4, cursor: 'pointer', fontSize: 13, marginTop: 8, marginLeft: 8,
  },
};

function summarizeEvent(ev: SimEvent): string {
  if (ev.kind === EventKind.SET && ev.changes) {
    return Object.entries(ev.changes).map(([k, v]) => `${k}=${v}`).join(', ');
  }
  if (ev.kind === EventKind.PULSE && ev.deltas) {
    return Object.entries(ev.deltas).map(([k, v]) => `${k}: ${v > 0 ? '+' : ''}${v}`).join(', ');
  }
  if (ev.kind === EventKind.RAMP && ev.targets) {
    const vars = Object.entries(ev.targets).map(([k, v]) => `${k}\u2192${v}`).join(', ');
    return `${vars} over ${ev.duration ?? '?'}d`;
  }
  return '';
}

export default function EventTimeline({ events, onChange, tEnd }: EventTimelineProps) {
  const [adding, setAdding] = useState(false);
  const [formKind, setFormKind] = useState<EventKind>(EventKind.SET);
  const [formDay, setFormDay] = useState(0);
  const [formLabel, setFormLabel] = useState('');
  const [formDuration, setFormDuration] = useState(30);
  const [selectedVars, setSelectedVars] = useState<Record<string, boolean>>({});
  const [varValues, setVarValues] = useState<Record<string, number>>({});

  const sorted = [...events].sort((a, b) => a.time - b.time);

  const updateEvent = (idx: number, patch: Partial<SimEvent>) => {
    const original = sorted[idx];
    const updated = { ...original, ...patch };
    const newEvents = events.map((e) => (e === original ? updated : e));
    onChange(newEvents);
  };

  const deleteEvent = (idx: number) => {
    const original = sorted[idx];
    onChange(events.filter((e) => e !== original));
  };

  const resetForm = () => {
    setFormDay(0);
    setFormLabel('');
    setFormDuration(30);
    setSelectedVars({});
    setVarValues({});
  };

  const addEvent = () => {
    const ev: SimEvent = { time: formDay, kind: formKind, label: formLabel || 'Event' };
    const activeVars = Object.entries(selectedVars).filter(([, on]) => on).map(([k]) => k);

    if (formKind === EventKind.SET) {
      ev.changes = {};
      for (const v of activeVars) ev.changes[v] = varValues[v] ?? 0;
    } else if (formKind === EventKind.PULSE) {
      ev.deltas = {};
      for (const v of activeVars) ev.deltas[v] = varValues[v] ?? 0;
    } else {
      ev.targets = {};
      for (const v of activeVars) ev.targets[v] = varValues[v] ?? 0;
      ev.duration = formDuration;
    }
    onChange([...events, ev]);
    setAdding(false);
    resetForm();
  };

  const applyEjaculationPreset = () => {
    setSelectedVars({ phi_jing: true, phi_qi: true });
    setVarValues({ phi_jing: -1.5, phi_qi: -0.3 });
  };

  const varsForKind = formKind === EventKind.PULSE
    ? STATE_VARS as readonly string[]
    : DRIVING_VARS as readonly string[];

  const rangeFor = (v: string) => {
    if (formKind === EventKind.PULSE) return { min: -10, max: 10, step: 0.1 };
    return DRIVING_RANGES[v] ?? { min: 0, max: 1, step: 0.05 };
  };

  return (
    <div style={c.wrapper}>
      <div style={c.sectionLabel}>Event Timeline</div>

      {/* Visual timeline bar */}
      <div style={c.timelineBar}>
        {sorted.map((ev, i) => {
          const pct = (ev.time / tEnd) * 100;
          const col = KIND_COLORS[ev.kind] ?? '#888';

          if (ev.kind === EventKind.RAMP && ev.duration) {
            const endPct = Math.min(((ev.time + ev.duration) / tEnd) * 100, 100);
            return (
              <div key={i}>
                <div style={{ ...c.rampBar, left: `${pct}%`, width: `${endPct - pct}%`, background: col }} />
                <div style={{ ...c.marker, left: `${pct}%` }}>
                  <div style={{ ...c.markerDot, background: col }} />
                </div>
                <div style={{ ...c.markerLabel, left: `${pct}%` }}>{ev.label}</div>
              </div>
            );
          }

          const shape = ev.kind === EventKind.SET
            ? <div style={{ ...c.markerDiamond, background: col }} />
            : <div style={{ ...c.markerDot, background: col }} />;

          return (
            <div key={i}>
              <div style={{ ...c.marker, left: `${pct}%` }}>{shape}</div>
              <div style={{ ...c.markerLabel, left: `${pct}%` }}>{ev.label}</div>
            </div>
          );
        })}
      </div>
      <div style={c.tickLabels}>
        <span>Day 0</span>
        <span>Day {Math.round(tEnd / 4)}</span>
        <span>Day {Math.round(tEnd / 2)}</span>
        <span>Day {Math.round((tEnd * 3) / 4)}</span>
        <span>Day {tEnd}</span>
      </div>

      {/* Event list */}
      <div style={c.eventList}>
        {sorted.map((ev, i) => (
          <div key={i} style={c.eventRow}>
            <span style={{ fontSize: 11, color: '#666' }}>Day</span>
            <input
              type="number"
              style={c.dayInput as CSSProperties}
              value={ev.time}
              min={0}
              max={tEnd}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v)) updateEvent(i, { time: Math.max(0, Math.min(tEnd, v)) });
              }}
            />
            <span
              style={{
                ...c.badge,
                background: KIND_COLORS[ev.kind] + '22',
                color: KIND_COLORS[ev.kind],
                border: `1px solid ${KIND_COLORS[ev.kind]}44`,
              }}
            >
              {ev.kind.toUpperCase()}
            </span>
            <input
              type="text"
              style={c.labelInput as CSSProperties}
              value={ev.label}
              onChange={(e) => updateEvent(i, { label: e.target.value })}
            />
            <span style={c.summary}>{summarizeEvent(ev)}</span>
            <button
              style={c.deleteBtn}
              onClick={() => deleteEvent(i)}
              onMouseEnter={(e) => { (e.target as HTMLElement).style.color = '#ff4444'; }}
              onMouseLeave={(e) => { (e.target as HTMLElement).style.color = '#666'; }}
            >
              &times;
            </button>
          </div>
        ))}
      </div>

      {/* Add event */}
      {!adding ? (
        <button
          style={c.addBtn}
          onClick={() => { resetForm(); setAdding(true); }}
          onMouseEnter={(e) => { (e.target as HTMLElement).style.borderColor = '#4169E1'; }}
          onMouseLeave={(e) => { (e.target as HTMLElement).style.borderColor = '#2a2a35'; }}
        >
          + Add Event
        </button>
      ) : (
        <div style={c.form}>
          {/* Kind tabs */}
          <div style={c.tabBar}>
            {([EventKind.SET, EventKind.PULSE, EventKind.RAMP] as const).map((k) => (
              <button
                key={k}
                style={formKind === k ? c.tabActive : c.tab}
                onClick={() => { setFormKind(k); setSelectedVars({}); setVarValues({}); }}
              >
                {k.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Day + label */}
          <div style={c.formRow}>
            <span style={c.formLabel}>Day</span>
            <input
              type="number"
              min={0}
              max={tEnd}
              value={formDay}
              onChange={(e) => setFormDay(Math.max(0, parseFloat(e.target.value) || 0))}
              style={{ ...c.formInput, width: 70 } as CSSProperties}
            />
            <span style={{ ...c.formLabel, width: 40 }}>Label</span>
            <input
              type="text"
              value={formLabel}
              placeholder="Event name"
              onChange={(e) => setFormLabel(e.target.value)}
              style={{ ...c.formInput, flex: 1, minWidth: 120 } as CSSProperties}
            />
          </div>

          {/* Duration for RAMP */}
          {formKind === EventKind.RAMP && (
            <div style={c.formRow}>
              <span style={c.formLabel}>Duration</span>
              <input
                type="number"
                min={1}
                max={tEnd}
                value={formDuration}
                onChange={(e) => setFormDuration(Math.max(1, parseFloat(e.target.value) || 1))}
                style={{ ...c.formInput, width: 70 } as CSSProperties}
              />
              <span style={{ fontSize: 12, color: '#888' }}>days</span>
            </div>
          )}

          {/* Ejaculation preset for PULSE */}
          {formKind === EventKind.PULSE && (
            <div style={{ marginBottom: 10 }}>
              <button style={c.presetBtn} onClick={applyEjaculationPreset}>
                Ejaculation Preset
              </button>
            </div>
          )}

          {/* Variable selectors */}
          <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>
            {formKind === EventKind.SET
              ? 'Select driving variables to set:'
              : formKind === EventKind.PULSE
                ? 'Select state variables to apply delta:'
                : 'Select driving variables to ramp toward:'}
          </div>
          {varsForKind.map((v) => {
            const range = rangeFor(v);
            const active = selectedVars[v] ?? false;
            return (
              <div key={v} style={c.varRow}>
                <input
                  type="checkbox"
                  checked={active}
                  style={c.checkbox}
                  onChange={(e) => setSelectedVars({ ...selectedVars, [v]: e.target.checked })}
                />
                <span style={c.varLabel}>{v}</span>
                {active && (
                  <>
                    <input
                      type="range"
                      min={range.min}
                      max={range.max}
                      step={range.step}
                      value={varValues[v] ?? 0}
                      onChange={(e) => setVarValues({ ...varValues, [v]: parseFloat(e.target.value) })}
                      style={c.slider}
                    />
                    <input
                      type="number"
                      min={range.min}
                      max={range.max}
                      step={range.step}
                      value={varValues[v] ?? 0}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) setVarValues({ ...varValues, [v]: val });
                      }}
                      style={c.numInput as CSSProperties}
                    />
                  </>
                )}
                {!active && DRIVING_LABELS[v] && (
                  <span style={{ fontSize: 11, color: '#555' }}>{DRIVING_LABELS[v]}</span>
                )}
              </div>
            );
          })}

          {/* Confirm / cancel */}
          <div style={{ display: 'flex', gap: 8 }}>
            <button style={c.confirmBtn} onClick={addEvent}>Add</button>
            <button style={c.cancelBtn} onClick={() => setAdding(false)}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}
