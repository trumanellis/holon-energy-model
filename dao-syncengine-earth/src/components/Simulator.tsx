import { useState, useCallback, type CSSProperties } from 'react';
import { defaultParams, type Params } from '../lib/holon-model';
import {
  type SimEvent,
  type StitchedResult,
  type EventScenario,
  PRESET_SCENARIOS,
  runEventScenario,
} from '../lib/event-engine';
import EventTimeline from './EventTimeline';
import TrajectoryChart from './TrajectoryChart';
import PhasePortrait from './PhasePortrait';
import ParamPanel from './ParamPanel';

// --- Styles ---

const s: Record<string, CSSProperties> = {
  container: {
    maxWidth: 960,
    margin: '0 auto',
    padding: '0 16px',
    fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
    color: '#e8ddd0',
  },
  header: {
    marginBottom: 28,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    letterSpacing: '-0.02em',
    color: '#e8ddd0',
    margin: 0,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#999',
    lineHeight: 1.6,
    margin: 0,
    maxWidth: 640,
  },
  presetRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 20,
    flexWrap: 'wrap',
  },
  presetLabel: {
    fontSize: 13,
    fontWeight: 600,
    color: '#999',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
  },
  select: {
    background: '#14141e',
    border: '1px solid #2a2a35',
    borderRadius: 6,
    color: '#e8ddd0',
    padding: '8px 12px',
    fontSize: 14,
    cursor: 'pointer',
    minWidth: 200,
  },
  controlsBar: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
    marginBottom: 24,
    flexWrap: 'wrap',
  },
  runBtn: {
    background: '#4169E1',
    border: 'none',
    color: '#fff',
    padding: '10px 28px',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: 15,
    fontWeight: 600,
    letterSpacing: '0.01em',
    transition: 'background 0.15s',
  },
  durationGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  durationLabel: {
    fontSize: 13,
    color: '#888',
  },
  durationInput: {
    width: 70,
    background: '#14141e',
    border: '1px solid #2a2a35',
    borderRadius: 4,
    color: '#e8ddd0',
    padding: '6px 10px',
    fontSize: 14,
    fontFamily: 'monospace',
    textAlign: 'center',
  },
  resultsSection: {
    marginTop: 32,
    borderTop: '1px solid #2a2a35',
    paddingTop: 24,
  },
  divider: {
    border: 'none',
    borderTop: '1px solid #1e1e28',
    margin: '24px 0',
  },
};

export default function Simulator() {
  const [params, setParams] = useState<Params>(defaultParams);
  const [events, setEvents] = useState<SimEvent[]>(PRESET_SCENARIOS[0].events);
  const [baseline, setBaseline] = useState(PRESET_SCENARIOS[0].baseline);
  const [selectedPreset, setSelectedPreset] = useState<string>(PRESET_SCENARIOS[0].name);
  const [duration, setDuration] = useState(108);
  const [result, setResult] = useState<StitchedResult | null>(null);

  const loadPreset = useCallback((name: string) => {
    setSelectedPreset(name);
    if (name === 'Custom') {
      // Keep current events and baseline
      return;
    }
    const preset = PRESET_SCENARIOS.find((p) => p.name === name);
    if (preset) {
      setEvents([...preset.events]);
      setBaseline({ ...preset.baseline });
      setResult(null);
    }
  }, []);

  const runSim = useCallback(() => {
    const es: EventScenario = { name: selectedPreset, baseline, events };
    const res = runEventScenario(params, es, duration);
    setResult(res);
  }, [params, events, baseline, selectedPreset, duration]);

  const handleEventsChange = useCallback((newEvents: SimEvent[]) => {
    setEvents(newEvents);
    if (selectedPreset !== 'Custom') setSelectedPreset('Custom');
  }, [selectedPreset]);

  return (
    <div style={s.container}>
      {/* Header */}
      <div style={s.header}>
        <h2 style={s.title}>Interactive Simulator</h2>
        <p style={s.subtitle}>
          Define life events and watch the model predict how your energy system evolves over 108 days.
          Everything runs locally in your browser.
        </p>
      </div>

      {/* Preset selector */}
      <div style={s.presetRow}>
        <span style={s.presetLabel}>Scenario</span>
        <select
          style={s.select}
          value={selectedPreset}
          onChange={(e) => loadPreset(e.target.value)}
        >
          {PRESET_SCENARIOS.map((p) => (
            <option key={p.name} value={p.name}>{p.name}</option>
          ))}
          <option value="Custom">Custom</option>
        </select>
      </div>

      {/* Event timeline */}
      <EventTimeline events={events} onChange={handleEventsChange} tEnd={duration} />

      {/* Controls bar */}
      <div style={s.controlsBar}>
        <button
          style={s.runBtn}
          onClick={runSim}
          onMouseEnter={(e) => { (e.target as HTMLElement).style.background = '#5279f0'; }}
          onMouseLeave={(e) => { (e.target as HTMLElement).style.background = '#4169E1'; }}
        >
          Run Simulation
        </button>
        <div style={s.durationGroup}>
          <span style={s.durationLabel}>Duration</span>
          <input
            type="number"
            min={1}
            max={1000}
            value={duration}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              if (!isNaN(v) && v > 0) setDuration(v);
            }}
            style={s.durationInput as CSSProperties}
          />
          <span style={s.durationLabel}>days</span>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div style={s.resultsSection}>
          <TrajectoryChart result={result} />
          <hr style={s.divider} />
          <PhasePortrait result={result} params={params} />
        </div>
      )}

      {/* Advanced parameters */}
      <ParamPanel params={params} onChange={setParams} />
    </div>
  );
}
