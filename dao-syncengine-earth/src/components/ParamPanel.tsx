import { useState, type CSSProperties } from 'react';
import { type Params, defaultParams } from '../lib/holon-model';

interface ParamPanelProps {
  params: Params;
  onChange: (params: Params) => void;
}

interface ParamDef {
  key: keyof Params;
  label: string;
  min: number;
  max: number;
  step: number;
  tip: string;
}

const PARAM_GROUPS: { title: string; params: ParamDef[] }[] = [
  {
    title: 'Generation',
    params: [
      { key: 'Gamma0', label: 'Γ₀', min: 0, max: 5, step: 0.1, tip: 'Baseline jing generation rate' },
      { key: 'p_min', label: 'p_min', min: 0, max: 1, step: 0.01, tip: 'Floor on generation modulation' },
      { key: 'Kp', label: 'Kp', min: 0.01, max: 5, step: 0.1, tip: 'Half-saturation for generation modulation' },
    ],
  },
  {
    title: 'Dissipation',
    params: [
      { key: 'lam_jing', label: 'λ_jing', min: 0, max: 1, step: 0.01, tip: 'Jing dissipation (slow, durable)' },
      { key: 'lam_qi', label: 'λ_qi', min: 0, max: 1, step: 0.01, tip: 'Qi dissipation (medium)' },
      { key: 'lam_shen', label: 'λ_shen', min: 0, max: 2, step: 0.01, tip: 'Shen dissipation (fast, fragile)' },
    ],
  },
  {
    title: 'Transmutation',
    params: [
      { key: 'kappa10', label: 'κ₁₀', min: 0, max: 2, step: 0.01, tip: 'Baseline jing→qi rate' },
      { key: 'kappa20', label: 'κ₂₀', min: 0, max: 1, step: 0.01, tip: 'Baseline qi→shen rate' },
      { key: 'a1', label: 'a₁', min: 0.1, max: 5, step: 0.1, tip: 'κ₁ coherence exponent' },
      { key: 'a2', label: 'a₂', min: 0.1, max: 5, step: 0.1, tip: 'κ₂ coherence exponent' },
      { key: 'b1', label: 'b₁', min: 0.1, max: 5, step: 0.1, tip: 'κ₁ imagination exponent' },
      { key: 'b2', label: 'b₂', min: 0.1, max: 5, step: 0.1, tip: 'κ₂ imagination exponent' },
    ],
  },
  {
    title: 'Re-infusion',
    params: [
      { key: 'mu1', label: 'μ₁', min: 0, max: 1, step: 0.01, tip: 'Qi→jing passive re-infusion' },
      { key: 'mu2', label: 'μ₂', min: 0, max: 1, step: 0.01, tip: 'Shen→qi passive re-infusion' },
    ],
  },
  {
    title: 'Catalytic',
    params: [
      { key: 'f_min', label: 'f_min', min: 0, max: 0.5, step: 0.01, tip: 'κ₁ catalytic floor' },
      { key: 'g_min', label: 'g_min', min: 0, max: 0.5, step: 0.01, tip: 'κ₂ catalytic floor' },
      { key: 'Kf', label: 'Kf', min: 0.01, max: 5, step: 0.1, tip: 'κ₁ catalytic half-saturation' },
      { key: 'Kg', label: 'Kg', min: 0.01, max: 5, step: 0.1, tip: 'κ₂ catalytic half-saturation' },
    ],
  },
  {
    title: 'Congestion',
    params: [
      { key: 'sigma', label: 'σ', min: 0, max: 1, step: 0.01, tip: 'Congestion-stress coupling coefficient' },
    ],
  },
  {
    title: 'Infrastructure',
    params: [
      { key: 'h', label: 'h', min: 0, max: 1, step: 0.01, tip: 'Coherence building rate' },
      { key: 'd_ext', label: 'd_ext', min: 0, max: 1, step: 0.01, tip: 'External stimulation damage rate' },
      { key: 'd_nov', label: 'd_nov', min: 0, max: 1, step: 0.01, tip: 'Novelty escalation damage rate' },
      { key: 'rho', label: 'ρ', min: 0, max: 1, step: 0.01, tip: 'Reciprocity coherence building' },
      { key: 'eps', label: 'ε', min: 0, max: 1, step: 0.01, tip: 'Imagination building rate' },
      { key: 'omega', label: 'ω', min: 0, max: 1, step: 0.01, tip: 'Imagination atrophy from stimulation' },
      { key: 'delta_I', label: 'δ_I', min: 0, max: 0.5, step: 0.01, tip: 'Imagination natural decay from disuse' },
    ],
  },
];

const s: Record<string, CSSProperties> = {
  wrapper: {
    borderTop: '1px solid #2a2a35',
    marginTop: 24,
    paddingTop: 16,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    cursor: 'pointer',
    padding: '8px 0',
    userSelect: 'none',
  },
  title: {
    fontSize: 16,
    fontWeight: 600,
    color: '#e8ddd0',
    letterSpacing: '0.02em',
  },
  arrow: {
    color: '#777',
    fontSize: 14,
    transition: 'transform 0.2s',
  },
  body: {
    paddingTop: 12,
  },
  group: {
    marginBottom: 20,
  },
  groupTitle: {
    fontSize: 13,
    fontWeight: 600,
    color: '#9370DB',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    marginBottom: 10,
    borderBottom: '1px solid #1e1e28',
    paddingBottom: 4,
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    marginBottom: 6,
    flexWrap: 'wrap' as const,
  },
  label: {
    width: 60,
    fontSize: 13,
    color: '#c8c0b8',
    fontFamily: 'monospace',
    flexShrink: 0,
  },
  slider: {
    flex: 1,
    minWidth: 100,
    accentColor: '#4169E1',
    height: 4,
  },
  numInput: {
    width: 70,
    background: '#14141e',
    border: '1px solid #2a2a35',
    borderRadius: 4,
    color: '#e8ddd0',
    padding: '3px 6px',
    fontSize: 12,
    fontFamily: 'monospace',
    textAlign: 'right' as const,
  },
  tip: {
    fontSize: 11,
    color: '#666',
    width: '100%',
    paddingLeft: 70,
    marginTop: -2,
    marginBottom: 4,
  },
  resetBtn: {
    background: 'transparent',
    border: '1px solid #2a2a35',
    color: '#999',
    padding: '6px 14px',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 12,
    marginTop: 8,
  },
};

export default function ParamPanel({ params, onChange }: ParamPanelProps) {
  const [open, setOpen] = useState(false);

  const update = (key: keyof Params, value: number) => {
    onChange({ ...params, [key]: value });
  };

  return (
    <div style={s.wrapper}>
      <div style={s.header} onClick={() => setOpen(!open)}>
        <span style={s.title}>Advanced Parameters</span>
        <span style={s.arrow}>{open ? '\u25B2' : '\u25BC'}</span>
      </div>
      {open && (
        <div style={s.body}>
          {PARAM_GROUPS.map((group) => (
            <div key={group.title} style={s.group}>
              <div style={s.groupTitle}>{group.title}</div>
              {group.params.map((pd) => (
                <div key={pd.key}>
                  <div style={s.row}>
                    <span style={s.label} title={pd.tip}>{pd.label}</span>
                    <input
                      type="range"
                      min={pd.min}
                      max={pd.max}
                      step={pd.step}
                      value={params[pd.key]}
                      onChange={(e) => update(pd.key, parseFloat(e.target.value))}
                      style={s.slider}
                    />
                    <input
                      type="number"
                      min={pd.min}
                      max={pd.max}
                      step={pd.step}
                      value={params[pd.key]}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value);
                        if (!isNaN(v)) update(pd.key, v);
                      }}
                      style={s.numInput}
                    />
                  </div>
                  <div style={s.tip}>{pd.tip}</div>
                </div>
              ))}
            </div>
          ))}
          <button
            style={s.resetBtn}
            onClick={() => onChange(defaultParams())}
            onMouseEnter={(e) => { (e.target as HTMLElement).style.borderColor = '#4169E1'; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.borderColor = '#2a2a35'; }}
          >
            Reset to Defaults
          </button>
        </div>
      )}
    </div>
  );
}
