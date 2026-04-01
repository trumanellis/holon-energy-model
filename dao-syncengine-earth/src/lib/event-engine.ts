/**
 * Event-Driven Holon Simulation Engine (TypeScript)
 * ==================================================
 *
 * Port of holon_event_sim.py. Extends the base ODE model with an event-driven
 * architecture where scenarios are defined as timelines of discrete events
 * (practice changes, ejaculations, gradual ramp-ups) rather than fixed driving
 * variables.
 *
 * Three event types:
 *   SET   — change driving variables to new values
 *   PULSE — instantaneous state vector deltas (e.g. ejaculation)
 *   RAMP  — gradually transition variables over N days
 */

import { solveODE } from './ode-solver';
import {
  type Params,
  type Scenario,
  type ScenarioState,
  defaultParams,
  holonODE,
  DRIVING_VARS,
  STATE_VARS,
  type DrivingVar,
  type StateVar,
} from './holon-model';

// Re-export for convenience
export { DRIVING_VARS, STATE_VARS };

// ── Event Types ──────────────────────────────────────────────────────────

export enum EventKind {
  SET = 'set',
  PULSE = 'pulse',
  RAMP = 'ramp',
}

export interface SimEvent {
  time: number;
  kind: EventKind;
  label: string;
  /** SET: driving variable overrides */
  changes?: Record<string, number>;
  /** PULSE: state vector deltas (e.g. { phi_jing: -1.5 }) */
  deltas?: Record<string, number>;
  /** RAMP: target values for driving variables */
  targets?: Record<string, number>;
  /** RAMP: duration in days */
  duration?: number;
}

// ── Mutable Scenario Proxy ───────────────────────────────────────────────

interface RampEntry {
  startVal: number;
  targetVal: number;
  tStart: number;
  tEnd: number;
}

/**
 * Mutable proxy for Scenario that the ODE reads via attribute access.
 * Supports instant mutation (SET events) and time-interpolated ramps.
 */
export class MutableScenario implements ScenarioState {
  H: number;
  E: number;
  S: number;
  N: number;
  G: number;
  Rc: number;
  D_jing: number;
  R_qi: number;
  D_qi: number;

  private _ramps: Map<string, RampEntry> = new Map();

  constructor(base: ScenarioState) {
    this.H = base.H;
    this.E = base.E;
    this.S = base.S;
    this.N = base.N;
    this.G = base.G;
    this.Rc = base.Rc;
    this.D_jing = base.D_jing;
    this.R_qi = base.R_qi;
    this.D_qi = base.D_qi;
  }

  setVar(name: string, value: number): void {
    (this as any)[name] = value;
    this._ramps.delete(name);
  }

  addRamp(name: string, target: number, tStart: number, tEnd: number): void {
    const startVal = (this as any)[name] as number;
    this._ramps.set(name, { startVal, targetVal: target, tStart, tEnd });
  }

  updateForTime(t: number): void {
    const toDelete: string[] = [];
    for (const [name, ramp] of this._ramps) {
      if (t >= ramp.tEnd) {
        (this as any)[name] = ramp.targetVal;
        toDelete.push(name);
      } else if (t >= ramp.tStart) {
        const frac = (t - ramp.tStart) / (ramp.tEnd - ramp.tStart);
        (this as any)[name] = ramp.startVal + frac * (ramp.targetVal - ramp.startVal);
      }
    }
    for (const name of toDelete) {
      this._ramps.delete(name);
    }
  }

  get hasActiveRamps(): boolean {
    return this._ramps.size > 0;
  }
}

// ── Result types ─────────────────────────────────────────────────────────

export interface EventLogEntry {
  time: number;
  label: string;
  kind: string;
}

export interface StitchedResult {
  t: number[];
  y: number[][]; // y[varIndex][timeIndex]
  eventsLog: EventLogEntry[];
}

// ── Event Scenario ───────────────────────────────────────────────────────

export interface EventScenario {
  name: string;
  baseline: Scenario;
  events: SimEvent[];
}

// ── Core Integration ─────────────────────────────────────────────────────

/**
 * Integrate the ODE system with events modifying driving variables.
 *
 * Segments the integration at event boundaries, applies SET/PULSE/RAMP
 * at each boundary, and stitches the results together.
 */
export function runEventScenario(
  params: Params,
  es: EventScenario,
  tEnd: number = 108.0,
  pointsPerDay: number = 20
): StitchedResult {
  const events = [...es.events].sort((a, b) => a.time - b.time);
  const baseline = es.baseline;

  // Collect all boundary times (event times + ramp end times)
  const boundarySet = new Set<number>([0.0, tEnd]);
  for (const e of events) {
    if (e.time >= 0.0 && e.time <= tEnd) {
      boundarySet.add(e.time);
    }
    if (e.kind === EventKind.RAMP && e.duration) {
      const rampEnd = e.time + e.duration;
      if (rampEnd > 0.0 && rampEnd < tEnd) {
        boundarySet.add(rampEnd);
      }
    }
  }
  const boundaries = Array.from(boundarySet).sort((a, b) => a - b);

  // Initialize
  const ms = new MutableScenario(baseline);
  const yCurrent: number[] = [
    baseline.phi_jing_0,
    baseline.phi_qi_0,
    baseline.phi_shen_0,
    baseline.C_0,
    baseline.I_0,
  ];

  const tSegments: number[][] = [];
  const ySegments: number[][][] = []; // each: [varIdx][timeIdx]
  const eventsLog: EventLogEntry[] = [];

  const stateVarsList = ['phi_jing', 'phi_qi', 'phi_shen', 'C', 'I'];

  for (let i = 0; i < boundaries.length - 1; i++) {
    const tStart = boundaries[i];
    const tStop = boundaries[i + 1];

    // Apply all events at tStart
    for (const ev of events) {
      if (ev.time !== tStart) continue;

      if (ev.kind === EventKind.SET && ev.changes) {
        for (const [varName, val] of Object.entries(ev.changes)) {
          ms.setVar(varName, val);
        }
        eventsLog.push({ time: tStart, label: ev.label, kind: 'set' });
      } else if (ev.kind === EventKind.PULSE && ev.deltas) {
        for (const [stateName, delta] of Object.entries(ev.deltas)) {
          const idx = stateVarsList.indexOf(stateName);
          if (idx >= 0) {
            yCurrent[idx] = Math.max(0.0, yCurrent[idx] + delta);
            if (idx >= 3) {
              // C, I bounded [0, 1]
              yCurrent[idx] = Math.min(1.0, yCurrent[idx]);
            }
          }
        }
        eventsLog.push({ time: tStart, label: ev.label, kind: 'pulse' });
      } else if (ev.kind === EventKind.RAMP && ev.targets && ev.duration) {
        for (const [varName, target] of Object.entries(ev.targets)) {
          ms.addRamp(varName, target, tStart, tStart + ev.duration);
        }
        eventsLog.push({ time: tStart, label: ev.label, kind: 'ramp' });
      }
    }

    // Skip zero-length segments
    if (tStop <= tStart) continue;

    const segDuration = tStop - tStart;
    const nPts = Math.max(Math.round(segDuration * pointsPerDay), 10);

    // Build the RHS function for this segment
    let rhsFn: (t: number, y: number[]) => number[];
    if (ms.hasActiveRamps) {
      // Wrap so ramps interpolate at each evaluation
      rhsFn = (t: number, y: number[]): number[] => {
        ms.updateForTime(t);
        return holonODE(t, y, params, ms);
      };
    } else {
      rhsFn = (t: number, y: number[]): number[] => {
        return holonODE(t, y, params, ms);
      };
    }

    const sol = solveODE(rhsFn, [tStart, tStop], yCurrent.slice(), {
      rtol: 1e-8,
      atol: 1e-10,
      maxStep: 0.5,
      pointsPerDay,
    });

    // Store segment (drop last point to avoid duplication, except final segment)
    const lastIdx = sol.t.length;
    const trimEnd = i < boundaries.length - 2 ? lastIdx - 1 : lastIdx;

    tSegments.push(sol.t.slice(0, trimEnd));

    const segY: number[][] = [];
    for (let v = 0; v < 5; v++) {
      segY.push(sol.y[v].slice(0, trimEnd));
    }
    ySegments.push(segY);

    // Update current state from end of segment
    for (let v = 0; v < 5; v++) {
      yCurrent[v] = sol.y[v][lastIdx - 1];
    }
  }

  // Stitch segments
  const tAll = flatConcat(tSegments);
  const yAll: number[][] = [];
  for (let v = 0; v < 5; v++) {
    const varSegs: number[][] = [];
    for (const seg of ySegments) {
      varSegs.push(seg[v]);
    }
    yAll.push(flatConcat(varSegs));
  }

  return { t: tAll, y: yAll, eventsLog };
}

/** Concatenate an array of number arrays into one. */
function flatConcat(arrs: number[][]): number[] {
  let totalLen = 0;
  for (const a of arrs) totalLen += a.length;
  const result = new Array(totalLen);
  let offset = 0;
  for (const a of arrs) {
    for (let i = 0; i < a.length; i++) {
      result[offset + i] = a[i];
    }
    offset += a.length;
  }
  return result;
}

// ── Preset Scenarios ─────────────────────────────────────────────────────

/**
 * Shared initial conditions for all preset scenarios.
 */
const BASE_IC = {
  phi_jing_0: 2.0,
  phi_qi_0: 2.5,
  phi_shen_0: 0.3,
  C_0: 0.30,
  I_0: 0.45,
};

function makeBaseline(name: string, overrides: Partial<Scenario> = {}): Scenario {
  return {
    name,
    H: 0.0,
    E: 0.5,
    S: 0.0,
    N: 0.0,
    G: 0.5,
    Rc: 0.0,
    D_jing: 0.0,
    R_qi: 0.1,
    D_qi: 0.0,
    ...BASE_IC,
    ...overrides,
  };
}

/**
 * Preset 1: Recovery Journey
 * Sequential SETs at days 0, 14, 30, 50, 70
 */
function makeRecoveryJourney(): EventScenario {
  return {
    name: 'Recovery Journey',
    baseline: makeBaseline('Recovery Journey', {
      H: 0.0, E: 0.2, S: 0.0, N: 0.0, G: 0.1, Rc: 0.0,
      D_jing: 0.0, R_qi: 0.08, D_qi: 0.0,
    }),
    events: [
      {
        time: 0, kind: EventKind.SET, label: 'Begin retention',
        changes: { S: 0.0, N: 0.0, D_jing: 0.0, D_qi: 0.0, E: 0.3 },
      },
      {
        time: 14, kind: EventKind.SET, label: 'Start walking + basic exercise',
        changes: { E: 0.5, R_qi: 0.15 },
      },
      {
        time: 30, kind: EventKind.SET, label: 'Add breathwork',
        changes: { E: 0.6, G: 0.3, R_qi: 0.2 },
      },
      {
        time: 50, kind: EventKind.SET, label: 'Start meditation',
        changes: { H: 0.4, G: 0.5 },
      },
      {
        time: 70, kind: EventKind.SET, label: 'Full cultivation',
        changes: { H: 0.7, E: 0.8, G: 0.75, R_qi: 0.3 },
      },
    ],
  };
}

/**
 * Preset 2: Relapse & Recovery
 * SET at 35 for relapse, 42 recommit, 60 full
 */
function makeRelapseRecovery(): EventScenario {
  return {
    name: 'Relapse & Recovery',
    baseline: makeBaseline('Relapse & Recovery', {
      H: 0.3, E: 0.5, S: 0.0, N: 0.0, G: 0.3, Rc: 0.0,
      D_jing: 0.0, R_qi: 0.15, D_qi: 0.0,
    }),
    events: [
      {
        time: 0, kind: EventKind.SET, label: 'Good start',
        changes: { H: 0.3, E: 0.5, G: 0.3, D_jing: 0.0, S: 0.0 },
      },
      {
        time: 35, kind: EventKind.SET, label: 'Relapse — porn + ejaculation',
        changes: { S: 1.5, N: 0.4, D_jing: 0.6, E: 0.1, H: 0.0, G: 0.0, D_qi: 0.1 },
      },
      {
        time: 42, kind: EventKind.SET, label: 'Recommit — stop porn',
        changes: { S: 0.0, N: 0.0, D_jing: 0.0, D_qi: 0.0, E: 0.4, H: 0.2, G: 0.2 },
      },
      {
        time: 60, kind: EventKind.SET, label: 'Full cultivation resumed',
        changes: { H: 0.7, E: 0.8, G: 0.7, R_qi: 0.3 },
      },
    ],
  };
}

/**
 * Preset 3: Periodic Ejaculation
 * PULSE every 14 days: phi_jing -1.5, phi_qi -0.3
 */
function makePeriodicEjaculation(): EventScenario {
  const events: SimEvent[] = [];
  // Moderate cultivation baseline
  events.push({
    time: 0, kind: EventKind.SET, label: 'Begin cultivation',
    changes: { H: 0.4, E: 0.6, G: 0.4, D_jing: 0.0, S: 0.0, R_qi: 0.2 },
  });
  // Periodic ejaculation every 14 days
  for (let day = 14; day <= 98; day += 14) {
    events.push({
      time: day, kind: EventKind.PULSE, label: `Ejaculation (day ${day})`,
      deltas: { phi_jing: -1.5, phi_qi: -0.3 },
    });
  }
  return {
    name: 'Periodic Ejaculation',
    baseline: makeBaseline('Periodic Ejaculation', {
      H: 0.4, E: 0.6, S: 0.0, N: 0.0, G: 0.4, Rc: 0.0,
      D_jing: 0.0, R_qi: 0.2, D_qi: 0.0,
    }),
    events,
  };
}

/**
 * Preset 4: Partnership
 * SET at 30 and 50
 */
function makePartnership(): EventScenario {
  return {
    name: 'Partnership',
    baseline: makeBaseline('Partnership', {
      H: 0.3, E: 0.5, S: 0.0, N: 0.0, G: 0.3, Rc: 0.0,
      D_jing: 0.0, R_qi: 0.15, D_qi: 0.0,
    }),
    events: [
      {
        time: 0, kind: EventKind.SET, label: 'Solo cultivation begins',
        changes: { H: 0.3, E: 0.6, G: 0.4, D_jing: 0.0, S: 0.0 },
      },
      {
        time: 30, kind: EventKind.SET, label: 'Meet partner — heart opens',
        changes: { H: 0.6, Rc: 0.3, E: 0.7 },
      },
      {
        time: 50, kind: EventKind.SET, label: 'Deepen bond — reciprocal practice',
        changes: { H: 0.8, Rc: 0.6, G: 0.7, E: 0.85 },
      },
    ],
  };
}

/**
 * Preset 5: Gradual Cultivation
 * RAMP from day 10, duration 60
 */
function makeGradualCultivation(): EventScenario {
  return {
    name: 'Gradual Cultivation',
    baseline: makeBaseline('Gradual Cultivation', {
      H: 0.05, E: 0.3, S: 0.0, N: 0.0, G: 0.1, Rc: 0.0,
      D_jing: 0.0, R_qi: 0.1, D_qi: 0.0,
    }),
    events: [
      {
        time: 0, kind: EventKind.SET, label: 'Begin retention (minimal practice)',
        changes: { D_jing: 0.0, S: 0.0, E: 0.3, G: 0.1 },
      },
      {
        time: 10, kind: EventKind.RAMP, label: 'Gradually build all practices',
        targets: { H: 0.75, E: 0.85, G: 0.8, R_qi: 0.3 },
        duration: 60,
      },
    ],
  };
}

/**
 * Preset 6: Mixed Events
 * SET + RAMP + PULSE interleaved
 */
function makeMixedEvents(): EventScenario {
  return {
    name: 'Mixed Events',
    baseline: makeBaseline('Mixed Events', {
      H: 0.1, E: 0.3, S: 0.0, N: 0.0, G: 0.15, Rc: 0.0,
      D_jing: 0.0, R_qi: 0.1, D_qi: 0.0,
    }),
    events: [
      {
        time: 0, kind: EventKind.SET, label: 'Start retention',
        changes: { D_jing: 0.0, S: 0.0, E: 0.35 },
      },
      {
        time: 10, kind: EventKind.RAMP, label: 'Ramp up physical practice',
        targets: { E: 0.7, R_qi: 0.25 },
        duration: 20,
      },
      {
        time: 25, kind: EventKind.PULSE, label: 'Ejaculation slip',
        deltas: { phi_jing: -1.5, phi_qi: -0.3 },
      },
      {
        time: 35, kind: EventKind.SET, label: 'Add meditation',
        changes: { H: 0.4, G: 0.4 },
      },
      {
        time: 45, kind: EventKind.RAMP, label: 'Deepen heart + imagination',
        targets: { H: 0.7, G: 0.7 },
        duration: 25,
      },
      {
        time: 55, kind: EventKind.PULSE, label: 'Stress event — qi drain',
        deltas: { phi_qi: -0.5 },
      },
      {
        time: 70, kind: EventKind.SET, label: 'Full cultivation + partner',
        changes: { H: 0.8, E: 0.85, G: 0.8, Rc: 0.3, R_qi: 0.3 },
      },
      {
        time: 85, kind: EventKind.PULSE, label: 'Minor ejaculation',
        deltas: { phi_jing: -0.8, phi_qi: -0.15 },
      },
    ],
  };
}

/**
 * All 6 preset event-driven scenarios.
 */
export const PRESET_SCENARIOS: EventScenario[] = [
  makeRecoveryJourney(),
  makeRelapseRecovery(),
  makePeriodicEjaculation(),
  makePartnership(),
  makeGradualCultivation(),
  makeMixedEvents(),
];
