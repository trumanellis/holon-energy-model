/**
 * Conservation Laws of a Creative Holon -- Computational Model (TypeScript)
 * =========================================================================
 *
 * Port of holon_sim.py. Five coupled ODEs (phi_jing, phi_qi, phi_shen, C, I)
 * modeling the Daoist Three Treasures as phases of a single conserved field.
 *
 * The transmutation coefficients kappa1_eff and kappa2_eff depend superlinearly
 * on both C (coherence) and I (imagination), creating threshold behavior.
 */

// ── Driving variable and state variable constants ────────────────────────

export const DRIVING_VARS = [
  'H', 'E', 'S', 'N', 'G', 'Rc', 'D_jing', 'R_qi', 'D_qi',
] as const;

export type DrivingVar = (typeof DRIVING_VARS)[number];

export const STATE_VARS = [
  'phi_jing', 'phi_qi', 'phi_shen', 'C', 'I',
] as const;

export type StateVar = (typeof STATE_VARS)[number];

// ── Parameters ───────────────────────────────────────────────────────────

export interface Params {
  // Endogenous generation
  Gamma0: number;   // baseline jing generation rate
  p_min: number;    // floor on generation modulation
  Kp: number;       // half-saturation for generation modulation

  // Dissipation rates (lambda): shen > qi > jing
  lam_jing: number; // jing dissipation (slow — durable)
  lam_qi: number;   // qi dissipation (medium)
  lam_shen: number; // shen dissipation (fast — fragile)

  // Transmutation baselines: kappa2 < kappa1
  kappa10: number;  // baseline jing→qi rate
  kappa20: number;  // baseline qi→shen rate

  // Re-infusion (downward, passive, infrastructure-independent)
  mu1: number;      // qi→jing re-infusion
  mu2: number;      // shen→qi re-infusion

  // Superlinearity exponents: second transition more sensitive
  a1: number;       // kappa1 coherence exponent
  a2: number;       // kappa2 coherence exponent
  b1: number;       // kappa1 imagination exponent
  b2: number;       // kappa2 imagination exponent

  // Catalytic floor and half-saturation
  f_min: number;    // kappa1 catalytic floor
  g_min: number;    // kappa2 catalytic floor
  Kf: number;       // kappa1 catalytic half-saturation
  Kg: number;       // kappa2 catalytic half-saturation

  // Congestion coupling
  sigma: number;    // congestion-stress coefficient

  // Infrastructure dynamics
  h: number;        // coherence building rate
  d_ext: number;    // external stimulation damage rate
  d_nov: number;    // novelty escalation damage rate
  rho: number;      // reciprocity coherence building
  eps: number;      // imagination building rate
  omega: number;    // imagination atrophy rate from stimulation
  delta_I: number;  // imagination natural decay from disuse
}

export function defaultParams(): Params {
  return {
    Gamma0: 1.0,
    p_min: 0.05,
    Kp: 0.5,
    lam_jing: 0.05,
    lam_qi: 0.15,
    lam_shen: 0.40,
    kappa10: 0.30,
    kappa20: 0.10,
    mu1: 0.05,
    mu2: 0.08,
    a1: 1.5,
    a2: 2.0,
    b1: 1.3,
    b2: 2.0,
    f_min: 0.03,
    g_min: 0.02,
    Kf: 0.5,
    Kg: 0.8,
    sigma: 0.10,
    h: 0.15,
    d_ext: 0.20,
    d_nov: 0.25,
    rho: 0.10,
    eps: 0.10,
    omega: 0.30,
    delta_I: 0.04,
  };
}

// ── Scenario types ───────────────────────────────────────────────────────

/** The driving variables that can change over time (subset of Scenario). */
export interface ScenarioState {
  H: number;       // heart engagement
  E: number;       // embodiment
  S: number;       // external stimulation
  N: number;       // novelty escalation
  G: number;       // generative fraction
  Rc: number;      // reciprocity
  D_jing: number;  // ejaculatory discharge rate
  R_qi: number;    // environmental vitality support
  D_qi: number;    // stress/overstimulation qi drain
}

/** Full scenario: driving variables + initial conditions. */
export interface Scenario extends ScenarioState {
  name: string;
  phi_jing_0: number;
  phi_qi_0: number;
  phi_shen_0: number;
  C_0: number;
  I_0: number;
}

export function defaultScenario(): Scenario {
  return {
    name: '',
    H: 0.0,
    E: 0.5,
    S: 0.0,
    N: 0.0,
    G: 0.5,
    Rc: 0.0,
    D_jing: 0.0,
    R_qi: 0.1,
    D_qi: 0.0,
    phi_jing_0: 2.0,
    phi_qi_0: 3.0,
    phi_shen_0: 0.5,
    C_0: 0.3,
    I_0: 0.5,
  };
}

// ── ODE System ───────────────────────────────────────────────────────────

/**
 * Right-hand side of the five coupled ODEs.
 *
 * Direct port of holon_ode() from holon_sim.py.
 * The scenario state `s` can be any object with the driving variable fields
 * (ScenarioState, Scenario, or MutableScenario).
 */
export function holonODE(t: number, y: number[], p: Params, s: ScenarioState): number[] {
  let phi_jing = y[0];
  let phi_qi = y[1];
  let phi_shen = y[2];
  let C = y[3];
  let I = y[4];

  // Enforce non-negativity (the integrator can overshoot)
  phi_jing = Math.max(phi_jing, 0.0);
  phi_qi = Math.max(phi_qi, 0.0);
  phi_shen = Math.max(phi_shen, 0.0);
  C = Math.max(0.0, Math.min(C, 1.0));
  I = Math.max(0.0, Math.min(I, 1.0));

  // Effective generation
  const p_mod = p.p_min + (1.0 - p.p_min) * phi_qi / (phi_qi + p.Kp);
  const Gamma_eff = p.Gamma0 * p_mod;

  // Catalytic functions
  const f_qi = p.f_min + (1.0 - p.f_min) * phi_qi / (phi_qi + p.Kf);
  const g_qi = p.g_min + (1.0 - p.g_min) * phi_qi / (phi_qi + p.Kg);

  // Effective transmutation coefficients
  const C_safe = Math.max(C, 1e-12);
  const I_safe = Math.max(I, 1e-12);

  const kappa1_eff = p.kappa10 * Math.pow(C_safe, p.a1) * Math.pow(I_safe, p.b1) * f_qi;
  const kappa2_eff = p.kappa20 * Math.pow(C_safe, p.a2) * Math.pow(I_safe, p.b2) * g_qi;

  // Natural equilibrium for congestion reference
  const phi_jing_eq = Gamma_eff / p.lam_jing;

  // Congestion stress
  const congestion = p.sigma * Math.max(0.0, phi_jing - phi_jing_eq) * phi_qi;

  // Discharge can't exceed available reserves (soft cap)
  const eff_D_jing = Math.min(s.D_jing, phi_jing * 10.0);
  const eff_D_qi = Math.min(s.D_qi, phi_qi * 10.0);

  // Phase equations
  let d_jing =
    Gamma_eff - (kappa1_eff + p.lam_jing) * phi_jing + p.mu1 * phi_qi - eff_D_jing;

  let d_qi =
    kappa1_eff * phi_jing -
    (p.mu1 + kappa2_eff + p.lam_qi) * phi_qi +
    p.mu2 * phi_shen +
    s.R_qi -
    eff_D_qi -
    congestion;

  let d_shen = kappa2_eff * phi_qi - (p.mu2 + p.lam_shen) * phi_shen;

  // Infrastructure equations
  const dC =
    p.h * s.H * s.E * (1.0 - C) -
    p.d_ext * s.S * (1.0 - s.E) * C -
    p.d_nov * s.N * C +
    p.rho * s.Rc * (1.0 - C);

  const dI =
    p.eps * s.G * (1.0 - I) -
    p.omega * s.S * I -
    p.delta_I * (1.0 - s.G) * I;

  // Non-negativity enforcement: if at zero, derivative can't be negative
  if (y[0] <= 0 && d_jing < 0) d_jing = 0.0;
  if (y[1] <= 0 && d_qi < 0) d_qi = 0.0;
  if (y[2] <= 0 && d_shen < 0) d_shen = 0.0;

  return [d_jing, d_qi, d_shen, dC, dI];
}
