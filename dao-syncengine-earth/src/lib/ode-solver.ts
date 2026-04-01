/**
 * Dormand-Prince RK45 Adaptive Step ODE Integrator
 * =================================================
 *
 * A browser-compatible implementation of the RK45 (Dormand-Prince) method
 * with adaptive step-size control via embedded error estimation, matching
 * scipy.integrate.solve_ivp defaults (rtol=1e-8, atol=1e-10).
 */

export interface ODEResult {
  t: number[];
  y: number[][]; // y[varIndex][timeIndex]
}

// Dormand-Prince coefficients (standard RK45)
const A2 = 1 / 5;
const A3 = 3 / 10;
const A4 = 4 / 5;
const A5 = 8 / 9;
// a6 = 1, a7 = 1

const B21 = 1 / 5;

const B31 = 3 / 40;
const B32 = 9 / 40;

const B41 = 44 / 45;
const B42 = -56 / 15;
const B43 = 32 / 9;

const B51 = 19372 / 6561;
const B52 = -25360 / 2187;
const B53 = 64448 / 6561;
const B54 = -212 / 729;

const B61 = 9017 / 3168;
const B62 = -355 / 33;
const B63 = 46732 / 5247;
const B64 = 49 / 176;
const B65 = -5103 / 18656;

// 5th-order weights (solution)
const C1 = 35 / 384;
// C2 = 0
const C3 = 500 / 1113;
const C4 = 125 / 192;
const C5 = -2187 / 6784;
const C6 = 11 / 84;

// 4th-order weights (error estimator)
const D1 = 5179 / 57600;
// D2 = 0
const D3 = 7571 / 16695;
const D4 = 393 / 640;
const D5 = -92097 / 339200;
const D6 = 187 / 2100;
const D7 = 1 / 40;

// Error coefficients: E_i = C_i - D_i
const E1 = C1 - D1;
const E3 = C3 - D3;
const E4 = C4 - D4;
const E5 = C5 - D5;
const E6 = C6 - D6;
const E7 = -D7;

// Dense output coefficients for Dormand-Prince (4th-order interpolant)
const R1 = 1;  // will be multiplied by h * k1
const R3 = -100 / 13;
const R4 = 125 / 16;
const R5 = -2187 / 2816;
const R6 = 11 / 84;

/**
 * Solve an ODE system using Dormand-Prince RK45 with adaptive stepping.
 *
 * @param rhs - Right-hand side function: (t, y) => dy/dt
 * @param tSpan - Integration interval [t0, tEnd]
 * @param y0 - Initial state vector
 * @param options - Solver options
 * @returns ODEResult with evenly-spaced time points
 */
export function solveODE(
  rhs: (t: number, y: number[]) => number[],
  tSpan: [number, number],
  y0: number[],
  options?: {
    rtol?: number;
    atol?: number;
    maxStep?: number;
    pointsPerDay?: number;
  }
): ODEResult {
  const rtol = options?.rtol ?? 1e-8;
  const atol = options?.atol ?? 1e-10;
  const maxStep = options?.maxStep ?? 0.5;
  const pointsPerDay = options?.pointsPerDay ?? 20;

  const t0 = tSpan[0];
  const tEnd = tSpan[1];
  const n = y0.length;
  const direction = tEnd >= t0 ? 1 : -1;
  const span = Math.abs(tEnd - t0);

  // Generate evenly-spaced output times
  const nPoints = Math.max(Math.round(span * pointsPerDay), 10);
  const tOut: number[] = new Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    tOut[i] = t0 + (i / (nPoints - 1)) * (tEnd - t0);
  }

  // Result arrays
  const yOut: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    yOut[i] = new Array(nPoints);
  }

  // Store initial values at t0
  let outputIdx = 0;
  for (let i = 0; i < n; i++) {
    yOut[i][0] = y0[i];
  }
  outputIdx = 1;

  // Working arrays
  let y = y0.slice();
  let t = t0;

  // Initial step size estimate
  const f0 = rhs(t, y);
  let h = estimateInitialStep(t, y, f0, rhs, rtol, atol, direction, maxStep);
  h = Math.min(h, maxStep);

  // Temporary arrays (pre-allocate to avoid GC)
  const k1 = new Array(n);
  const k2 = new Array(n);
  const k3 = new Array(n);
  const k4 = new Array(n);
  const k5 = new Array(n);
  const k6 = new Array(n);
  const k7 = new Array(n);
  const yTmp = new Array(n);
  const yNew = new Array(n);
  const errArr = new Array(n);

  // Copy f0 into k1 for first step (FSAL property)
  for (let i = 0; i < n; i++) k1[i] = f0[i];

  const MAX_ITER = 1_000_000;
  let iter = 0;
  let fsal = true; // k1 is already computed for first step

  while (direction * (t - tEnd) < 0 && iter < MAX_ITER) {
    iter++;

    // Clamp step to not overshoot tEnd
    if (direction * (t + h - tEnd) > 0) {
      h = tEnd - t;
    }

    // Also clamp to maxStep
    if (Math.abs(h) > maxStep) {
      h = direction * maxStep;
    }

    // Compute k1 if not available via FSAL
    if (!fsal) {
      const f = rhs(t, y);
      for (let i = 0; i < n; i++) k1[i] = f[i];
    }

    // k2
    for (let i = 0; i < n; i++) yTmp[i] = y[i] + h * B21 * k1[i];
    const fk2 = rhs(t + A2 * h, yTmp);
    for (let i = 0; i < n; i++) k2[i] = fk2[i];

    // k3
    for (let i = 0; i < n; i++) yTmp[i] = y[i] + h * (B31 * k1[i] + B32 * k2[i]);
    const fk3 = rhs(t + A3 * h, yTmp);
    for (let i = 0; i < n; i++) k3[i] = fk3[i];

    // k4
    for (let i = 0; i < n; i++) yTmp[i] = y[i] + h * (B41 * k1[i] + B42 * k2[i] + B43 * k3[i]);
    const fk4 = rhs(t + A4 * h, yTmp);
    for (let i = 0; i < n; i++) k4[i] = fk4[i];

    // k5
    for (let i = 0; i < n; i++)
      yTmp[i] = y[i] + h * (B51 * k1[i] + B52 * k2[i] + B53 * k3[i] + B54 * k4[i]);
    const fk5 = rhs(t + A5 * h, yTmp);
    for (let i = 0; i < n; i++) k5[i] = fk5[i];

    // k6
    for (let i = 0; i < n; i++)
      yTmp[i] =
        y[i] + h * (B61 * k1[i] + B62 * k2[i] + B63 * k3[i] + B64 * k4[i] + B65 * k5[i]);
    const fk6 = rhs(t + h, yTmp);
    for (let i = 0; i < n; i++) k6[i] = fk6[i];

    // 5th-order solution
    for (let i = 0; i < n; i++) {
      yNew[i] = y[i] + h * (C1 * k1[i] + C3 * k3[i] + C4 * k4[i] + C5 * k5[i] + C6 * k6[i]);
    }

    // k7 = f(t + h, yNew)  (needed for error estimate and FSAL)
    const fk7 = rhs(t + h, yNew);
    for (let i = 0; i < n; i++) k7[i] = fk7[i];

    // Error estimate
    let errNorm = 0;
    for (let i = 0; i < n; i++) {
      errArr[i] =
        h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
      const sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(yNew[i]));
      errNorm += (errArr[i] / sc) ** 2;
    }
    errNorm = Math.sqrt(errNorm / n);

    if (errNorm <= 1.0) {
      // Step accepted
      const tNew = t + h;

      // Dense output: interpolate to fill any output points in [t, tNew]
      while (outputIdx < nPoints && direction * (tOut[outputIdx] - tNew) <= 0) {
        const theta = (tOut[outputIdx] - t) / h;
        hermiteInterp(theta, h, y, k1, k3, k4, k5, k6, k7, yOut, outputIdx, n);
        outputIdx++;
      }

      // Advance state (FSAL: k7 becomes k1 for next step)
      t = tNew;
      for (let i = 0; i < n; i++) {
        y[i] = yNew[i];
        k1[i] = k7[i];
      }
      fsal = true;

      // Step size adjustment (PI controller, order 5)
      const factor = Math.min(5.0, Math.max(0.2, 0.9 * errNorm ** (-1 / 5)));
      h = h * factor;
    } else {
      // Step rejected
      const factor = Math.max(0.2, 0.9 * errNorm ** (-1 / 5));
      h = h * factor;
      fsal = false;
    }
  }

  // Fill any remaining output points with the final state
  while (outputIdx < nPoints) {
    for (let i = 0; i < n; i++) {
      yOut[i][outputIdx] = y[i];
    }
    outputIdx++;
  }

  return { t: tOut, y: yOut };
}

/**
 * 4th-order Hermite interpolation using the Dormand-Prince stages.
 * Uses the standard "free" dense output for DOPRI5.
 */
function hermiteInterp(
  theta: number,
  h: number,
  y: number[],
  k1: number[],
  k3: number[],
  k4: number[],
  k5: number[],
  k6: number[],
  k7: number[],
  yOut: number[][],
  outIdx: number,
  n: number
): void {
  // Dense output formula for Dormand-Prince:
  // y(t + theta*h) = y + h * theta * (b1 + theta*(b3 + ...))
  // Using the standard 4th-order dense output coefficients
  const th2 = theta * theta;
  const th3 = th2 * theta;
  const th4 = th3 * theta;

  // Coefficients from Hairer-Norsett-Wanner (dense output for DOPRI5)
  // bi(theta) polynomials
  const b1 = theta - 1.0 * (1337 / 480) * th2 + 1.0 * (1039 / 360) * th3 - 1.0 * (1163 / 1152) * th4;
  const b3 = (100 / 1113) * (theta * 0 + th2 * (1113 / 50) - th3 * (2574 / 75) + th4 * (261 / 20));
  const b4 = -(125 / 192) * (-th2 * (192 / 125) + th3 * (288 / 125) - th4 * (128 / 125));
  const b5 = (2187 / 6784) * (-th2 * (6784 / 2187) + th3 * (13552 / 2187) - th4 * (8224 / 2187));
  const b6 = -(11 / 84) * (-th2 * (84 / 11) + th3 * (210 / 11) - th4 * (160 / 11));
  const b7 = th2 * (theta - 1) * (theta - 1);

  // Simplified: use the standard approach
  // Actually, use the well-known Dormand-Prince free interpolant:
  for (let i = 0; i < n; i++) {
    yOut[i][outIdx] =
      y[i] +
      h *
        (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i] + b7 * k7[i]);
  }
}

/**
 * Estimate initial step size using the approach from Hairer-Norsett-Wanner.
 */
function estimateInitialStep(
  t0: number,
  y0: number[],
  f0: number[],
  rhs: (t: number, y: number[]) => number[],
  rtol: number,
  atol: number,
  direction: number,
  maxStep: number
): number {
  const n = y0.length;

  // Compute norms
  let d0 = 0;
  let d1 = 0;
  for (let i = 0; i < n; i++) {
    const sc = atol + Math.abs(y0[i]) * rtol;
    d0 += (y0[i] / sc) ** 2;
    d1 += (f0[i] / sc) ** 2;
  }
  d0 = Math.sqrt(d0 / n);
  d1 = Math.sqrt(d1 / n);

  let h0: number;
  if (d0 < 1e-5 || d1 < 1e-5) {
    h0 = 1e-6;
  } else {
    h0 = 0.01 * (d0 / d1);
  }
  h0 = Math.min(h0, maxStep);

  // Explicit Euler step
  const y1 = new Array(n);
  for (let i = 0; i < n; i++) {
    y1[i] = y0[i] + direction * h0 * f0[i];
  }
  const f1 = rhs(t0 + direction * h0, y1);

  // Estimate second derivative norm
  let d2 = 0;
  for (let i = 0; i < n; i++) {
    const sc = atol + Math.abs(y0[i]) * rtol;
    d2 += ((f1[i] - f0[i]) / sc) ** 2;
  }
  d2 = Math.sqrt(d2 / n) / h0;

  // Step size from second derivative
  let h1: number;
  if (Math.max(d1, d2) <= 1e-15) {
    h1 = Math.max(1e-6, h0 * 1e-3);
  } else {
    h1 = (0.01 / Math.max(d1, d2)) ** (1 / 5);
  }

  return direction * Math.min(100 * h0, h1, maxStep);
}
