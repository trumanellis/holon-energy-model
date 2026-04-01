# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A computational model formalizing the Daoist Three Treasures (jing, qi, shen) as a system of five coupled ordinary differential equations. The model simulates how sexual energy, vitality, and awareness interact under different practitioner scenarios (retention, cultivation, porn consumption, etc.) over a 108-day period.

## Running the Simulation

```bash
python holon_sim.py
```

Dependencies: `numpy`, `scipy`, `matplotlib`. Uses the `Agg` backend (no display needed). Output plots are saved as PNG files — note the hardcoded output paths in `__main__` point to `/home/claude/`; update these to the desired local directory before running.

## Architecture

**Single-file simulation** (`holon_sim.py`):

- `Params` dataclass: All model parameters with physically motivated defaults (~30 parameters controlling generation, dissipation, transmutation, congestion, and infrastructure dynamics)
- `Scenario` dataclass: Practitioner-controlled driving variables (H, E, S, N, G, Rc, D_jing, R_qi, D_qi) plus initial conditions
- `holon_ode()`: The RHS of the 5-variable ODE system (phi_jing, phi_qi, phi_shen, C, I) with non-negativity enforcement
- `run_scenario()`: Integrates using `scipy.integrate.solve_ivp` with LSODA method at high tolerance (rtol=1e-8, atol=1e-10)
- `make_scenarios()`: Defines the six comparison scenarios from the paper
- Three plotting functions: `plot_results()` (time series), `plot_phase_portrait()` (C-I plane), `plot_kappa_surface()` (transmutation coefficient contours)

**Key coupling mechanism**: The transmutation coefficients kappa1_eff and kappa2_eff depend superlinearly on both C (coherence) and I (imagination), creating threshold behavior — below certain C/I values, energy cannot refine upward and dissipates instead.

**Event-driven simulation** (`holon_event_sim.py`):

Imports `Params`, `Scenario`, `holon_ode` from `holon_sim.py` — extends the model with time-varying scenarios. A scenario is a baseline `Scenario` plus a list of `Event` objects, each with a day and type:

- `SET`: Change driving variables to new values (e.g., "quit porn" → S=0, N=0)
- `PULSE`: Instantaneous state vector deltas (e.g., ejaculation → phi_jing -= 1.5)
- `RAMP`: Gradually transition variables over N days (linear interpolation)

Core function `run_event_scenario()` integrates segment-by-segment between events using the same LSODA solver, stitching results into a `StitchedResult` (compatible `.t` and `.y` arrays). For RAMP events, wraps `holon_ode` to update driving variables at each ODE evaluation.

```bash
python3 holon_event_sim.py
```

Outputs: `event_scenario_comparison.png`, `event_phase_portrait.png`

Six test scenarios: Recovery Journey (sequential SET), Relapse & Recovery (SET with reversal), Periodic Ejaculation (repeated PULSE), Partnership (SET + reciprocity), Gradual Cultivation (RAMP), Mixed Events (all three kinds).

## Theory Documents

- `holon-energy-model.md`: Primary paper. Derives the model from first principles, explains each equation, defines all variables and parameters.
- `dao-energy-model.md`: Alternative framing emphasizing the Daoist tradition and empirical validation against practitioner reports. More detailed on the dual role of phi_qi (catalytic vs. reactant).

Both documents describe the same mathematical model from different angles.

## Key Domain Concepts

- **phi_jing/qi/shen**: Three phases of one conserved field (dense to subtle), not three separate substances
- **C (coherence)** and **I (imagination)**: Infrastructure variables in [0,1] that gate transmutation — the model's core insight is that retention alone is insufficient without these
- **Congestion**: When phi_jing exceeds its natural equilibrium (Gamma_eff/lam_jing), excess creates stress on phi_qi via the sigma coupling term
- The six scenarios range from "Retention Only" (passive) through "Retention + Full Cultivation" (optimal) to "Retention + Porn" (worst case per the model)
