"""
Event-Driven Holon Simulation
==============================

Extends holon_sim.py with an event-driven architecture where scenarios are
defined as timelines of discrete events (practice changes, ejaculations,
gradual ramp-ups) rather than fixed driving variables.

Three event types:
  SET   — change driving variables to new values
  PULSE — instantaneous state vector deltas (e.g. ejaculation)
  RAMP  — gradually transition variables over N days
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from holon_sim import Params, Scenario, holon_ode


# ── Constants ────────────────────────────────────────────────────────────

DRIVING_VARS = {"H", "E", "S", "N", "G", "Rc", "D_jing", "R_qi", "D_qi"}
STATE_VARS = ("phi_jing", "phi_qi", "phi_shen", "C", "I")


# ── Event Types ──────────────────────────────────────────────────────────

class EventKind(Enum):
    SET = "set"
    PULSE = "pulse"
    RAMP = "ramp"


@dataclass
class Event:
    """A single event in the simulation timeline."""
    time: float
    kind: EventKind
    label: str = ""

    # SET: driving variable overrides
    changes: Dict[str, float] = field(default_factory=dict)

    # PULSE: state vector deltas (e.g. {"phi_jing": -1.5})
    deltas: Dict[str, float] = field(default_factory=dict)

    # RAMP: target values and duration in days
    targets: Dict[str, float] = field(default_factory=dict)
    duration: float = 0.0

    # Whether to show a vertical marker line on plots
    marker: bool = True


@dataclass
class EventScenario:
    """A scenario defined by a baseline plus a timeline of events."""
    name: str
    baseline: Scenario
    events: List[Event] = field(default_factory=list)
    t_end: float = 365.0


# ── Practice Event Templates ────────────────────────────────────────────
#
# Each function returns a PULSE Event for a single practice session.
# Effects are small per session but compound over weeks of regular practice.
#
# Positive practices:

def workout(day, label="Workout"):
    """Heavy resistance training or intense cardio. Boosts qi and jing."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": 0.20, "phi_jing": 0.05})

def yoga(day, label="Yoga"):
    """Hatha/vinyasa yoga. Boosts qi, small shen and coherence gains."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": 0.12, "phi_shen": 0.01, "C": 0.008})

def meditate(day, label="Meditate"):
    """Seated meditation (20-45 min). Primary shen builder, coherence gain."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_shen": 0.03, "phi_qi": 0.05, "C": 0.012, "I": 0.005})

def breathwork(day, label="Breathwork"):
    """Pranayama, Wim Hof, or holotropic. Strong qi activation."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": 0.18, "phi_shen": 0.01, "C": 0.006, "I": 0.003})

def cold_exposure(day, label="Cold exposure"):
    """Cold shower or ice bath. Acute qi and jing boost, small coherence."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": 0.10, "phi_jing": 0.03, "C": 0.004})

def qigong(day, label="Qigong"):
    """Qigong or tai chi. Balanced qi and shen, good coherence builder."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": 0.15, "phi_shen": 0.02, "C": 0.010, "I": 0.004})

def creative_work(day, label="Creative work"):
    """Writing, music, art — self-generated creative output. Builds I."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": 0.05, "phi_shen": 0.02, "I": 0.010})

def heart_practice(day, label="Heart practice"):
    """Devotional practice, metta, prayer. Primary coherence builder."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_shen": 0.04, "phi_qi": 0.03, "C": 0.015, "I": 0.006})

def nature_walk(day, label="Nature"):
    """Extended time in nature. Gentle qi and coherence support."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": 0.08, "phi_shen": 0.01, "C": 0.005, "I": 0.003})

def journaling(day, label="Journal"):
    """Reflective journaling. Builds imagination and small shen."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": 0.03, "phi_shen": 0.02, "C": 0.005, "I": 0.008})

# Negative events:

def poor_sleep(day, label="Bad sleep"):
    """Poor or insufficient sleep. Drains qi and shen."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": -0.10, "phi_shen": -0.01, "C": -0.003})

def alcohol(day, label="Alcohol"):
    """Drinking session. Damages qi, coherence, and imagination."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": -0.15, "phi_shen": -0.02, "C": -0.010, "I": -0.005})

def doom_scrolling(day, label="Doom scroll"):
    """Extended social media / news binge. Drains imagination and qi."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": -0.05, "phi_shen": -0.01, "C": -0.003, "I": -0.006})

def argument(day, label="Conflict"):
    """Heated argument or interpersonal conflict. Drains qi and coherence."""
    return Event(time=day, kind=EventKind.PULSE, label=label, marker=False,
                 deltas={"phi_qi": -0.10, "phi_shen": -0.01, "C": -0.008})


# ── Scheduling Helper ───────────────────────────────────────────────────

def schedule(practice_fn, start, end, every, label=None):
    """Generate repeated practice events at regular intervals.

    Args:
        practice_fn: One of the practice template functions (workout, meditate, etc.)
        start: First day (inclusive)
        end: Last day (inclusive)
        every: Interval in days between sessions
        label: Optional label override (otherwise uses the function's default)
    """
    events = []
    day = float(start)
    while day <= end:
        kwargs = {"day": day}
        if label:
            kwargs["label"] = label
        events.append(practice_fn(**kwargs))
        day += every
    return events


# ── Mutable Scenario Proxy ──────────────────────────────────────────────

class MutableScenario:
    """Mutable proxy for Scenario that holon_ode reads via attribute access.

    Supports instant mutation (SET events) and time-interpolated ramps.
    """

    def __init__(self, base: Scenario):
        for var in DRIVING_VARS:
            object.__setattr__(self, var, getattr(base, var))
        self._ramps: Dict[str, tuple] = {}

    def set_var(self, name: str, value: float):
        object.__setattr__(self, name, value)
        self._ramps.pop(name, None)

    def add_ramp(self, name: str, target: float, t_start: float, t_end: float):
        start_val = getattr(self, name)
        self._ramps[name] = (start_val, target, t_start, t_end)

    def update_for_time(self, t: float):
        """Update ramped attributes to their interpolated values at time t."""
        for name, (v0, v1, ts, te) in list(self._ramps.items()):
            if t >= te:
                object.__setattr__(self, name, v1)
                del self._ramps[name]
            elif t >= ts:
                frac = (t - ts) / (te - ts)
                object.__setattr__(self, name, v0 + frac * (v1 - v0))

    @property
    def has_active_ramps(self):
        return len(self._ramps) > 0


# ── Stitched Result ─────────────────────────────────────────────────────

class StitchedResult:
    """Mimics scipy OdeResult with .t and .y from stitched segments."""

    def __init__(self, t: np.ndarray, y: np.ndarray, events_log: List[dict]):
        self.t = t
        self.y = y
        self.success = True
        self.events_log = events_log


# ── Core Integration ─────────────────────────────────────────────────────

def _make_ramp_ode(p, ms):
    """Wrap holon_ode so ramped driving variables update each evaluation."""
    def ode_with_ramp(t, y):
        ms.update_for_time(t)
        return holon_ode(t, y, p, ms)
    return ode_with_ramp


def run_event_scenario(
    p: Params,
    es: EventScenario,
    t_end: float = 108.0,
    points_per_day: int = 20,
) -> StitchedResult:
    """Integrate the ODE system with events modifying driving variables."""

    events = sorted(es.events, key=lambda e: e.time)
    baseline = es.baseline

    # Collect all boundary times (event times + ramp end times)
    boundary_set = {0.0, t_end}
    for e in events:
        if 0.0 <= e.time <= t_end:
            boundary_set.add(e.time)
        if e.kind == EventKind.RAMP:
            ramp_end = e.time + e.duration
            if 0.0 < ramp_end < t_end:
                boundary_set.add(ramp_end)
    boundaries = sorted(boundary_set)

    # Initialize
    ms = MutableScenario(baseline)
    y_current = np.array([
        baseline.phi_jing_0, baseline.phi_qi_0, baseline.phi_shen_0,
        baseline.C_0, baseline.I_0,
    ])

    t_segments = []
    y_segments = []
    events_log = []

    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_stop = boundaries[i + 1]

        # Apply all events at t_start
        for ev in events:
            if ev.time != t_start:
                continue

            if ev.kind == EventKind.SET:
                for var, val in ev.changes.items():
                    ms.set_var(var, val)
                events_log.append({"time": t_start, "label": ev.label,
                                   "kind": "set", "marker": ev.marker})

            elif ev.kind == EventKind.PULSE:
                for state_name, delta in ev.deltas.items():
                    idx = STATE_VARS.index(state_name)
                    y_current[idx] = max(0.0, y_current[idx] + delta)
                    if idx >= 3:  # C, I bounded [0, 1]
                        y_current[idx] = min(1.0, y_current[idx])
                events_log.append({"time": t_start, "label": ev.label,
                                   "kind": "pulse", "marker": ev.marker})

            elif ev.kind == EventKind.RAMP:
                for var, target in ev.targets.items():
                    ms.add_ramp(var, target, t_start, t_start + ev.duration)
                events_log.append({"time": t_start, "label": ev.label,
                                   "kind": "ramp", "marker": ev.marker})

        # Skip zero-length segments
        if t_stop <= t_start:
            continue

        seg_duration = t_stop - t_start
        n_pts = max(int(seg_duration * points_per_day), 10)
        t_eval = np.linspace(t_start, t_stop, n_pts)

        if ms.has_active_ramps:
            rhs = _make_ramp_ode(p, ms)
            sol = solve_ivp(rhs, (t_start, t_stop), y_current,
                            method="LSODA", t_eval=t_eval,
                            rtol=1e-8, atol=1e-10, max_step=0.5)
        else:
            sol = solve_ivp(holon_ode, (t_start, t_stop), y_current,
                            args=(p, ms), method="LSODA", t_eval=t_eval,
                            rtol=1e-8, atol=1e-10, max_step=0.5)

        if not sol.success:
            print(f"  WARNING: segment [{t_start:.1f}, {t_stop:.1f}] failed: {sol.message}")

        # Store segment (drop last point to avoid duplication, except final)
        if i < len(boundaries) - 2:
            t_segments.append(sol.t[:-1])
            y_segments.append(sol.y[:, :-1])
        else:
            t_segments.append(sol.t)
            y_segments.append(sol.y)

        y_current = sol.y[:, -1].copy()

    return StitchedResult(
        t=np.concatenate(t_segments),
        y=np.hstack(y_segments),
        events_log=events_log,
    )


# ── Scenarios ────────────────────────────────────────────────────────────

# Rock-bottom initial conditions: steady state of the addiction baseline
# (S=2.0, N=0.5, D_jing=0.7, D_qi=0.15) run to convergence (5000 days).
# Total depletion — the spring barely trickles, infrastructure destroyed.
def _rock_bottom_ic():
    return dict(phi_jing_0=0.006, phi_qi_0=0.005, phi_shen_0=0.0, C_0=0.0, I_0=0.0)

# The active addiction baseline driving variables
_ADDICTION_DRIVES = dict(H=0.0, E=0.05, S=2.0, N=0.5, G=0.0, Rc=0.0,
                         D_jing=0.7, R_qi=0.05, D_qi=0.15)


def make_the_slow_climb():
    """Classic recovery: rock bottom → many false starts → habits gradually stick.

    Months 1-2: Raw willpower, frequent slips. Doom scrolling replaces porn.
    Months 3-4: Starts gym, nature walks. Fewer slips.
    Months 5-7: Adds breathwork and meditation. One major relapse.
    Months 8-12: Deepening practice, rare slips, meets partner.
    """
    narrative = [
        # Day 0: Decides to quit. Stops porn and ejaculation cold turkey.
        Event(time=0, kind=EventKind.SET, label="Quit cold turkey",
              changes={"S": 0.0, "N": 0.0, "D_jing": 0.0, "D_qi": 0.0, "E": 0.15}),

        # Week 2: First slip — masturbation without porn
        Event(time=12, kind=EventKind.PULSE, label="MO slip",
              deltas={"phi_jing": -1.0, "phi_qi": -0.2}),

        # Week 4: Another slip
        Event(time=25, kind=EventKind.PULSE, label="MO slip",
              deltas={"phi_jing": -1.0, "phi_qi": -0.2}),

        # Day 35: Starts going to the gym
        Event(time=35, kind=EventKind.SET, label="Start gym",
              changes={"E": 0.45, "R_qi": 0.15}),

        # Week 7: Slip with porn (brief, 1 day)
        Event(time=48, kind=EventKind.SET, label="Porn slip",
              changes={"S": 1.0, "D_jing": 0.5}),
        Event(time=49, kind=EventKind.SET, label="Back on track",
              changes={"S": 0.0, "D_jing": 0.0}),

        # Day 60: Slip — ejaculation only
        Event(time=60, kind=EventKind.PULSE, label="MO slip",
              deltas={"phi_jing": -1.2, "phi_qi": -0.2}),

        # Day 75: Gym becomes consistent, adds cold showers
        Event(time=75, kind=EventKind.SET, label="Consistent gym + cold",
              changes={"E": 0.6, "R_qi": 0.25}),

        # Day 95: Tries breathwork for the first time
        Event(time=95, kind=EventKind.RAMP, label="Learn breathwork",
              targets={"G": 0.25, "E": 0.7}, duration=20.0),

        # Day 110: Slip — ejaculation
        Event(time=110, kind=EventKind.PULSE, label="MO slip",
              deltas={"phi_jing": -1.0, "phi_qi": -0.15}),

        # Day 130: Starts meditation (short sessions)
        Event(time=130, kind=EventKind.SET, label="Start meditation",
              changes={"H": 0.2, "G": 0.35}),

        # Day 155: Major relapse — 5-day porn binge during stressful period
        Event(time=155, kind=EventKind.SET, label="Stress relapse",
              changes={"S": 1.8, "N": 0.5, "D_jing": 0.6, "E": 0.2,
                        "H": 0.0, "G": 0.0, "D_qi": 0.1}),
        Event(time=160, kind=EventKind.SET, label="End binge, recommit",
              changes={"S": 0.0, "N": 0.0, "D_jing": 0.0, "D_qi": 0.0,
                        "E": 0.5, "H": 0.1, "G": 0.15, "R_qi": 0.2}),

        # Day 180: Rebuilding — back to gym and breathwork
        Event(time=180, kind=EventKind.RAMP, label="Rebuild practices",
              targets={"E": 0.75, "H": 0.35, "G": 0.45, "R_qi": 0.3},
              duration=30.0),

        # Day 220: Deepening meditation, adds visualization
        Event(time=220, kind=EventKind.SET, label="Deepen meditation",
              changes={"H": 0.55, "G": 0.6}),

        # Day 240: Slip — single ejaculation, no porn
        Event(time=240, kind=EventKind.PULSE, label="MO slip",
              deltas={"phi_jing": -0.8, "phi_qi": -0.1}),

        # Day 270: Full cultivation practice established
        Event(time=270, kind=EventKind.SET, label="Full cultivation",
              changes={"H": 0.75, "E": 0.85, "G": 0.8, "R_qi": 0.3}),

        # Day 310: Meets partner, begins reciprocal practice
        Event(time=310, kind=EventKind.SET, label="Meet partner",
              changes={"Rc": 0.3, "H": 0.8}),

        # Day 340: Deepening partnership
        Event(time=340, kind=EventKind.SET, label="Deepen bond",
              changes={"Rc": 0.5, "H": 0.85, "E": 0.9}),
    ]

    practices = (
        # Early weeks: doom scrolling replaces porn
        schedule(doom_scrolling, 1, 30, every=3) +
        # Nature walks before gym (low barrier entry)
        schedule(nature_walk, 10, 35, every=5) +
        # Gym phase: 3x/week from day 35 (gap during relapse 155-175)
        schedule(workout, 35, 153, every=3) +
        schedule(workout, 180, 365, every=3) +
        # Cold exposure from day 75
        schedule(cold_exposure, 75, 153, every=3) +
        schedule(cold_exposure, 180, 365, every=3) +
        # Breathwork from day 95 (2x/week, gap during relapse)
        schedule(breathwork, 100, 153, every=4) +
        schedule(breathwork, 185, 365, every=4) +
        # Meditation from day 130 (daily-ish, gap during relapse)
        schedule(meditate, 132, 153, every=2) +
        schedule(meditate, 165, 365, every=2) +
        # Journaling during dark periods
        schedule(journaling, 5, 35, every=5) +
        schedule(journaling, 160, 200, every=4) +
        # Yoga from month 7
        schedule(yoga, 210, 365, every=4) +
        # Heart practice from month 9
        schedule(heart_practice, 270, 365, every=3) +
        # Creative work once things stabilize
        schedule(creative_work, 250, 365, every=5) +
        # Occasional bad sleep early on (withdrawal)
        schedule(poor_sleep, 2, 30, every=5) +
        # Alcohol a few times early on
        [alcohol(18), alcohol(40), alcohol(70)]
    )

    return EventScenario(
        name="The Slow Climb",
        baseline=Scenario(name="The Slow Climb", **_ADDICTION_DRIVES, **_rock_bottom_ic()),
        events=narrative + practices,
    )


def make_two_steps_forward():
    """Strong start but life intervenes: breakup, depression, recovery through community.

    Months 1-3: Disciplined retention + physical practice, rapid gains.
    Month 4: Breakup triggers emotional collapse and partial relapse.
    Months 5-6: Depressed, minimal practice, doom scrolling, poor sleep.
    Month 7: Joins men's group / community, reciprocity begins.
    Months 8-12: Gradual rebuild with community support, stronger than before.
    """
    narrative = [
        # Day 0: Clean break — motivated by hitting rock bottom
        Event(time=0, kind=EventKind.SET, label="Clean break",
              changes={"S": 0.0, "N": 0.0, "D_jing": 0.0, "D_qi": 0.0,
                        "E": 0.3, "R_qi": 0.15}),

        # Day 14: Starts running + bodyweight training
        Event(time=14, kind=EventKind.SET, label="Start training",
              changes={"E": 0.65, "R_qi": 0.25}),

        # Day 30: Adds cold exposure and basic breathwork
        Event(time=30, kind=EventKind.SET, label="Cold + breathwork",
              changes={"E": 0.75, "G": 0.3, "R_qi": 0.3}),

        # Day 50: Feeling great, starts meditation
        Event(time=50, kind=EventKind.SET, label="Start meditation",
              changes={"H": 0.4, "G": 0.5}),

        # Day 75: Peak — everything flowing
        Event(time=75, kind=EventKind.SET, label="Peak practice",
              changes={"H": 0.6, "E": 0.8, "G": 0.65}),

        # Day 105: Breakup. Emotional devastation.
        Event(time=105, kind=EventKind.SET, label="Breakup",
              changes={"H": 0.0, "E": 0.3, "G": 0.1, "R_qi": 0.1, "D_qi": 0.08}),

        # Day 112: First relapse — porn + ejaculation
        Event(time=112, kind=EventKind.SET, label="Relapse: porn + ejac",
              changes={"S": 1.2, "N": 0.3, "D_jing": 0.5, "D_qi": 0.1}),

        # Day 118: Stops porn but keeps masturbating intermittently
        Event(time=118, kind=EventKind.SET, label="Stop porn, still MO",
              changes={"S": 0.0, "N": 0.0, "D_jing": 0.25, "D_qi": 0.0}),

        # Day 130: Another porn slip (2 days)
        Event(time=130, kind=EventKind.SET, label="Porn slip",
              changes={"S": 1.0, "D_jing": 0.4}),
        Event(time=132, kind=EventKind.SET, label="Stop again",
              changes={"S": 0.0, "D_jing": 0.15}),

        # Day 150: Depression bottom — minimal everything
        Event(time=150, kind=EventKind.SET, label="Depression bottom",
              changes={"E": 0.15, "G": 0.0, "H": 0.0, "R_qi": 0.08,
                        "D_jing": 0.1}),

        # Day 180: Finds men's group, begins recovery
        Event(time=180, kind=EventKind.SET, label="Join men's group",
              changes={"D_jing": 0.0, "Rc": 0.15, "H": 0.1, "E": 0.3}),

        # Day 200: Group accountability helps — back to gym
        Event(time=200, kind=EventKind.RAMP, label="Rebuild with community",
              targets={"E": 0.7, "H": 0.3, "G": 0.35, "R_qi": 0.25, "Rc": 0.3},
              duration=40.0),

        # Day 250: Single ejaculation slip, no porn
        Event(time=250, kind=EventKind.PULSE, label="MO slip",
              deltas={"phi_jing": -0.8, "phi_qi": -0.1}),

        # Day 270: Meditation deepening with group
        Event(time=270, kind=EventKind.SET, label="Group meditation",
              changes={"H": 0.5, "G": 0.6, "Rc": 0.4}),

        # Day 310: Full practice restored, stronger foundation
        Event(time=310, kind=EventKind.SET, label="Full practice + community",
              changes={"H": 0.7, "E": 0.85, "G": 0.75, "R_qi": 0.3, "Rc": 0.5}),

        # Day 345: Deepening — heart practice opens
        Event(time=345, kind=EventKind.SET, label="Heart opening",
              changes={"H": 0.85, "G": 0.85}),
    ]

    practices = (
        # Strong initial training block (months 1-3)
        schedule(workout, 14, 103, every=2) +
        schedule(cold_exposure, 30, 103, every=3) +
        schedule(breathwork, 32, 103, every=3) +
        schedule(meditate, 52, 103, every=2) +
        schedule(yoga, 60, 103, every=5) +
        schedule(nature_walk, 20, 103, every=7) +
        # Peak period: add heart practice
        schedule(heart_practice, 75, 103, every=3) +
        # Breakup devastation: doom scrolling, poor sleep, alcohol
        schedule(doom_scrolling, 106, 170, every=2) +
        schedule(poor_sleep, 106, 170, every=3) +
        [alcohol(108), alcohol(115), alcohol(125), alcohol(140), alcohol(155)] +
        # Depression: occasional walks are all he can manage
        schedule(nature_walk, 135, 178, every=7) +
        schedule(journaling, 150, 180, every=5) +
        # Rebuild with community support (month 7+)
        schedule(workout, 200, 365, every=3) +
        schedule(nature_walk, 185, 365, every=7) +
        schedule(breathwork, 210, 365, every=4) +
        schedule(cold_exposure, 215, 365, every=3) +
        schedule(meditate, 220, 365, every=2) +
        schedule(yoga, 240, 365, every=5) +
        schedule(heart_practice, 270, 365, every=3) +
        schedule(creative_work, 280, 365, every=5) +
        schedule(journaling, 200, 365, every=7)
    )

    return EventScenario(
        name="Two Steps Forward",
        baseline=Scenario(name="Two Steps Forward", **_ADDICTION_DRIVES, **_rock_bottom_ic()),
        events=narrative + practices,
    )


def make_the_grinder():
    """Muscular approach: retention + heavy training but no inner work for months.

    Months 1-4: Brute force retention + intense gym, cold. No meditation, no heart.
        Congestion builds. Periodic ejaculation from pressure.
    Month 5: Agitation peaks. Starts breathwork out of desperation.
    Months 6-8: Grudgingly adds meditation. Agitation starts to resolve.
    Months 9-12: Finally integrates heart practice. Transformation accelerates.
    """
    narrative = [
        # Day 0: Quit porn and ejaculation, immediately hit the gym hard
        Event(time=0, kind=EventKind.SET, label="Quit + gym hard",
              changes={"S": 0.0, "N": 0.0, "D_jing": 0.0, "D_qi": 0.0,
                        "E": 0.7, "R_qi": 0.35}),

        # Pressure releases from congestion (no inner practice to transmute)
        Event(time=21, kind=EventKind.PULSE, label="Pressure release",
              deltas={"phi_jing": -1.5, "phi_qi": -0.3}),
        Event(time=45, kind=EventKind.PULSE, label="Pressure release",
              deltas={"phi_jing": -1.5, "phi_qi": -0.3}),

        # Day 65: Getting agitated but doubles down on training
        Event(time=65, kind=EventKind.SET, label="Train harder",
              changes={"E": 0.85, "R_qi": 0.35}),

        Event(time=75, kind=EventKind.PULSE, label="Pressure release",
              deltas={"phi_jing": -1.3, "phi_qi": -0.25}),
        Event(time=95, kind=EventKind.PULSE, label="Pressure release",
              deltas={"phi_jing": -1.5, "phi_qi": -0.3}),

        # Day 110: Near-relapse with porn, catches himself
        Event(time=110, kind=EventKind.SET, label="Brief porn peek",
              changes={"S": 0.5}),
        Event(time=111, kind=EventKind.SET, label="Caught himself",
              changes={"S": 0.0}),
        Event(time=111, kind=EventKind.PULSE, label="Ejac from edging",
              deltas={"phi_jing": -1.5, "phi_qi": -0.4}),

        # Day 130: Desperation — starts breathwork
        Event(time=130, kind=EventKind.RAMP, label="Learn breathwork",
              targets={"G": 0.3, "E": 0.8}, duration=20.0),

        # Day 170: Grudgingly tries meditation (short sessions)
        Event(time=170, kind=EventKind.SET, label="Try meditation",
              changes={"H": 0.15, "G": 0.4}),

        # Day 200: Meditation becoming regular
        Event(time=200, kind=EventKind.RAMP, label="Deepen meditation",
              targets={"H": 0.4, "G": 0.55}, duration=30.0),

        # Day 240: Slip — ejaculation during stress
        Event(time=240, kind=EventKind.PULSE, label="Stress ejac",
              deltas={"phi_jing": -1.0, "phi_qi": -0.15}),

        # Day 260: Finally opens to heart practice
        Event(time=260, kind=EventKind.RAMP, label="Heart practice begins",
              targets={"H": 0.65, "G": 0.7}, duration=30.0),

        # Day 310: Integration — everything connects
        Event(time=310, kind=EventKind.SET, label="Full integration",
              changes={"H": 0.8, "E": 0.85, "G": 0.8, "R_qi": 0.3}),

        # Day 350: Deepening
        Event(time=350, kind=EventKind.SET, label="Deepening",
              changes={"H": 0.85, "G": 0.85}),
    ]

    practices = (
        # Intense gym from day 1 — this guy never misses a workout
        schedule(workout, 1, 365, every=2) +
        # Cold exposure from day 1 — hardcore
        schedule(cold_exposure, 1, 365, every=2) +
        # No breathwork/meditation/heart until forced by agitation
        # Breathwork from day 135
        schedule(breathwork, 135, 365, every=3) +
        # Meditation reluctantly from day 170
        schedule(meditate, 172, 365, every=3) +
        # Qigong from day 200 (trainer suggests it)
        schedule(qigong, 200, 365, every=5) +
        # Heart practice from day 260
        schedule(heart_practice, 265, 365, every=3) +
        # Yoga only in the final months
        schedule(yoga, 300, 365, every=4) +
        # Some insomnia from agitation in months 2-4
        schedule(poor_sleep, 30, 120, every=5) +
        # Nature on weekends (even grinders hike)
        schedule(nature_walk, 15, 365, every=7)
    )

    return EventScenario(
        name="The Grinder",
        baseline=Scenario(name="The Grinder", **_ADDICTION_DRIVES, **_rock_bottom_ic()),
        events=narrative + practices,
    )


def make_the_revolving_door():
    """Chronic relapse pattern: knows what to do but can't sustain it.

    Multiple 30-60 day streaks followed by multi-day binges.
    Each relapse is slightly less severe. Each recovery slightly faster.
    Practice is inconsistent — gym on and off, meditation attempted and dropped.
    By month 10, streaks finally start holding. Barely crosses threshold by year end.
    """
    narrative = [
        # === Attempt 1: 25 days, no practice ===
        Event(time=0, kind=EventKind.SET, label="Attempt 1",
              changes={"S": 0.0, "N": 0.0, "D_jing": 0.0, "D_qi": 0.0,
                        "E": 0.3, "R_qi": 0.1}),
        Event(time=25, kind=EventKind.SET, label="Binge relapse",
              changes={"S": 2.0, "N": 0.5, "D_jing": 0.7, "D_qi": 0.15,
                        "E": 0.05}),
        Event(time=30, kind=EventKind.SET, label="Binge over",
              changes={"S": 0.8, "N": 0.3, "D_jing": 0.4, "D_qi": 0.08}),
        Event(time=42, kind=EventKind.SET, label="Tapering",
              changes={"S": 0.3, "D_jing": 0.2, "D_qi": 0.03}),

        # === Attempt 2: 40 days, tries gym ===
        Event(time=55, kind=EventKind.SET, label="Attempt 2 + gym",
              changes={"S": 0.0, "N": 0.0, "D_jing": 0.0, "D_qi": 0.0,
                        "E": 0.5, "R_qi": 0.2}),
        Event(time=95, kind=EventKind.SET, label="Relapse (3 days)",
              changes={"S": 1.5, "N": 0.4, "D_jing": 0.6, "D_qi": 0.1,
                        "E": 0.1}),
        Event(time=98, kind=EventKind.SET, label="Stop + guilt",
              changes={"S": 0.0, "N": 0.0, "D_jing": 0.0, "D_qi": 0.0,
                        "E": 0.2, "R_qi": 0.1}),

        # === Attempt 3: 55 days, gym + some breathwork ===
        Event(time=110, kind=EventKind.SET, label="Attempt 3",
              changes={"E": 0.5, "R_qi": 0.2, "G": 0.1}),
        Event(time=130, kind=EventKind.SET, label="Add breathwork",
              changes={"E": 0.6, "G": 0.25}),
        Event(time=155, kind=EventKind.PULSE, label="MO slip",
              deltas={"phi_jing": -1.2, "phi_qi": -0.2}),
        Event(time=165, kind=EventKind.SET, label="Relapse (2 days)",
              changes={"S": 1.2, "D_jing": 0.5, "D_qi": 0.08}),
        Event(time=167, kind=EventKind.SET, label="Stop",
              changes={"S": 0.0, "D_jing": 0.0, "D_qi": 0.0}),

        # === Attempt 4: 75 days, more complete practice ===
        Event(time=180, kind=EventKind.SET, label="Attempt 4",
              changes={"E": 0.55, "R_qi": 0.2, "G": 0.2, "H": 0.1}),
        Event(time=200, kind=EventKind.RAMP, label="Build practices",
              targets={"E": 0.7, "G": 0.4, "H": 0.3, "R_qi": 0.25},
              duration=30.0),
        Event(time=240, kind=EventKind.PULSE, label="MO slip",
              deltas={"phi_jing": -0.8, "phi_qi": -0.1}),
        Event(time=255, kind=EventKind.SET, label="Minor relapse (1 day)",
              changes={"S": 0.8, "D_jing": 0.3}),
        Event(time=256, kind=EventKind.SET, label="Recover fast",
              changes={"S": 0.0, "D_jing": 0.0, "E": 0.6, "G": 0.35, "H": 0.25}),

        # === Attempt 5: Finally holds ===
        Event(time=275, kind=EventKind.SET, label="Attempt 5",
              changes={"E": 0.65, "R_qi": 0.25, "G": 0.4, "H": 0.3}),
        Event(time=300, kind=EventKind.RAMP, label="Gradual deepening",
              targets={"E": 0.8, "G": 0.6, "H": 0.5, "R_qi": 0.3},
              duration=40.0),
        Event(time=330, kind=EventKind.PULSE, label="MO slip (last one)",
              deltas={"phi_jing": -0.6, "phi_qi": -0.05}),
        Event(time=350, kind=EventKind.SET, label="Stable practice",
              changes={"H": 0.6, "E": 0.8, "G": 0.65}),
    ]

    practices = (
        # Attempt 1: just doom scrolling and poor sleep (no practice)
        schedule(doom_scrolling, 1, 24, every=2) +
        schedule(poor_sleep, 3, 24, every=4) +
        # Binge period: heavy doom scrolling
        schedule(doom_scrolling, 25, 50, every=2) +
        [alcohol(26), alcohol(32), alcohol(45)] +

        # Attempt 2: inconsistent gym
        schedule(workout, 57, 93, every=4) +
        schedule(nature_walk, 60, 93, every=7) +
        # Relapse period
        schedule(doom_scrolling, 95, 108, every=2) +
        [alcohol(96), alcohol(100)] +

        # Attempt 3: gym + breathwork (more consistent)
        schedule(workout, 112, 163, every=3) +
        schedule(breathwork, 132, 163, every=4) +
        schedule(journaling, 115, 163, every=7) +
        # Relapse — brief
        schedule(doom_scrolling, 165, 178, every=3) +

        # Attempt 4: gym + breathwork + tries meditation
        schedule(workout, 182, 253, every=3) +
        schedule(breathwork, 190, 253, every=4) +
        schedule(meditate, 200, 253, every=3) +
        schedule(nature_walk, 185, 253, every=7) +
        schedule(journaling, 185, 253, every=7) +
        # Minor relapse
        [doom_scrolling(255)] +

        # Attempt 5: full practice (finally sticks)
        schedule(workout, 277, 365, every=3) +
        schedule(breathwork, 280, 365, every=4) +
        schedule(meditate, 280, 365, every=2) +
        schedule(cold_exposure, 285, 365, every=3) +
        schedule(yoga, 300, 365, every=5) +
        schedule(heart_practice, 310, 365, every=4) +
        schedule(nature_walk, 280, 365, every=7) +
        schedule(creative_work, 320, 365, every=5) +
        schedule(journaling, 280, 365, every=7)
    )

    return EventScenario(
        name="The Revolving Door",
        baseline=Scenario(name="The Revolving Door", **_ADDICTION_DRIVES, **_rock_bottom_ic()),
        events=narrative + practices,
    )


# ── Plotting ─────────────────────────────────────────────────────────────

EVENT_MARKER_STYLES = {
    "set":   {"color": "#4488ff", "linestyle": "--", "alpha": 0.4},
    "pulse": {"color": "#ff4444", "linestyle": ":",  "alpha": 0.5},
    "ramp":  {"color": "#44cc88", "linestyle": "-.", "alpha": 0.4},
}


def _add_event_markers(ax, events_log):
    """Draw vertical lines and labels at event times (only for marker=True events)."""
    for entry in events_log:
        if not entry.get("marker", True):
            continue
        style = EVENT_MARKER_STYLES.get(entry["kind"], {})
        ax.axvline(entry["time"], linewidth=0.8, **style)
        if entry.get("label"):
            ax.text(entry["time"] + 0.5, ax.get_ylim()[1] * 0.95,
                    entry["label"], rotation=90, fontsize=6.5,
                    color=style.get("color", "#888"),
                    va="top", ha="left", alpha=0.7)


def plot_single_scenario(name, res, filename, t_end=108, color="#4169E1"):
    """Generate a 6-panel plot for a single event-driven scenario."""

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.patch.set_facecolor("#0a0a0f")

    var_names = [
        (r"$\varphi_{jing}$ — Generative Essence", 0),
        (r"$\varphi_{qi}$ — Vitality", 1),
        (r"$\varphi_{shen}$ — Awareness", 2),
        ("C — Circuit Coherence", 3),
        ("I — Imaginative Capacity", 4),
    ]

    def _style_ax(ax):
        ax.set_facecolor("#0d0d14")
        ax.set_xlabel("Days", color="#887e70", fontsize=10)
        ax.tick_params(colors="#665e52", labelsize=9)
        ax.spines["bottom"].set_color("#2a2a35")
        ax.spines["left"].set_color("#2a2a35")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.1, color="#444455")

    for idx, (title, var_idx) in enumerate(var_names):
        ax = axes[idx // 2, idx % 2]
        _style_ax(ax)
        ax.plot(res.t, res.y[var_idx], color=color, linewidth=2.0, alpha=0.9)
        ax.set_title(title, color="#c8c0b8", fontsize=13, pad=10)
        if hasattr(res, "events_log"):
            _add_event_markers(ax, res.events_log)

    # Total Phi in the sixth panel
    ax = axes[2, 1]
    _style_ax(ax)
    total_phi = res.y[0] + res.y[1] + res.y[2]
    ax.plot(res.t, total_phi, color=color, linewidth=2.0, alpha=0.9)
    ax.set_title(r"$\Phi_{total}$ — Total Creative Energy", color="#c8c0b8",
                 fontsize=13, pad=10)
    if hasattr(res, "events_log"):
        _add_event_markers(ax, res.events_log)

    fig.suptitle(f"{name} ({int(t_end)} Days)",
                 color="#e8ddd0", fontsize=16, y=0.995, fontweight="light")

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0f", edgecolor="none")
    plt.close()
    print(f"Saved: {filename}")


def plot_event_phase_portrait(results, filename):
    """C-I phase portrait for event-driven scenarios."""

    n_scenarios = len(results)
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(n_scenarios)
    colors = {name: matplotlib.colors.rgb2hex(cmap(i))
              for i, (name, _) in enumerate(results)}

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0d0d14")

    # Draw threshold curves (same as holon_sim)
    p = Params()
    phi_qi_ref = 2.5
    f_ref = p.f_min + (1.0 - p.f_min) * phi_qi_ref / (phi_qi_ref + p.Kf)
    g_ref = p.g_min + (1.0 - p.g_min) * phi_qi_ref / (phi_qi_ref + p.Kg)

    kappa1_thresh = 0.08
    kappa2_thresh = 0.04
    C_range = np.linspace(0.01, 1.0, 200)

    I_thresh1 = (kappa1_thresh / (p.kappa10 * C_range**p.a1 * f_ref)) ** (1.0 / p.b1)
    I_thresh1 = np.clip(I_thresh1, 0, 1.0)
    I_thresh2 = (kappa2_thresh / (p.kappa20 * C_range**p.a2 * g_ref)) ** (1.0 / p.b2)
    I_thresh2 = np.clip(I_thresh2, 0, 1.0)

    ax.fill_between(C_range, I_thresh2, 1.0, where=(I_thresh2 < 1.0),
                     alpha=0.08, color="#4169E1", label="Full Circulation Regime")
    ax.fill_between(C_range, I_thresh1, I_thresh2, where=(I_thresh1 < I_thresh2),
                     alpha=0.06, color="#2E8B57", label="Vitality-Only Regime")
    ax.fill_between(C_range, 0, I_thresh1, where=(I_thresh1 > 0),
                     alpha=0.05, color="#8B0000", label="Dissipative Regime")

    ax.plot(C_range, I_thresh1, color="#2E8B57", linewidth=1.5,
            linestyle="--", alpha=0.6, label=r"$\kappa_1$ threshold")
    ax.plot(C_range, I_thresh2, color="#4169E1", linewidth=1.5,
            linestyle="--", alpha=0.6, label=r"$\kappa_2$ threshold")

    for name, res in results:
        c = colors[name]
        C_traj = res.y[3]
        I_traj = res.y[4]

        ax.plot(C_traj, I_traj, color=c, linewidth=2.0, alpha=0.8)
        ax.plot(C_traj[0], I_traj[0], 'o', color=c, markersize=6, alpha=0.8)
        ax.plot(C_traj[-1], I_traj[-1], 's', color=c, markersize=8, alpha=0.9)
        ax.annotate(name, (C_traj[-1], I_traj[-1]),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=8, color=c, alpha=0.9)

        # Mark event times as dots along the trajectory (only marker=True events)
        if hasattr(res, "events_log"):
            for entry in res.events_log:
                if not entry.get("marker", True):
                    continue
                t_event = entry["time"]
                idx = np.argmin(np.abs(res.t - t_event))
                m = "D" if entry["kind"] == "pulse" else "^"
                ax.plot(C_traj[idx], I_traj[idx], m, color=c,
                        markersize=7, alpha=0.9, markeredgecolor="white",
                        markeredgewidth=0.5)

    ax.set_xlabel("C — Circuit Coherence", color="#887e70", fontsize=12)
    ax.set_ylabel("I — Imaginative Capacity", color="#887e70", fontsize=12)
    ax.set_title("Phase Portrait: C-I Plane (Event-Driven Scenarios)",
                  color="#e8ddd0", fontsize=14, pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#665e52", labelsize=10)
    ax.spines["bottom"].set_color("#2a2a35")
    ax.spines["left"].set_color("#2a2a35")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.08, color="#444455")

    ax.legend(fontsize=9, loc="upper left", frameon=False, labelcolor="#c8c0b8")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0f", edgecolor="none")
    plt.close()
    print(f"Saved: {filename}")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = Params()

    scenario_makers = [
        make_the_slow_climb,
        make_two_steps_forward,
        make_the_grinder,
        make_the_revolving_door,
    ]

    scenario_colors = [
        "#4169E1",  # The Slow Climb — royal blue
        "#2E8B57",  # Two Steps Forward — seagreen
        "#B8860B",  # The Grinder — darkgoldenrod
        "#C71585",  # The Revolving Door — mediumvioletred
    ]

    print("Running event-driven simulations...")
    print()

    results = []
    for i, maker in enumerate(scenario_makers):
        es = maker()
        print(f"  Simulating: {es.name} ({int(es.t_end)} days)")
        n_events = len(es.events)
        event_kinds = [e.kind.value for e in es.events]
        print(f"    Events: {n_events} ({', '.join(event_kinds)})")

        res = run_event_scenario(p, es, t_end=es.t_end)
        final = res.y[:, -1]
        print(f"    Final: jing={final[0]:.2f}, qi={final[1]:.2f}, "
              f"shen={final[2]:.2f}, C={final[3]:.3f}, I={final[4]:.3f}")
        results.append((es.name, res, scenario_colors[i], es.t_end))

    print()
    print("Generating plots...")

    for name, res, color, t_end in results:
        safe_name = name.lower().replace(" ", "_").replace("&", "and")
        plot_single_scenario(name, res, f"event_{safe_name}.png",
                             t_end=t_end, color=color)

    plot_event_phase_portrait(
        [(name, res) for name, res, _, _ in results],
        "event_phase_portrait.png",
    )

    print("Done.")
