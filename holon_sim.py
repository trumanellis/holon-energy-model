"""
Conservation Laws of a Creative Holon -- Computational Model
=============================================================

Simulates the five coupled ODEs (phi_jing, phi_qi, phi_shen, C, I)
under six practitioner scenarios and compares trajectories.

The system:
  dphi_jing/dt = Gamma_eff - (kappa1_eff + lam_jing)*phi_jing + mu1*phi_qi - D_jing
  dphi_qi/dt   = kappa1_eff*phi_jing - (mu1 + kappa2_eff + lam_qi)*phi_qi + mu2*phi_shen
                 + R_qi - D_qi - sigma*max(0, phi_jing - phi_jing_eq)*phi_qi
  dphi_shen/dt = kappa2_eff*phi_qi - (mu2 + lam_shen)*phi_shen
  dC/dt        = h*H*E*(1-C) - d_ext*S*(1-E)*C - d_nov*N*C + rho*Rc*(1-C)
  dI/dt        = eps*G*(1-I) - omega*S*I

With nonlinear coupling through:
  Gamma_eff  = Gamma0 * (p_min + (1-p_min)*phi_qi/(phi_qi + Kp))
  kappa1_eff = kappa10 * C^a1 * I^b1 * (f_min + (1-f_min)*phi_qi/(phi_qi + Kf))
  kappa2_eff = kappa20 * C^a2 * I^b2 * (g_min + (1-g_min)*phi_qi/(phi_qi + Kg))

Congestion reference uses processing-capacity equilibrium:
  phi_jing_eq = Gamma_eff / (kappa1_eff + lam_jing)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ── Parameters ────────────────────────────────────────────────────────────

@dataclass
class Params:
    """All model parameters with physically motivated defaults."""

    # Endogenous generation
    Gamma0: float = 1.0        # baseline jing generation rate
    p_min: float = 0.05        # floor on generation modulation
    Kp: float = 0.5            # half-saturation for generation modulation

    # Dissipation rates (lambda): shen > qi > jing
    lam_jing: float = 0.02     # jing dissipation (very slow -- 50-day e-fold)
    lam_qi: float = 0.08       # qi dissipation (medium -- 12-day e-fold)
    lam_shen: float = 0.40     # shen dissipation (fast -- 2.5-day e-fold)

    # Transmutation baselines: kappa2 < kappa1
    kappa10: float = 0.40      # baseline jing->qi rate
    kappa20: float = 0.10      # baseline qi->shen rate

    # Re-infusion (downward, passive, infrastructure-independent)
    mu1: float = 0.05          # qi->jing re-infusion
    mu2: float = 0.08          # shen->qi re-infusion

    # Superlinearity exponents: second transition more sensitive
    a1: float = 1.5            # kappa1 coherence exponent
    a2: float = 2.0            # kappa2 coherence exponent
    b1: float = 1.3            # kappa1 imagination exponent
    b2: float = 2.0            # kappa2 imagination exponent

    # Catalytic floor and half-saturation
    f_min: float = 0.03        # kappa1 catalytic floor
    g_min: float = 0.02        # kappa2 catalytic floor
    Kf: float = 0.5            # kappa1 catalytic half-saturation
    Kg: float = 0.8            # kappa2 catalytic half-saturation

    # Congestion coupling
    sigma: float = 0.08        # congestion-stress coefficient

    # Infrastructure dynamics
    h: float = 0.15            # coherence building rate
    d_ext: float = 0.20        # external stimulation damage rate
    d_nov: float = 0.25        # novelty escalation damage rate
    rho: float = 0.10          # reciprocity coherence building
    eps: float = 0.10          # imagination building rate
    omega: float = 0.08        # imagination atrophy rate from stimulation
    delta_I: float = 0.02      # imagination natural decay from disuse


@dataclass
class Scenario:
    """Practitioner-controlled driving variables."""
    name: str = ""
    H: float = 0.0             # heart engagement
    E: float = 0.5             # embodiment
    S: float = 0.0             # external stimulation
    N: float = 0.0             # novelty escalation
    G: float = 0.5             # generative fraction
    Rc: float = 0.0            # reciprocity
    D_jing: float = 0.0        # ejaculatory discharge rate
    R_qi: float = 0.1          # environmental vitality support (baseline)
    D_qi: float = 0.0          # stress/overstimulation qi drain

    # Initial conditions
    phi_jing_0: float = 2.0
    phi_qi_0: float = 3.0
    phi_shen_0: float = 0.5
    C_0: float = 0.3
    I_0: float = 0.5


# ── ODE System ────────────────────────────────────────────────────────────

def holon_ode(t, y, p: Params, s: Scenario):
    """Right-hand side of the five coupled ODEs."""

    phi_jing, phi_qi, phi_shen, C, I = y

    # Enforce non-negativity (the integrator can overshoot)
    phi_jing = max(phi_jing, 0.0)
    phi_qi   = max(phi_qi, 0.0)
    phi_shen = max(phi_shen, 0.0)
    C = np.clip(C, 0.0, 1.0)
    I = np.clip(I, 0.0, 1.0)

    # ── Effective generation ──
    p_mod = p.p_min + (1.0 - p.p_min) * phi_qi / (phi_qi + p.Kp)
    Gamma_eff = p.Gamma0 * p_mod

    # ── Catalytic functions ──
    f_qi = p.f_min + (1.0 - p.f_min) * phi_qi / (phi_qi + p.Kf)
    g_qi = p.g_min + (1.0 - p.g_min) * phi_qi / (phi_qi + p.Kg)

    # ── Effective transmutation coefficients ──
    C_safe = max(C, 1e-12)
    I_safe = max(I, 1e-12)

    kappa1_eff = p.kappa10 * (C_safe ** p.a1) * (I_safe ** p.b1) * f_qi
    kappa2_eff = p.kappa20 * (C_safe ** p.a2) * (I_safe ** p.b2) * g_qi

    # ── Processing-capacity equilibrium for congestion reference ──
    # Uses current kappa1_eff: congestion activates when phi_jing exceeds
    # what the system can currently process, not the theoretical maximum.
    phi_jing_eq = Gamma_eff / (kappa1_eff + p.lam_jing)

    # ── Congestion stress ──
    congestion = p.sigma * max(0.0, phi_jing - phi_jing_eq) * phi_qi

    # ── Discharge can't exceed available reserves ──
    eff_D_jing = min(s.D_jing, phi_jing * 10.0)  # soft cap: can't drain below zero
    eff_D_qi   = min(s.D_qi, phi_qi * 10.0)

    # ── Phase equations ──
    d_jing = (Gamma_eff
              - (kappa1_eff + p.lam_jing) * phi_jing
              + p.mu1 * phi_qi
              - eff_D_jing)

    d_qi = (kappa1_eff * phi_jing
            - (p.mu1 + kappa2_eff + p.lam_qi) * phi_qi
            + p.mu2 * phi_shen
            + s.R_qi - eff_D_qi
            - congestion)

    d_shen = (kappa2_eff * phi_qi
              - (p.mu2 + p.lam_shen) * phi_shen)

    # ── Infrastructure equations ──
    dC = (p.h * s.H * s.E * (1.0 - C)
          - p.d_ext * s.S * (1.0 - s.E) * C
          - p.d_nov * s.N * C
          + p.rho * s.Rc * (1.0 - C))

    dI = (p.eps * s.G * (1.0 - I)
          - p.omega * s.S * I
          - p.delta_I * (1.0 - s.G) * I)  # disuse atrophy: decays when not generating

    # ── Non-negativity enforcement: if at zero, derivative can't be negative ──
    if y[0] <= 0 and d_jing < 0:
        d_jing = 0.0
    if y[1] <= 0 and d_qi < 0:
        d_qi = 0.0
    if y[2] <= 0 and d_shen < 0:
        d_shen = 0.0

    return [d_jing, d_qi, d_shen, dC, dI]


# ── Simulation Runner ─────────────────────────────────────────────────────

def run_scenario(p: Params, s: Scenario, t_span=(0, 108), n_points=2000):
    """Integrate the ODE system for a given scenario."""
    y0 = [s.phi_jing_0, s.phi_qi_0, s.phi_shen_0, s.C_0, s.I_0]
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        holon_ode, t_span, y0,
        args=(p, s),
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-8, atol=1e-10,
        max_step=0.5,
    )

    return sol


# ── Define Scenarios ──────────────────────────────────────────────────────

def make_scenarios():
    """The six scenarios from the paper."""

    # Shared depleted initial conditions (representing a chronic baseline)
    base_ic = dict(phi_jing_0=2.0, phi_qi_0=2.5, phi_shen_0=0.3, C_0=0.30, I_0=0.45)

    scenarios = []

    # 1. Pure retention, no cultivation
    #    No discharge, no stimulation, but also no active practice.
    #    G is low because the person isn't actively exercising imagination.
    scenarios.append(Scenario(
        name="Retention Only",
        H=0.05, E=0.45, S=0.0, N=0.0, G=0.15, Rc=0.0,
        D_jing=0.0, R_qi=0.10, D_qi=0.0,
        **base_ic
    ))

    # 2. Retention + physical cultivation
    #    Active body practice but limited heart/imagination work.
    scenarios.append(Scenario(
        name="Retention + Physical",
        H=0.25, E=0.80, S=0.0, N=0.0, G=0.35, Rc=0.0,
        D_jing=0.0, R_qi=0.30, D_qi=0.0,
        **base_ic
    ))

    # 3. Retention + full cultivation (physical + contemplative + heart)
    scenarios.append(Scenario(
        name="Retention + Full Cultivation",
        H=0.80, E=0.90, S=0.0, N=0.0, G=0.85, Rc=0.0,
        D_jing=0.0, R_qi=0.30, D_qi=0.0,
        **base_ic
    ))

    # 4. Habitual ejaculation, no porn
    #    Moderate frequency ejaculation, no external input, minimal practice.
    scenarios.append(Scenario(
        name="Habitual Ejaculation",
        H=0.10, E=0.45, S=0.0, N=0.0, G=0.10, Rc=0.0,
        D_jing=0.40, R_qi=0.10, D_qi=0.0,
        **base_ic
    ))

    # 5. Habitual ejaculation + porn
    #    High external stimulation, frequent discharge, escalation.
    scenarios.append(Scenario(
        name="Ejaculation + Porn",
        H=0.0, E=0.10, S=1.5, N=0.4, G=0.0, Rc=0.0,
        D_jing=0.60, R_qi=0.08, D_qi=0.12,
        **base_ic
    ))

    # 6. Retention + porn (no ejaculation, continued viewing)
    #    The worst-case scenario per the model.
    scenarios.append(Scenario(
        name="Retention + Porn",
        H=0.0, E=0.10, S=1.5, N=0.4, G=0.0, Rc=0.0,
        D_jing=0.0, R_qi=0.08, D_qi=0.12,
        **base_ic
    ))

    return scenarios


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_results(results, filename):
    """Generate a comprehensive comparison plot."""

    colors = {
        "Retention Only":              "#B8860B",
        "Retention + Physical":        "#2E8B57",
        "Retention + Full Cultivation": "#4169E1",
        "Habitual Ejaculation":        "#808080",
        "Ejaculation + Porn":          "#8B0000",
        "Retention + Porn":            "#C71585",
    }

    fig, axes = plt.subplots(4, 2, figsize=(16, 26),
                              gridspec_kw={"height_ratios": [1, 1, 1, 0.7]})
    fig.patch.set_facecolor("#0a0a0f")

    var_names = [
        (r"$\varphi_{jing}$ -- Generative Essence", 0),
        (r"$\varphi_{qi}$ -- Vitality", 1),
        (r"$\varphi_{shen}$ -- Awareness", 2),
        ("C -- Circuit Coherence", 3),
        ("I -- Imaginative Capacity", 4),
    ]

    for idx, (title, var_idx) in enumerate(var_names):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor("#0d0d14")

        for name, sol in results:
            c = colors.get(name, "#888888")
            t_days = sol.t
            ax.plot(t_days, sol.y[var_idx], color=c, linewidth=1.8,
                    label=name, alpha=0.9)

        ax.set_title(title, color="#c8c0b8", fontsize=13, pad=10)
        ax.set_xlabel("Days", color="#887e70", fontsize=10)
        ax.tick_params(colors="#665e52", labelsize=9)
        ax.spines["bottom"].set_color("#2a2a35")
        ax.spines["left"].set_color("#2a2a35")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.1, color="#444455")

    # Total Phi in the sixth panel
    ax = axes[2, 1]
    ax.set_facecolor("#0d0d14")
    for name, sol in results:
        c = colors.get(name, "#888888")
        total_phi = sol.y[0] + sol.y[1] + sol.y[2]
        ax.plot(sol.t, total_phi, color=c, linewidth=1.8, label=name, alpha=0.9)
    ax.set_title(r"$\Phi_{total}$ -- Total Creative Energy", color="#c8c0b8",
                 fontsize=13, pad=10)
    ax.set_xlabel("Days", color="#887e70", fontsize=10)
    ax.tick_params(colors="#665e52", labelsize=9)
    ax.spines["bottom"].set_color("#2a2a35")
    ax.spines["left"].set_color("#2a2a35")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.1, color="#444455")

    # Legend is now in the description panel below

    fig.suptitle("Holon Energy Model -- Scenario Comparison (108 Days)",
                 color="#e8ddd0", fontsize=16, y=0.995, fontweight="light")

    # ── Scenario descriptions legend in bottom row ──
    for ax in [axes[3, 0], axes[3, 1]]:
        ax.set_facecolor("#0a0a0f")
        ax.axis("off")

    # Merge bottom row into one axis for the legend
    ax_legend = fig.add_axes([0.05, 0.01, 0.90, 0.14])
    ax_legend.set_facecolor("#0d0d14")
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")

    # Add border
    for spine in ax_legend.spines.values():
        spine.set_visible(True)
        spine.set_color("#2a2a35")

    descriptions = [
        ("Retention Only",
         "#B8860B",
         "No ejaculation. No practice. No external stimulation.\n"
         "Just willpower and waiting."),
        ("Retention + Physical",
         "#2E8B57",
         "No ejaculation. Active physical practice (gym, yoga,\n"
         "martial arts, breathwork). Limited heart/imagination work."),
        ("Retention + Full Cultivation",
         "#4169E1",
         "No ejaculation. Physical practice + meditation +\n"
         "visualization + devotional/heart practice."),
        ("Habitual Ejaculation",
         "#808080",
         "Regular ejaculation at moderate frequency.\n"
         "No external stimulation. No practice. The baseline."),
        ("Ejaculation + Porn",
         "#8B0000",
         "Frequent ejaculation with porn. Novelty escalation.\n"
         "No embodiment, no heart, no self-generated content."),
        ("Retention + Porn",
         "#C71585",
         "No ejaculation but continued porn viewing.\n"
         "Retaining the fluid while destroying the system."),
    ]

    x_positions = [0.02, 0.35, 0.68]
    y_positions = [0.55, 0.05]

    for i, (name, color, desc) in enumerate(descriptions):
        col = i % 3
        row = i // 3
        x = x_positions[col]
        y = y_positions[row]

        # Color swatch
        ax_legend.add_patch(plt.Rectangle((x, y + 0.28), 0.025, 0.12,
                                           facecolor=color, edgecolor="none",
                                           transform=ax_legend.transAxes))
        # Scenario name
        ax_legend.text(x + 0.035, y + 0.30, name,
                        color=color, fontsize=10, fontweight="bold",
                        transform=ax_legend.transAxes, va="bottom")
        # Description
        ax_legend.text(x + 0.035, y + 0.25, desc,
                        color="#887e70", fontsize=8.5,
                        transform=ax_legend.transAxes, va="top",
                        linespacing=1.4)

    plt.tight_layout(rect=[0, 0.16, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0f", edgecolor="none")
    plt.close()
    print(f"Saved: {filename}")


def plot_phase_portrait(results, filename):
    """C-I phase portrait showing threshold curves."""

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0d0d14")

    p = Params()

    # Draw threshold curves at reference phi_qi
    phi_qi_ref = 2.5
    f_ref = p.f_min + (1.0 - p.f_min) * phi_qi_ref / (phi_qi_ref + p.Kf)
    g_ref = p.g_min + (1.0 - p.g_min) * phi_qi_ref / (phi_qi_ref + p.Kg)

    # Approximate threshold values (from steady-state at reasonable params)
    kappa1_thresh = 0.08
    kappa2_thresh = 0.04

    C_range = np.linspace(0.01, 1.0, 200)

    # kappa1 threshold: kappa10 * C^a1 * I^b1 * f = kappa1_thresh
    # => I = (kappa1_thresh / (kappa10 * C^a1 * f))^(1/b1)
    I_thresh1 = (kappa1_thresh / (p.kappa10 * C_range**p.a1 * f_ref)) ** (1.0/p.b1)
    I_thresh1 = np.clip(I_thresh1, 0, 1.0)

    I_thresh2 = (kappa2_thresh / (p.kappa20 * C_range**p.a2 * g_ref)) ** (1.0/p.b2)
    I_thresh2 = np.clip(I_thresh2, 0, 1.0)

    ax.fill_between(C_range, I_thresh2, 1.0,
                     where=(I_thresh2 < 1.0),
                     alpha=0.08, color="#4169E1", label="Full Circulation Regime")
    ax.fill_between(C_range, I_thresh1, I_thresh2,
                     where=(I_thresh1 < I_thresh2),
                     alpha=0.06, color="#2E8B57", label="Vitality-Only Regime")
    ax.fill_between(C_range, 0, I_thresh1,
                     where=(I_thresh1 > 0),
                     alpha=0.05, color="#8B0000", label="Dissipative Regime")

    ax.plot(C_range, I_thresh1, color="#2E8B57", linewidth=1.5,
            linestyle="--", alpha=0.6, label=r"$\kappa_1$ threshold")
    ax.plot(C_range, I_thresh2, color="#4169E1", linewidth=1.5,
            linestyle="--", alpha=0.6, label=r"$\kappa_2$ threshold")

    # Plot trajectories
    colors = {
        "Retention Only":              "#B8860B",
        "Retention + Physical":        "#2E8B57",
        "Retention + Full Cultivation": "#4169E1",
        "Habitual Ejaculation":        "#808080",
        "Ejaculation + Porn":          "#8B0000",
        "Retention + Porn":            "#C71585",
    }

    for name, sol in results:
        c = colors.get(name, "#888888")
        C_traj = sol.y[3]
        I_traj = sol.y[4]
        ax.plot(C_traj, I_traj, color=c, linewidth=2.0, alpha=0.8)
        # Start marker
        ax.plot(C_traj[0], I_traj[0], 'o', color=c, markersize=6, alpha=0.8)
        # End marker
        ax.plot(C_traj[-1], I_traj[-1], 's', color=c, markersize=8, alpha=0.9)
        # Label at endpoint
        ax.annotate(name, (C_traj[-1], I_traj[-1]),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=8, color=c, alpha=0.9)

    ax.set_xlabel("C -- Circuit Coherence", color="#887e70", fontsize=12)
    ax.set_ylabel("I -- Imaginative Capacity", color="#887e70", fontsize=12)
    ax.set_title("Phase Portrait: C-I Plane with Threshold Curves",
                  color="#e8ddd0", fontsize=14, pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#665e52", labelsize=10)
    ax.spines["bottom"].set_color("#2a2a35")
    ax.spines["left"].set_color("#2a2a35")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.08, color="#444455")

    legend = ax.legend(fontsize=9, loc="upper left", frameon=False, labelcolor="#c8c0b8")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0f", edgecolor="none")
    plt.close()
    print(f"Saved: {filename}")


def plot_kappa_surface(filename):
    """Visualize kappa_eff as a function of C and I."""

    p = Params()
    phi_qi_ref = 2.5
    f_ref = p.f_min + (1.0 - p.f_min) * phi_qi_ref / (phi_qi_ref + p.Kf)
    g_ref = p.g_min + (1.0 - p.g_min) * phi_qi_ref / (phi_qi_ref + p.Kg)

    C_range = np.linspace(0.01, 1.0, 100)
    I_range = np.linspace(0.01, 1.0, 100)
    CC, II = np.meshgrid(C_range, I_range)

    K1 = p.kappa10 * CC**p.a1 * II**p.b1 * f_ref
    K2 = p.kappa20 * CC**p.a2 * II**p.b2 * g_ref

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0a0a0f")

    for ax, K, title, cmap_name in [
        (ax1, K1, r"$\kappa_1^{eff}$ -- Jing $\rightarrow$ Qi", "YlOrBr"),
        (ax2, K2, r"$\kappa_2^{eff}$ -- Qi $\rightarrow$ Shen", "PuBu"),
    ]:
        ax.set_facecolor("#0d0d14")
        im = ax.contourf(CC, II, K, levels=20, cmap=cmap_name, alpha=0.8)
        ax.contour(CC, II, K, levels=[0.04, 0.08], colors=["#ffffff"],
                    linewidths=1.0, linestyles="--", alpha=0.5)
        ax.set_xlabel("C -- Circuit Coherence", color="#887e70", fontsize=11)
        ax.set_ylabel("I -- Imaginative Capacity", color="#887e70", fontsize=11)
        ax.set_title(title, color="#e8ddd0", fontsize=13, pad=10)
        ax.tick_params(colors="#665e52", labelsize=9)
        ax.spines["bottom"].set_color("#2a2a35")
        ax.spines["left"].set_color("#2a2a35")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors="#665e52", labelsize=8)

    fig.suptitle("Transmutation Coefficient Surfaces (at reference " +
                 r"$\varphi_{qi}$" + " = 2.5)",
                 color="#e8ddd0", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0f", edgecolor="none")
    plt.close()
    print(f"Saved: {filename}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = Params()
    scenarios = make_scenarios()

    print("Running simulations...")
    print(f"Parameters: Gamma0={p.Gamma0}, lam_jing={p.lam_jing}, "
          f"lam_qi={p.lam_qi}, lam_shen={p.lam_shen}")
    print(f"phi_jing_eq (natural) = Gamma0/lam_jing = {p.Gamma0/p.lam_jing:.1f}")
    print()

    results = []
    for s in scenarios:
        print(f"  Simulating: {s.name}")
        sol = run_scenario(p, s, t_span=(0, 108))
        if sol.success:
            final = sol.y[:, -1]
            print(f"    Final state: jing={final[0]:.2f}, qi={final[1]:.2f}, "
                  f"shen={final[2]:.2f}, C={final[3]:.3f}, I={final[4]:.3f}")
        else:
            print(f"    FAILED: {sol.message}")
        results.append((s.name, sol))

    print()
    print("Generating plots...")

    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(results, os.path.join(base_dir, "scenario_comparison.png"))
    plot_phase_portrait(results, os.path.join(base_dir, "phase_portrait.png"))
    plot_kappa_surface(os.path.join(base_dir, "kappa_surfaces.png"))

    print("Done.")
