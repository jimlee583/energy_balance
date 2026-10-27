"""Generate the figures embedded in ``docs/Solar_Array_Pointing_Algorithm.docx``.

Every figure is drawn from the simulator itself, using the default configuration
(the opening ``LEO 1000 km, 80 deg`` preset) with the sidebar's default gimbal
limits, so the document and the app can never drift apart.

Run with::

    uv run --with matplotlib python scripts/make_pointing_figures.py [output_dir]

With no argument the figures land in ``docs/figures/``. ``scripts/make_pointing_doc.py``
calls :func:`build_figures` directly and normally renders into a temporary directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

from energy_balance.attitude import body_to_eci
from energy_balance.config import SimulationConfig, SolarArrayConfig
from energy_balance.eclipse import illumination_fraction_conical
from energy_balance.orbit import OrbitState, propagate_orbit
from energy_balance.solar_array import panel_frames_eci
from energy_balance.sun import sun_vector_eci

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

#: Keys are the names ``make_pointing_doc.py`` asks for; values are the file names.
FIGURES = {
    "frames": "pointing_frames.png",
    "gimbal": "pointing_gimbal.png",
    "trace": "pointing_trace.png",
}

NAVY = "#1F3A5F"
ACCENT = "#C1440E"
BLUE = "#2E6F9E"
GREEN = "#166534"
GREY = "#6B7280"

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#9CA3AF",
    "text.color": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "axes.titlecolor": NAVY,
}


def _default_limited_config() -> SimulationConfig:
    """The opening preset with the sidebar's default gimbal limits applied."""
    base = SimulationConfig()
    panels = [
        panel.model_copy(
            update={
                "gimbal_min_deg": -180.0,
                "gimbal_max_deg": 180.0,
                "gimbal2_min_deg": -60.0,
                "gimbal2_max_deg": 60.0,
            }
        )
        for panel in base.solar_array.panels
    ]
    return base.model_copy(update={"solar_array": SolarArrayConfig(panels=panels)})


def _arrow(ax, origin, vector, color, label=None, lw=1.7, ls="-", loff=1.16,
           ha="center", va="center", fs=8.5) -> None:
    """Draw a 2-D arrow from ``origin`` along ``vector``, optionally labelled at its tip."""
    origin = np.asarray(origin, dtype=float)
    vector = np.asarray(vector, dtype=float)
    ax.annotate(
        "",
        xy=origin + vector,
        xytext=origin,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                        shrinkA=0, shrinkB=0),
        zorder=6,
    )
    if label:
        tip = origin + vector * loff
        ax.text(tip[0], tip[1], label, color=color, fontsize=fs, ha=ha, va=va, zorder=7)


def _figure_frames(path: Path) -> None:
    """Figure 1: the orbit frame and the body frame for nadir pointing."""
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.05}
    )
    for ax in (axl, axr):
        ax.set_aspect("equal")
        ax.axis("off")

    # (a) orbit-level view: Earth, the orbit, and the R/S/W triad at the spacecraft.
    axl.add_patch(Circle((0, 0), 1.0, facecolor="#DBEAFE", edgecolor=BLUE, lw=1.1, zorder=1))
    axl.text(0, 0, "Earth", ha="center", va="center", color=NAVY, fontsize=8.5, zorder=2)
    sweep = np.linspace(0, 2 * np.pi, 400)
    r_orb = 2.0
    axl.plot(r_orb * np.cos(sweep), r_orb * np.sin(sweep), color=GREY, lw=0.9,
             ls=(0, (5, 4)), zorder=1)

    anomaly = np.deg2rad(52.0)
    pos = np.array([r_orb * np.cos(anomaly), r_orb * np.sin(anomaly)])
    r_hat = pos / np.linalg.norm(pos)
    s_hat = np.array([-r_hat[1], r_hat[0]])
    axl.plot(*pos, marker="s", ms=6, color=NAVY, zorder=8)
    axl.annotate("", xy=pos, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, ls=":"), zorder=2)

    arm = 0.95
    _arrow(axl, pos, r_hat * arm, BLUE, r"$\hat{R}$", loff=1.2)
    _arrow(axl, pos, s_hat * arm, BLUE, r"$\hat{S}$", loff=1.2)
    axl.plot(*pos, marker="o", ms=7, mfc="white", mec=BLUE, mew=1.4, zorder=9)
    axl.plot(*pos, marker=".", ms=3, color=BLUE, zorder=10)
    axl.text(*(pos - r_hat * 0.18 + s_hat * 0.52), r"$\hat{W}$ (out of page)",
             color=BLUE, fontsize=8, ha="center", va="top")

    for k in range(3):
        base = np.array([3.3, -1.25 + 0.5 * k])
        axl.annotate("", xy=base - np.array([0.85, 0.0]), xytext=base,
                     arrowprops=dict(arrowstyle="-|>", color="#B45309", lw=1.2), zorder=3)
    axl.text(3.35, 0.35, "sunlight", color="#B45309", fontsize=8.5, ha="center")
    axl.set_xlim(-2.4, 3.9)
    axl.set_ylim(-2.4, 2.6)
    axl.set_title("(a) orbit frame", fontsize=9, loc="left", color=NAVY)

    # (b) close-up: the bus, the two booms, and the body axes.
    axr.add_patch(plt.Rectangle((-0.42, -0.42), 0.84, 0.84, facecolor="#E5E7EB",
                                edgecolor="#374151", lw=1.1, zorder=4))
    for sign in (+1, -1):
        axr.plot([0.42 * sign, 1.25 * sign], [0, 0], color="#374151", lw=1.6, zorder=3)
        axr.add_patch(plt.Rectangle((1.25 * sign - (0.0 if sign > 0 else 0.62), -0.1), 0.62, 0.2,
                                    facecolor="#1D4ED8", edgecolor="#1E3A8A", lw=0.8,
                                    alpha=0.85, zorder=3))
    axr.text(1.56, -0.42, "wing (+X)", color="#1D4ED8", fontsize=7.5, ha="center", va="top")
    axr.text(-1.56, -0.42, "wing (-X)", color="#1D4ED8", fontsize=7.5, ha="center", va="top")

    _arrow(axr, (0, 0), (1.15, 0), ACCENT, lw=1.8)
    axr.text(1.15, 0.58, r"$+X_b$ = boom = $\hat{a}_1$", color=ACCENT, fontsize=8.5,
             ha="center", va="bottom")
    _arrow(axr, (0, 0), (0, -1.45), ACCENT, r"$+Z_b$ = nadir = $\hat{n}_0$ (rest normal)",
           loff=1.06, ha="center", va="top", lw=1.8)
    axr.plot(0, 0, marker="o", ms=8, mfc="white", mec=ACCENT, mew=1.5, zorder=9)
    axr.plot(0, 0, marker="x", ms=5, mew=1.4, color=ACCENT, zorder=10)
    axr.text(-0.58, 0.58, r"$+Y_b$ (into page)", color=ACCENT, fontsize=8, ha="right",
             va="bottom")
    _arrow(axr, (0, 0), (0, 1.35), GREY, lw=1.0, ls=(0, (3, 3)))
    axr.text(0.09, 1.40, "zenith", color=GREY, fontsize=7.5, ha="left", va="bottom")
    axr.set_xlim(-2.9, 2.9)
    axr.set_ylim(-2.3, 1.9)
    axr.set_title("(b) body frame and array geometry", fontsize=9, loc="left", color=NAVY)

    fig.suptitle("Figure 1   Orbit frame and body frame for the default nadir-pointing attitude",
                 fontsize=9.5, color=NAVY, x=0.012, ha="left", y=1.02)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figure_gimbal(path: Path) -> None:
    """Figure 2: the two gimbal rotations, drawn as two orthogonal views."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"wspace": 0.12})
    for ax in (axl, axr):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.5, 1.65)

    # (a) azimuth, looking down the boom axis.
    axl.add_patch(Wedge((0, 0), 1.0, -180, 180, width=0.012, facecolor="#D1D5DB", zorder=1))
    _arrow(axl, (0, 0), (0, 1.15), GREY, r"$\hat{n}_0$  ($\theta_1=0$)", loff=1.1, va="bottom")
    _arrow(axl, (0, 0), (1.15, 0), BLUE, r"$\hat{a}_2$", loff=1.12, ha="left")
    theta1 = np.deg2rad(118.0)
    swung = np.array([np.sin(theta1), np.cos(theta1)])
    _arrow(axl, (0, 0), swung * 1.15, ACCENT, r"$\hat{n}$ after $\theta_1$", loff=1.14,
           ha="right", va="top")
    arc = np.linspace(0, theta1, 80)
    axl.plot(0.46 * np.sin(arc), 0.46 * np.cos(arc), color=ACCENT, lw=1.4, zorder=5)
    axl.text(0.60 * np.sin(theta1 / 2), 0.60 * np.cos(theta1 / 2), r"$\theta_1$",
             color=ACCENT, fontsize=11)
    axl.plot(0, 0, marker="o", ms=9, mfc="white", mec=NAVY, mew=1.5, zorder=9)
    axl.plot(0, 0, marker=".", ms=4, color=NAVY, zorder=10)
    axl.text(-0.14, -0.30, r"$\hat{a}_1$ out of page", color=NAVY, fontsize=8, ha="right")
    axl.text(0, -1.38,
             r"primary axis: swings $\hat{n}$ in the $\hat{n}_0$–$\hat{a}_2$ plane",  # noqa: RUF001
             fontsize=8, ha="center", color=GREY)
    axl.set_title(r"(a) view down the boom axis $\hat{a}_1$", fontsize=9, loc="left", color=NAVY)

    # (b) elevation, in the plane containing the swung normal and the boom.
    axr.add_patch(Wedge((0, 0), 1.0, 30, 150, width=0.012, facecolor="#D1D5DB", zorder=1))
    axr.add_patch(Wedge((0, 0), 0.99, 30, 150, facecolor=GREEN, alpha=0.07, zorder=0))
    _arrow(axr, (0, 0), (0, 1.15), GREY, r"$\hat{n}$ after $\theta_1$  ($\theta_2=0$)",
           loff=1.08, va="bottom")
    _arrow(axr, (0, 0), (-1.2, 0), NAVY, r"$-\hat{a}_1$", loff=1.1, ha="right")
    _arrow(axr, (0, 0), (1.2, 0), NAVY, r"$+\hat{a}_1$", loff=1.1, ha="left")
    theta2 = np.deg2rad(46.0)
    tipped = np.array([-np.sin(theta2), np.cos(theta2)])
    _arrow(axr, (0, 0), tipped * 1.2, ACCENT, r"$\hat{n}=\hat{u}_{sun}$", loff=1.1,
           ha="right", va="bottom")
    arc = np.linspace(0, theta2, 80)
    axr.plot(-0.5 * np.sin(arc), 0.5 * np.cos(arc), color=ACCENT, lw=1.4, zorder=5)
    axr.text(-0.40 * np.sin(theta2 / 2) - 0.14, 0.64 * np.cos(theta2 / 2), r"$\theta_2$",
             color=ACCENT, fontsize=11)
    for sign, label in ((+1, r"$-60^\circ$"), (-1, r"$+60^\circ$")):
        stop = np.array([sign * np.sin(np.deg2rad(60.0)), np.cos(np.deg2rad(60.0))])
        axr.plot([0, stop[0]], [0, stop[1]], color=GREEN, lw=1.1, ls=(0, (3, 2)), zorder=4)
        axr.text(stop[0] * 1.14, stop[1] * 1.14, label, color=GREEN, fontsize=8,
                 ha="left" if sign > 0 else "right")
    axr.text(0, -0.62, "shaded: legal travel for\n" r"the default $\pm 60^\circ$ stop",
             fontsize=8, ha="center", color=GREEN)
    axr.text(0, -1.38, r"secondary axis: tilts $\hat{n}$ toward the boom",
             fontsize=8, ha="center", color=GREY)
    axr.set_title(r"(b) view along the secondary axis $\hat{a}_2$", fontsize=9, loc="left",
                  color=NAVY)

    fig.suptitle("Figure 2   Two-axis gimbal kinematics: azimuth "
                 r"$\theta_1$ about $\hat{a}_1$, then elevation $\theta_2$ about $\hat{a}_2$",
                 fontsize=9.5, color=NAVY, x=0.012, ha="left", y=1.02)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figure_trace(path: Path) -> None:
    """Figure 3: commanded vs actual gimbal angles and incidence over two orbits."""
    cfg = _default_limited_config()
    state = OrbitState(cfg.orbit, cfg.mission.epoch)
    times_s = np.arange(0.0, 2.0 * state.period_s + 1e-9, cfg.mission.time_step_s)
    orbit = propagate_orbit(state, times_s)
    sun = sun_vector_eci(cfg.mission.epoch, times_s)
    illum = illumination_fraction_conical(orbit["r_eci_km"], sun["r_sun_eci_km"])
    rotation = body_to_eci(cfg.attitude, orbit["r_eci_km"], orbit["v_eci_km_s"],
                           sun["u_sun_eci"])

    panel = cfg.solar_array.panels[0]
    limited = panel_frames_eci(panel, rotation, sun["u_sun_eci"], times_s)
    unlimited = panel_frames_eci(
        panel.model_copy(
            update={"gimbal_min_deg": None, "gimbal_max_deg": None,
                    "gimbal2_min_deg": None, "gimbal2_max_deg": None}
        ),
        rotation,
        sun["u_sun_eci"],
        times_s,
    )
    minutes = times_s / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.9], "hspace": 0.16})

    def shade_eclipse(ax) -> None:
        eclipsed = illum < 0.01
        edges = np.flatnonzero(np.diff(np.r_[0, eclipsed.astype(int), 0]))
        for start, end in edges.reshape(-1, 2):
            ax.axvspan(minutes[start], minutes[min(end, len(minutes) - 1)],
                       color="#111827", alpha=0.07, lw=0)

    ax = axes[0]
    shade_eclipse(ax)
    ax.plot(minutes, unlimited["angle1_deg"], color=GREY, lw=1.6, ls=(0, (4, 3)),
            label="ideal Sun-tracking command")
    ax.plot(minutes, limited["angle1_deg"], color=NAVY, lw=1.5,
            label="actual (limited + rate-capped)")
    for limit in (-180, 180):
        ax.axhline(limit, color=ACCENT, lw=1.0, ls=(0, (2, 2)))
    ax.text(minutes[-1] * 0.995, 186, "travel limit", color=ACCENT, fontsize=7.5,
            va="bottom", ha="right")
    ax.set_ylabel(r"azimuth $\theta_1$ (deg)")
    ax.set_ylim(-235, 300)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncols=2, handlelength=2.2,
              columnspacing=1.4)

    ax = axes[1]
    shade_eclipse(ax)
    ax.plot(minutes, unlimited["angle2_deg"], color=GREY, lw=1.6, ls=(0, (4, 3)))
    ax.plot(minutes, limited["angle2_deg"], color=NAVY, lw=1.5)
    for stop in (-60, 60):
        ax.axhline(stop, color=ACCENT, lw=1.0, ls=(0, (2, 2)))
    ax.text(minutes[-1] * 0.995, 64, "hard stop", color=ACCENT, fontsize=7.5,
            va="bottom", ha="right")
    ax.set_ylabel(r"elevation $\theta_2$ (deg)")
    ax.set_ylim(-108, 118)
    ax.set_yticks([-90, -60, 0, 60, 90])

    ax = axes[2]
    shade_eclipse(ax)
    ax.plot(minutes, unlimited["cos_incidence"], color=GREY, lw=1.6, ls=(0, (4, 3)))
    ax.plot(minutes, limited["cos_incidence"], color=GREEN, lw=1.6)
    ax.set_ylabel("cos(incidence)")
    ax.set_ylim(-0.04, 1.12)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("time from epoch (min)         shaded bands = eclipse")
    ax.set_xlim(0, minutes[-1])
    ax.annotate("start-up slew from\nthe rest position", xy=(1.6, 0.08), xytext=(9, 0.30),
                fontsize=7.5, color=GREY, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))
    ax.annotate("elevation pinned at the stop", xy=(79.5, 0.855), xytext=(96, 0.42),
                fontsize=7.5, color=GREEN, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.9))

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)

    fig.suptitle("Figure 3   Default configuration: commanded vs actual gimbal angles and "
                 "incidence over two orbits", fontsize=9.5, color=NAVY, x=0.012, ha="left",
                 y=0.985)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures(out_dir: Path) -> dict[str, Path]:
    """Render every figure into ``out_dir``; return the key -> path mapping."""
    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: out_dir / name for key, name in FIGURES.items()}
    with plt.rc_context(_RC):
        _figure_frames(paths["frames"])
        _figure_gimbal(paths["gimbal"])
        _figure_trace(paths["trace"])
    return paths


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    for path in build_figures(out_dir).values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
