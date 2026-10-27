# Satellite Energy Balance

An engineering-level energy balance simulator for satellites, with a Streamlit UI.

Given an orbit, an attitude mode, a set of solar panels, a load schedule, and a battery, the
app simulates the satellite over many orbits and reports:

- orbit facts: period, beta angle, RAAN drift, eclipse duration/fraction
- time series of generated power, load power, net power, and battery state of charge
- battery-dominated bus voltage and current, including internal-resistance sag
- per-orbit energy totals and a worst-orbit summary
- a pass/fail verdict against the depth-of-discharge limit and power budget
- sizing hints for the minimum solar array area and battery capacity
- an optional seasonal (year-long) sweep of beta angle and eclipse fraction
- optional gimbal travel limits (cable-wrap rewind or hard mechanical stops) for 1-axis
  and 2-axis solar-array gimbals

## Install and run

```bash
uv sync --all-extras
uv run streamlit run app.py
```

## Run tests

```bash
uv run pytest
```

## Project layout

```
src/energy_balance/
    config.py        # pydantic input models + presets + JSON save/load
    orbit.py         # Keplerian elements, J2 secular drift, SSO helper
    sun.py           # low-precision solar ephemeris
    eclipse.py       # conical umbra/penumbra illumination fraction
    attitude.py      # body-to-ECI rotations for sun/nadir/inertial pointing
    solar_array.py   # per-panel power model with tracking options
    loads.py         # load schedule evaluator
    battery.py       # SOC integrator with efficiencies and shunting
    simulation.py    # top-level run_simulation(config) -> DataFrame
    metrics.py       # per-orbit summaries, verdict, sizing hints
app.py               # Streamlit UI
tests/               # validation tests
```

## Notes

The physics is deliberately kept in a plain Python package with no Streamlit imports, so it
can be reused from notebooks, scripts, or a CLI later.

## Power regulation and bus architecture

Solar regulation and bus architecture are modeled as separate choices:

- **MPPT / Direct Energy Transfer (DET)** selects the solar-array power path.
- **Regulated / Battery-dominated** selects how the bus voltage behaves.

The default battery-dominated configuration represents a typical unregulated 8-series
lithium-ion pack: 24.0 V empty open-circuit voltage, 33.6 V full open-circuit voltage,
0.08 ohm equivalent pack resistance, and a 24.0 V minimum usable bus voltage. The simulator
interpolates open-circuit voltage with state of charge, calculates terminal voltage and
current under load, and reports demand that cannot be served without crossing the minimum
voltage as unmet load.

This is a pack-level energy and voltage model. In DET mode, the current version does not
solve a solar-array I-V curve at the battery-clamped operating voltage.

## Gimbal travel limits

Both 1-axis and 2-axis arrays support a limit on the **primary** gimbal axis (the one
driven by `rotation_axis_body`): enable **Limit gimbal travel** / **Limit primary axis
travel** in the Solar arrays sidebar and enter the minimum angle, maximum angle, and
maximum slew rate. Limits must contain the 0° rest position.

- A span of **360° or more** is a cable-wrap limit: the drive stays on its current
  continuous Sun-tracking branch until it reaches a limit, then slews in the reverse
  direction toward the equivalent angle 360° away. Generated power uses the array's
  actual orientation during that rewind.
- A span **under 360°** is a hard mechanical stop: the drive simply clamps at the limit,
  with no rewind. If the Sun moves out of reach, incidence (and generated power) degrades
  smoothly until the Sun comes back within range.

2-axis arrays additionally support an independent **secondary axis** limit (**Limit
secondary axis travel**), bounded to ±90°, always a hard stop (a 2-axis gimbal's
secondary/tip angle can never exceed that range). The two axes clamp independently, so a
2-axis panel can have, for example, ±80° on the primary axis and ±70° on the secondary
axis at the same time.

## Frames tab views

The Frames tab animates one orbit across three stacked 3D scenes:

1. **Earth-centered inertial** — the orbit seen nearly edge-on, with the Sun above or below
   the orbit plane according to beta angle. Earth and the Sun are drawn not to scale, and
   the orbit radius is compressed so the spacecraft stays visible.
2. **Satellite close-up** — the satellite filling the frame, either body-fixed (the
   satellite is stationary and the Sun arrow sweeps) or velocity-fixed.
3. **Sun viewpoint** — an orthographic camera parked at the Sun, looking back along the Sun
   line at the Earth. This is what the Sun sees: a panel square to the Sun shows its full
   rectangle, a panel edge-on to the Sun collapses to a line. Unlike the first scene, the
   Earth and the orbit are drawn at the same scale, and the camera is orthographic because
   the real Sun's rays are parallel across the orbit. The Earth's silhouette is therefore
   the actual eclipse boundary: the satellite slips behind the disk exactly when the
   simulation puts it in eclipse (`test_sun_view_silhouette_matches_simulated_eclipse`
   checks this against the simulated illumination).

All three scenes share the Play/Pause control, the time slider, and the display checkboxes.

The Frames tab plots the legal Sun-tracking target against the actual gimbal angle(s) for
every limited axis and marks rewind intervals (primary axis, cable-wrap mode only). The
model does not currently optimize rewind timing around eclipse; a rewind begins when
continued tracking would cross a configured limit. For a limited 2-axis panel, the
"Inspect frame" table reports both axis angles and whether either is pinned at its limit.
