# Satellite Energy Balance

An engineering-level energy balance simulator for satellites, with a Streamlit UI.

Given an orbit, an attitude mode, a set of solar panels, a load schedule, and a battery, the
app simulates the satellite over many orbits and reports:

- orbit facts: period, beta angle, RAAN drift, eclipse duration/fraction
- time series of generated power, load power, net power, and battery state of charge
- per-orbit energy totals and a worst-orbit summary
- a pass/fail verdict against the depth-of-discharge limit and power budget
- sizing hints for the minimum solar array area and battery capacity
- an optional seasonal (year-long) sweep of beta angle and eclipse fraction

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
