"""Tests for the Streamlit app's symmetric solar-wing configuration."""

from __future__ import annotations

import pytest

from app import _default_wing_side_axis, _symmetric_wing_panels
from energy_balance import PanelConfig, PanelMounting


@pytest.mark.parametrize(
    ("side_axis", "axis", "names"),
    [
        ("±Y", (0.0, 1.0, 0.0), ["Wing (+Y)", "Wing (-Y)"]),
        ("±X", (1.0, 0.0, 0.0), ["Wing (+X)", "Wing (-X)"]),
    ],
)
def test_symmetric_wings_share_per_wing_parameters(side_axis, axis, names):
    shared = PanelConfig(
        name="ignored",
        area_m2=3.25,
        normal_face="+Z",
        mounting=PanelMounting.TWO_AXIS,
        cell_efficiency=0.28,
        packing_factor=0.9,
        inherent_degradation=0.8,
        degradation_per_year=0.03,
        mission_year=4.0,
        temperature_loss=0.12,
    )

    wings = _symmetric_wing_panels(shared, side_axis)

    assert [wing.name for wing in wings] == names
    assert [wing.area_m2 for wing in wings] == [3.25, 3.25]
    assert [wing.rotation_axis_body for wing in wings] == [axis, axis]

    first = wings[0].model_dump(exclude={"name"})
    second = wings[1].model_dump(exclude={"name"})
    assert first == second


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ((1.0, 0.0, 0.0), "±X"),
        ((-0.8, 0.2, 0.0), "±X"),
        ((0.0, -1.0, 0.0), "±Y"),
        ((0.0, 0.0, 1.0), "±Y"),
    ],
)
def test_default_wing_side_axis_uses_stored_panel_axis(axis, expected):
    panel = PanelConfig(rotation_axis_body=axis)

    assert _default_wing_side_axis(panel) == expected
