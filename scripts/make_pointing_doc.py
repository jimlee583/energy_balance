"""Generate ``docs/Solar_Array_Pointing_Algorithm.docx``.

The document explains how the simulator aims each solar array: the Sun vector, the
reference frames, the gimbal kinematics, and the user-configurable travel limits.
Every worked number in it is computed from the package using the default
configuration, so regenerating the document re-derives them.

Run with::

    uv run --with python-docx --with matplotlib python scripts/make_pointing_doc.py

Figures are rendered into a temporary directory by ``make_pointing_figures.py``.
Pass a directory to keep them instead::

    uv run --with python-docx --with matplotlib python scripts/make_pointing_doc.py docs/figures
"""

# The body of this script is document copy: degree signs, true minus signs, en dashes and
# curly quotes are deliberate typography, not lookalikes for ASCII operators.
# ruff: noqa: RUF001

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from make_pointing_figures import build_figures

OUT_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "Solar_Array_Pointing_Algorithm.docx"
)

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x5F, 0x6D)


# --- helpers ---------------------------------------------------------------------------


def _add_heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = NAVY
    h.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True


def _add_para(doc: Document, text: str, style: str | None = None, italic: bool = False,
              size: float | None = None, color: RGBColor | None = None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.italic = italic
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color
    p.paragraph_format.space_after = Pt(5)
    return p


def _add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        # Bold a leading "Label — " fragment when present.
        if " — " in item and len(item.split(" — ")[0]) < 60:
            head, tail = item.split(" — ", 1)
            r = p.add_run(head + " — ")
            r.bold = True
            p.add_run(tail)
        else:
            p.add_run(item)
        p.paragraph_format.space_after = Pt(2)


def _add_code(doc: Document, text: str) -> None:
    for i, line in enumerate(text.split("\n")):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(0 if i < len(text.split("\n")) - 1 else 8)
        p.paragraph_format.space_before = Pt(8 if i == 0 else 0)
        run = p.add_run(line if line else " ")
        run.font.name = "Menlo"
        r = run._element
        rpr = r.get_or_add_rPr()
        r_fonts = rpr.find(qn("w:rFonts"))
        if r_fonts is None:
            r_fonts = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(r_fonts)
        r_fonts.set(qn("w:ascii"), "Menlo")
        r_fonts.set(qn("w:hAnsi"), "Menlo")
        run.font.size = Pt(8.5)


def _add_table(doc: Document, header: list[str], rows: list[list[str]],
               widths: list[float] | None = None, font_size: float = 9.0) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Light Grid Accent 1"
    table.autofit = not widths
    hdr = table.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            p.paragraph_format.space_after = Pt(1)
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(font_size)
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx].cells[c_idx]
            cell.text = val
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                for r in p.runs:
                    r.font.size = Pt(font_size)
    if widths:
        for row in table.rows:
            for cell, w in zip(row.cells, widths, strict=True):
                cell.width = Inches(w)
        for col, w in zip(table.columns, widths, strict=True):
            col.width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = GREY
    p.paragraph_format.space_after = Pt(10)


def _add_figure(doc: Document, figures: dict[str, Path], key: str, caption: str,
                width: float = 6.4) -> None:
    doc.add_picture(str(figures[key]), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_before = Pt(6)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(3)
    _add_caption(doc, caption)


def _add_page_field(paragraph) -> None:
    for instr in ("PAGE",):
        fld_begin = OxmlElement("w:fldChar")
        fld_begin.set(qn("w:fldCharType"), "begin")
        instr_text = OxmlElement("w:instrText")
        instr_text.set(qn("xml:space"), "preserve")
        instr_text.text = f" {instr} "
        fld_end = OxmlElement("w:fldChar")
        fld_end.set(qn("w:fldCharType"), "end")
        run = paragraph.add_run()
        run._r.append(fld_begin)
        run._r.append(instr_text)
        run._r.append(fld_end)


def _build_footer(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0]
    p.text = ""
    tab_stops = p.paragraph_format.tab_stops
    tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    run = p.add_run("Solar Array Pointing Algorithm\t")
    run.font.size = Pt(8)
    run.font.color.rgb = GREY
    page_run_para = p
    r = page_run_para.add_run("Page ")
    r.font.size = Pt(8)
    r.font.color.rgb = GREY
    _add_page_field(page_run_para)
    for run in p.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = GREY


# --- document --------------------------------------------------------------------------


def build_document(figures: dict[str, Path]) -> Document:
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        _build_footer(section)

    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(5)

    # ---------------------------------------------------------------- title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Solar Array Pointing Algorithm")
    run.bold = True
    run.font.size = Pt(23)
    run.font.color.rgb = NAVY
    title.paragraph_format.space_after = Pt(2)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(
        "Sun vector, reference frames, gimbal kinematics, and user-configurable travel limits"
    )
    run.italic = True
    run.font.size = Pt(11.5)
    run.font.color.rgb = GREY

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run(
        "Satellite Energy Balance simulator  ·  energy_balance/solar_array.py, "
        "attitude.py, sun.py, orbit.py"
    )
    run.font.size = Pt(9)
    run.font.color.rgb = GREY
    meta.paragraph_format.space_after = Pt(14)

    # ---------------------------------------------------------------- 1
    _add_heading(doc, "1. Scope and summary", 1)
    _add_para(
        doc,
        "This document describes how the simulator decides where each solar array points at "
        "every time step, and how that decision is constrained by the gimbal travel limits and "
        "slew rate the user enters in the sidebar. It covers the full chain: locating the Sun in "
        "inertial space, establishing the orbit and body reference frames, expressing the Sun "
        "direction in body coordinates, solving for the ideal gimbal angles, applying travel "
        "limits and rate limits to produce the achievable angles, and converting the resulting "
        "array orientation into a cosine-of-incidence factor for the power model.",
    )
    _add_para(
        doc,
        "Every numerical example in this document uses the simulator's default values — the "
        "opening preset “LEO 1000 km, 80 deg” with the sidebar's default gimbal limits "
        "(primary axis ±180°, secondary axis ±60°, 1.0 °/s slew). Section 9 "
        "works that case end to end.",
    )
    _add_para(
        doc,
        "The algorithm is deliberately kinematic, not dynamic. It computes the orientation a "
        "perfectly behaved drive would reach subject to its hard stops and maximum rate; it does "
        "not model motor torque, structural flexibility, controller bandwidth, or pointing jitter.",
    )

    _add_heading(doc, "1.1 The pointing chain in one table", 2)
    _add_table(
        doc,
        ["Step", "Module / function", "Produces"],
        [
            ["1. Propagate the orbit", "orbit.py → propagate_orbit",
             "r_eci(t), v_eci(t) in km and km/s"],
            ["2. Locate the Sun", "sun.py → sun_vector_eci",
             "u_sun_eci(t) unit vector, distance in AU"],
            ["3. Illumination", "eclipse.py → illumination_fraction_conical",
             "illumination fraction 0…1"],
            ["4. Build the attitude", "attitude.py → body_to_eci",
             "R(t), the 3×3 body→ECI rotation"],
            ["5. Sun in body frame", "solar_array.py → panel_frames_eci",
             "u_sun_body = Rᵀ u_sun_eci"],
            ["6. Ideal gimbal angles", "solar_array.py → panel_frames_eci",
             "θ₁, θ₂ that aim the normal at the Sun"],
            ["7. Apply travel limits", "solar_array.py → _limited_axis_angles",
             "achievable angles after stops and rate cap"],
            ["8. Actual normal and power", "solar_array.py → array_power_watts",
             "n̂_eci, cos(incidence), panel watts"],
        ],
        widths=[1.9, 2.4, 2.2],
    )
    _add_para(
        doc,
        "Steps 6 and 7 are the pointing algorithm proper. Steps 1–5 supply its inputs and "
        "step 8 consumes its output. All of the geometry is vectorised over the time grid: the "
        "rotation stack R has shape (N, 3, 3) and every angle series has shape (N,).",
    )

    # ---------------------------------------------------------------- 2
    _add_heading(doc, "2. Notation and conventions", 1)
    _add_table(
        doc,
        ["Symbol", "Meaning", "Where it comes from"],
        [
            ["r_eci, v_eci", "Satellite position and velocity in ECI (km, km/s)",
             "propagate_orbit"],
            ["û_sun", "Unit vector from Earth's centre toward the Sun, in ECI",
             "sun_vector_eci"],
            ["R", "Rotation matrix body → ECI, so that r_eci = R r_body", "body_to_eci"],
            ["ŝ", "Sun direction in body coordinates, ŝ = Rᵀ û_sun",
             "panel_frames_eci"],
            ["n̂₀", "Panel rest normal in body coordinates (normal_body)",
             "PanelConfig"],
            ["â₁", "Primary gimbal axis in body coordinates (rotation_axis_body)",
             "PanelConfig"],
            ["â₂", "Secondary gimbal axis, â₂ = â₁ × n̂₀",
             "derived"],
            ["θ₁, θ₂", "Primary (azimuth) and secondary (elevation) gimbal angles",
             "solved per time step"],
            ["n̂", "Actual panel normal after the gimbals have moved", "normal_eci"],
            ["β", "Beta angle: elevation of the Sun above the orbit plane",
             "beta_angle_deg"],
        ],
        widths=[0.9, 3.3, 2.3],
    )
    _add_bullets(
        doc,
        [
            "Rotation convention — a rotation matrix maps body coordinates into ECI "
            "coordinates: r_eci = R r_body. The inverse mapping is therefore the transpose, "
            "r_body = Rᵀ r_eci, which is how the Sun direction is brought into the body "
            "frame (the code writes this as an einsum with the index order reversed).",
            "Angle sign convention — all gimbal angles follow the right-hand rule about "
            "their rotation axis. A positive θ₁ rotates the rest normal toward "
            "+â₂ when viewed down +â₁.",
            "Units — lengths in km, angles in degrees at every interface the user or the "
            "charts touch, radians only inside the trigonometry. Time is seconds from epoch.",
            "Frames are right-handed and orthonormal — every basis the code builds is "
            "explicitly Gram–Schmidt orthogonalised, so no frame is ever slightly skewed by "
            "an input that was not quite perpendicular.",
        ],
    )

    # ---------------------------------------------------------------- 3
    _add_heading(doc, "3. Reference frames", 1)
    _add_para(
        doc,
        "Four frames appear in the pointing calculation. Two of them — the orbit frame and "
        "the body frame — are the ones the user needs to reason about when choosing a "
        "mounting face and a gimbal axis, so they are defined in full below.",
    )

    _add_heading(doc, "3.1 Earth-centred inertial (ECI)", 2)
    _add_para(
        doc,
        "ECI is the common frame in which everything is finally compared. Its origin is the "
        "centre of the Earth; +X points toward the vernal equinox; +Z points along the Earth's "
        "rotation axis toward the north pole; +Y completes the right-handed set in the equatorial "
        "plane. The frame does not rotate with the Earth, so the Sun vector and the satellite "
        "state can be differenced directly without any sidereal bookkeeping.",
    )
    _add_para(
        doc,
        "The simulator treats the mean-equinox-of-date frame produced by the low-precision solar "
        "ephemeris as if it were the same ECI frame used by the orbit propagator. Over the few "
        "decades around J2000 for which the ephemeris is valid, the resulting Sun-direction error "
        "is on the order of 0.01°, which is negligible against the cosine losses and gimbal "
        "limits this document is about.",
    )

    _add_heading(doc, "3.2 Perifocal frame (PQW)", 2)
    _add_para(
        doc,
        "The propagator first solves Kepler's equation in the orbit's own plane. The perifocal "
        "frame has +P toward perigee, +Q 90° ahead of perigee in the direction of motion, and "
        "+W along the orbital angular momentum. Position and velocity are formed there and then "
        "rotated into ECI by the standard three-angle sequence:",
    )
    _add_code(
        doc,
        "R_eci<-pqw = Rz(RAAN) . Rx(inclination) . Rz(argument of perigee)",
    )
    _add_para(
        doc,
        "The propagator advances RAAN and argument of perigee with their secular J2 rates and "
        "advances the mean anomaly with the J2-corrected mean motion, so this rotation changes "
        "slowly over the run. For the default orbit the node regresses at −1.0392° per "
        "day.",
    )

    _add_heading(doc, "3.3 The orbit frame (R, S, W)", 2)
    _add_para(
        doc,
        "The orbit frame — also called the local-vertical / local-horizontal or RSW frame "
        "— is the frame that rides with the satellite and is the reference for nadir-pointing "
        "attitude. It is defined at each instant directly from the position and velocity:",
    )
    _add_table(
        doc,
        ["Axis", "Definition", "Physical meaning"],
        [
            ["R̂", "r_eci / |r_eci|", "Radial, pointing away from Earth (zenith). "
                                            "Its negative is nadir."],
            ["Ŵ", "(r × v) / |r × v|", "Orbit normal, along the angular "
                                                        "momentum vector."],
            ["Ŝ", "Ŵ × R̂", "In-plane and transverse; equals the "
                                                   "velocity direction for a circular orbit."],
        ],
        widths=[0.7, 1.8, 4.0],
    )
    _add_para(
        doc,
        "Two quantities derived from this frame drive the whole pointing problem. The first is "
        "the nadir direction −R̂, which the attitude model aims a chosen body face at. "
        "The second is the beta angle, the elevation of the Sun above the orbit plane:",
    )
    _add_code(doc, "beta = 90 deg - angle( W_hat , u_sun )")
    _add_para(
        doc,
        "Beta sets how far the Sun ever moves out of the orbit plane, and therefore how much "
        "angular travel the gimbals have to cover. At β = 0 the Sun lies in the orbit plane "
        "and sweeps a full 360° around the satellite every revolution; at β = 90° it "
        "sits on the orbit normal and barely moves in the body frame. The default epoch is a "
        "near-equinox case with β ≈ 0.09°, which is the hardest case for a gimbal "
        "with limited travel — see section 9.",
    )

    _add_figure(
        doc, figures, "frames",
        "Figure 1. Left: the orbit frame at an instant, with the radial, along-track and "
        "orbit-normal directions. Right: the default body frame seen from the orbit normal, "
        "with the two wings on the ±X booms. The panel rest normal +Zᵇ is the nadir "
        "face, so at zero gimbal angle the arrays face the Earth, not the Sun.",
    )

    _add_heading(doc, "3.4 The body frame", 2)
    _add_para(
        doc,
        "The body frame is fixed to the spacecraft structure. Panel normals and gimbal axes are "
        "entered in this frame, and the user names its six faces by the labels +X, −X, +Y, "
        "−Y, +Z and −Z, which map to the obvious unit vectors (BODY_FACES in config.py). "
        "How the body frame sits relative to the orbit is decided entirely by the attitude mode:",
    )
    _add_table(
        doc,
        ["Attitude mode", "How the body frame is built", "Resulting orientation"],
        [
            ["Nadir pointing (default)",
             "+Zᵇ is driven to nadir (−R̂); the chosen yaw axis (default +Xᵇ) "
             "is driven to the velocity direction, orthogonalised against nadir; the third axis "
             "completes the right-handed set.",
             "+Zᵇ = nadir, +Xᵇ ≈ along-track, +Yᵇ = −Ŵ. The body "
             "frame rotates once per orbit in inertial space."],
            ["Sun pointing",
             "A chosen body axis (default +Zᵇ) is driven onto the Sun line; the remaining "
             "phase about that line is fixed by an arbitrary ECI +Z hint vector.",
             "The commanded axis holds the Sun. The Sun direction in body coordinates is "
             "constant, so an array whose rest normal is that axis needs no gimbal motion at all."],
            ["Inertial",
             "R is the identity matrix.",
             "Body axes coincide with ECI axes; the Sun sweeps through the body frame once per "
             "year."],
        ],
        widths=[1.3, 2.7, 2.5],
    )
    _add_para(
        doc,
        "The default attitude is nadir pointing with the +X yaw axis, so for the default circular "
        "orbit the body axes line up with the orbit frame as follows. This is worth internalising, "
        "because it explains the gimbal behaviour in section 9:",
    )
    _add_code(
        doc,
        "+X_body  ->  +S_hat   (along-track / velocity)\n"
        "+Y_body  ->  -W_hat   (negative orbit normal)\n"
        "+Z_body  ->  -R_hat   (nadir, toward the Earth)",
    )
    _add_para(
        doc,
        "Note that for an eccentric orbit the velocity is not exactly perpendicular to the radius, "
        "so the yaw axis is placed along the component of velocity perpendicular to nadir rather "
        "than along velocity itself. Nadir always wins; the yaw axis only resolves the remaining "
        "rotation about it.",
    )

    _add_heading(doc, "3.5 The panel (gimbal) frame", 2)
    _add_para(
        doc,
        "Each panel carries its own orthonormal triad, built from two user inputs and one derived "
        "vector:",
    )
    _add_bullets(
        doc,
        [
            "n̂₀ — the rest normal (normal_body). The direction the panel faces "
            "when both gimbal angles are zero. The user picks a body face or enters a custom "
            "vector; the code normalises it.",
            "â₁ — the primary gimbal axis (rotation_axis_body). Physically the "
            "boom or drive shaft the panel rotates about. For a two-wing array the sidebar sets "
            "this from the chosen mounting sides (±X or ±Y) and both wings share it.",
            "â₂ = â₁ × n̂₀ — the secondary gimbal axis, "
            "derived, not entered. It is perpendicular to both the boom and the rest normal.",
        ],
    )
    _add_para(
        doc,
        "The convention assumes the rest normal is perpendicular to the primary axis, which is "
        "what a real boom-mounted wing looks like, and which the defaults satisfy "
        "(n̂₀ = +Zᵇ, â₁ = +Xᵇ, so â₂ = −Yᵇ). "
        "Then (n̂₀, â₂, â₁) is an orthonormal right-handed triad "
        "and the gimbal angles below have a clean interpretation. If a user enters a rest normal "
        "that is not perpendicular to the axis, the one-axis model still behaves correctly — "
        "it keeps the component of the normal that lies along the axis fixed and rotates only the "
        "perpendicular part — but the two-axis angles lose their azimuth/elevation meaning.",
    )

    _add_heading(doc, "3.6 Frame chain summary", 2)
    _add_code(
        doc,
        "PQW  --Rz(RAAN) Rx(i) Rz(argp)-->  ECI  <--R--  BODY  <--gimbals--  PANEL\n"
        "\n"
        "u_sun_body = R^T u_sun_eci          (Sun into the body frame)\n"
        "n_eci      = R n_body               (panel normal back out to ECI)\n"
        "cos(inc)   = clip( n_eci . u_sun_eci , 0 , 1 )",
    )

    # ---------------------------------------------------------------- 4
    _add_heading(doc, "4. Locating the Sun", 1)
    _add_para(
        doc,
        "The Sun direction comes from the Astronomical Almanac / Vallado low-precision series, "
        "evaluated at each time step. Working in Julian centuries T from J2000, the code forms the "
        "mean longitude and mean anomaly of the Sun, applies the equation-of-centre correction to "
        "get the ecliptic longitude, computes the Earth–Sun distance, and rotates from the "
        "ecliptic into the equatorial frame by the obliquity:",
    )
    _add_code(
        doc,
        "lambda_M = 280.460 + 36000.771 T                       (deg)\n"
        "M        = 357.5277233 + 35999.05034 T                 (deg)\n"
        "lambda   = lambda_M + 1.914666471 sin M + 0.019994643 sin 2M\n"
        "r_AU     = 1.000140612 - 0.016708617 cos M - 0.000139589 cos 2M\n"
        "eps      = 23.439291 - 0.0130042 T                     (deg)\n"
        "\n"
        "r_sun_eci = r_AU * [ cos(lambda), cos(eps) sin(lambda), sin(eps) sin(lambda) ] * AU_KM\n"
        "u_sun_eci = r_sun_eci / |r_sun_eci|",
    )
    _add_para(
        doc,
        "Two products come out of this. The unit vector û_sun drives all of the pointing "
        "geometry. The distance r_AU drives the solar flux through an inverse-square scaling, so a "
        "run near perihelion sees a few percent more flux than one near aphelion:",
    )
    _add_code(doc, "flux(t) = solar_constant / r_AU(t)^2        (W/m^2)")
    _add_para(
        doc,
        "At the default epoch of 2026-03-20 12:00 UTC the Sun is at 0.995838 AU, so the 1361 "
        "W/m² solar constant becomes 1372.40 W/m² at the spacecraft, and the Sun "
        "direction is û_sun = (0.999998, −0.001756, −0.000761) — essentially "
        "along the ECI +X axis, as expected three days after the March equinox.",
    )

    # ---------------------------------------------------------------- 5
    _add_heading(doc, "5. The ideal pointing solution", 1)
    _add_para(
        doc,
        "With the Sun direction expressed in body coordinates as ŝ = Rᵀ û_sun, "
        "each panel solves for the orientation that maximises the cosine of incidence, subject to "
        "whatever degrees of freedom its mounting gives it. This is the “ideal” or "
        "“commanded” solution; limits are applied afterwards in section 6.",
    )

    _add_heading(doc, "5.1 Fixed panels", 2)
    _add_para(
        doc,
        "There is nothing to solve. The normal is the rest normal carried into ECI, "
        "n̂ = R n̂₀, both gimbal angles are reported as zero, and the cosine of "
        "incidence is whatever the attitude happens to deliver.",
    )

    _add_heading(doc, "5.2 One-axis tracking", 2)
    _add_para(
        doc,
        "A one-axis panel can rotate only about â₁, so its normal is confined to the "
        "plane perpendicular to that axis. The best achievable normal is therefore the Sun "
        "direction projected into that plane and renormalised:",
    )
    _add_code(
        doc,
        "s_par  = (s_hat . a1_hat) a1_hat        component along the axis\n"
        "s_perp = s_hat - s_par                  component in the rotation plane\n"
        "n_best = s_perp / |s_perp|              (falls back to n0 if |s_perp| ~ 0)",
    )
    _add_para(
        doc,
        "The resulting cosine of incidence has a closed form that is useful for sizing: "
        "cos(incidence) = |s_perp| = √(1 − (ŝ·â₁)²). The only "
        "loss is the Sun's angle out of the rotation plane; a one-axis array can never do better "
        "than that, no matter how it is driven. The commanded angle reported to the user is the "
        "signed rotation about the axis from the rest normal to this tracked normal, formed with "
        "a four-quadrant arctangent so it is continuous through ±180°:",
    )
    _add_code(
        doc,
        "sin_part = ( n0_hat x n_best ) . a1_hat\n"
        "cos_part =   n0_hat . n_best\n"
        "theta    = atan2( sin_part , cos_part )          (deg, in -180 .. +180)",
    )

    _add_heading(doc, "5.3 Two-axis tracking", 2)
    _add_para(
        doc,
        "A two-axis gimbal is modelled as a classic azimuth-over-elevation mount. The outer "
        "(primary) gimbal rotates the yoke by θ₁ about the body-fixed boom axis "
        "â₁; the inner (secondary) gimbal then tips the panel by θ₂ about the "
        "secondary axis, which rides on the yoke. Composed, the actual normal is:",
    )
    _add_code(
        doc,
        "n_body(theta1, theta2) =\n"
        "      cos(theta2) cos(theta1) * n0_hat\n"
        "    + cos(theta2) sin(theta1) * a2_hat\n"
        "    - sin(theta2)             * a1_hat",
    )
    _add_para(
        doc,
        "Because (n̂₀, â₂, â₁) is an orthonormal triad, this is "
        "nothing more than a spherical coordinate parameterisation of the unit sphere in that "
        "triad, which means the ideal angles can be read off directly by projecting the Sun "
        "direction onto the three axes:",
    )
    _add_code(
        doc,
        "p = s_hat . n0_hat          q = s_hat . a2_hat          r = s_hat . a1_hat\n"
        "\n"
        "theta1_ideal = atan2( q , p )          azimuth about the boom, -180 .. +180 deg\n"
        "theta2_ideal = -asin( r )              elevation toward the boom,  -90 ..  +90 deg",
    )
    _add_para(
        doc,
        "Substituting these back into the kinematics gives n_body = p n̂₀ + q "
        "â₂ + r â₁ = ŝ exactly, so an unconstrained two-axis array "
        "always achieves cos(incidence) = 1 while the satellite is in sunlight. The code exploits "
        "this: when no limits are configured it skips the kinematics entirely and sets the ECI "
        "normal equal to the Sun vector, which keeps the unlimited case bit-for-bit exact and "
        "cheap. The angle solution is still reported so the charts can show what the drives are "
        "doing.",
    )

    _add_figure(
        doc, figures, "gimbal",
        "Figure 2. The two gimbal rotations. (a) Looking down the boom axis, the primary gimbal "
        "swings the normal through θ₁ in the plane spanned by the rest normal and the "
        "secondary axis. (b) In the perpendicular view, the secondary gimbal tips the normal "
        "through θ₂ toward the boom; the shaded sector is the legal travel left by the "
        "default ±60° stop.",
    )

    _add_heading(doc, "5.4 The elevation singularity", 2)
    _add_para(
        doc,
        "An azimuth-over-elevation mount has a singularity where the inner axis points along the "
        "line of sight. Setting θ₂ = ±90° in the kinematics collapses the first "
        "two terms and leaves n_body = ∓â₁ regardless of θ₁: when the Sun "
        "lies exactly along the boom, the azimuth angle has no effect on where the panel points.",
    )
    _add_para(
        doc,
        "Near that configuration the azimuth command becomes ill-conditioned — a small motion "
        "of the Sun demands a large and fast change in θ₁. The simulator does not "
        "regularise this; it solves the exact inverse kinematics and lets the slew-rate limit "
        "absorb the excursion. The practical consequence is a brief azimuth “flip” "
        "whenever the Sun crosses the boom axis, visible in Figure 3. The energy cost is small, "
        "because near the singularity the elevation angle is close to ±90° and, as "
        "section 7.2 shows, an azimuth error there barely moves the normal.",
    )

    # ---------------------------------------------------------------- 6
    _add_heading(doc, "6. Gimbal travel limits", 1)
    _add_para(
        doc,
        "Real solar array drives are not free to rotate forever. Cable harnesses and slip-ring "
        "designs impose an accumulated-travel budget, and simple mounts have hard mechanical "
        "stops. The simulator models both, and which one applies is inferred from the width of the "
        "interval the user enters rather than from a separate switch.",
    )

    _add_heading(doc, "6.1 What the user enters", 2)
    _add_table(
        doc,
        ["Field (PanelConfig)", "Sidebar control", "Default", "Applies to"],
        [
            ["gimbal_min_deg / gimbal_max_deg", "“Limit gimbal travel” / "
             "“Limit primary axis travel”, minimum and maximum angle",
             "checked, −180° / +180°", "1-axis and 2-axis"],
            ["gimbal2_min_deg / gimbal2_max_deg", "“Limit secondary axis travel”, "
             "secondary minimum and maximum", "checked, −60° / +60°", "2-axis only"],
            ["max_slew_rate_deg_s", "“Max slew (deg/s)”", "1.0 °/s",
             "every limited axis"],
        ],
        widths=[1.9, 2.3, 1.3, 1.0],
    )
    _add_para(
        doc,
        "Note the distinction between the two kinds of default. PanelConfig leaves all four limit "
        "fields unset (None), which means unlimited travel; the sidebar, however, presents both "
        "limit checkboxes already ticked with ±180° and ±60°. A default run of "
        "the app is therefore a gimbal-limited run, and that is the case documented in section 9.",
    )
    _add_para(
        doc,
        "Validation happens in the PanelConfig model and rejects the combinations that have no "
        "physical meaning:",
    )
    _add_bullets(
        doc,
        [
            "A minimum and a maximum must be supplied together.",
            "The interval must contain the 0° rest position, and the maximum must exceed the "
            "minimum. This guarantees the drive can always sit at rest and that the interval is "
            "non-degenerate.",
            "Primary-axis limits are accepted only for one-axis and two-axis panels; a fixed panel "
            "has no drive to limit.",
            "Secondary-axis limits are accepted only for two-axis panels and must lie within "
            "±90°, because the elevation angle is an arcsine and can never exceed that "
            "range.",
        ],
    )

    _add_heading(doc, "6.2 Step one: unwrap the command into a continuous trajectory", 2)
    _add_para(
        doc,
        "The ideal angles come out of an arctangent and therefore live on (−180°, "
        "+180°]. A drive does not see it that way: a command that walks from +179° to "
        "−179° is a 2° motion, not a 358° one. Before any limit is applied the "
        "series is unwrapped into a continuous trajectory (numpy.unwrap on the radian series), "
        "which is also what makes accumulated travel a meaningful quantity. The time grid must be "
        "strictly increasing and must have one sample per geometry sample; both are checked.",
    )

    _add_heading(doc, "6.3 Step two: cable wrap or hard stop", 2)
    _add_para(
        doc,
        "The span of the interval decides which of two very different behaviours applies:",
    )
    _add_code(
        doc,
        "if (maximum - minimum) >= 360 deg :  cable-wrap limit, the drive can rewind\n"
        "else                              :  hard mechanical stop, the drive simply clamps",
    )
    _add_para(
        doc,
        "The reasoning is geometric. If the legal interval is at least a full turn wide then for "
        "any commanded angle θ there is always some equivalent angle θ + 360k inside the "
        "interval, which points the panel in exactly the same direction. The drive can therefore "
        "always keep tracking the Sun perfectly; the limit costs travel time, not incidence. If "
        "the interval is narrower than a full turn, no such equivalent angle need exist, and once "
        "the Sun leaves the reachable sector the panel has no choice but to sit at the stop and "
        "accept the cosine loss.",
    )

    _add_heading(doc, "6.4 Cable-wrap mode: branch tracking and rewind", 2)
    _add_para(
        doc,
        "In cable-wrap mode (_rewind_branch_angles) the controller holds a branch index k and "
        "follows the target θ_ideal + 360k. The branch is kept as long as it stays legal, so "
        "the drive does not thrash between equivalent angles. When continuing on the current "
        "branch would leave the interval, a new legal branch is chosen — the one nearest the "
        "drive's current position, so the unwind is as short as possible — and the panel is "
        "flagged as rewinding until it arrives:",
    )
    _add_code(
        doc,
        "branch = nearest legal k to the current position\n"
        "for each step i:\n"
        "    target = ideal_unwrapped[i] + 360 * branch\n"
        "    if target outside [minimum, maximum]:\n"
        "        branch = nearest legal k to actual[i-1]      # pick a 360-deg-equivalent angle\n"
        "        target = ideal_unwrapped[i] + 360 * branch\n"
        "        rewinding = True\n"
        "    step   = max_slew_rate * (t[i] - t[i-1])\n"
        "    actual[i] = clip( actual[i-1] + clip(target - actual[i-1], -step, +step),\n"
        "                      minimum, maximum )\n"
        "    if actual[i] has reached target:  rewinding = False",
    )
    _add_para(
        doc,
        "The rewind is not free. The drive turns through most of a revolution at its maximum slew "
        "rate, and the power model uses the array's actual orientation throughout, so generated "
        "power dips while the panel sweeps away from the Sun and back. The rewinding flag is "
        "carried through to the UI so those intervals can be marked on the angle chart. The model "
        "does not schedule rewinds opportunistically; a rewind begins at the moment continued "
        "tracking would cross a limit, even if waiting for the next eclipse would have been free.",
    )

    _add_heading(doc, "6.5 Hard-stop mode: clamping", 2)
    _add_para(
        doc,
        "A hard stop (_hard_stop_angles) has no equivalent branch to escape to, so the logic "
        "collapses to a clamp of the unwrapped command followed by the same rate limit:",
    )
    _add_code(
        doc,
        "target = clip( ideal_unwrapped, minimum, maximum )\n"
        "for each step i:\n"
        "    step      = max_slew_rate * (t[i] - t[i-1])\n"
        "    actual[i] = clip( actual[i-1] + clip(target[i] - actual[i-1], -step, +step),\n"
        "                      minimum, maximum )\n"
        "rewinding = False everywhere",
    )
    _add_para(
        doc,
        "When the Sun goes out of reach the panel parks at the nearer stop and the incidence "
        "angle degrades smoothly — there is no discontinuity and no attempt to give up and "
        "return to rest. The reported target angle for a hard-stop axis is the clamped "
        "(legal) command, not the unreachable ideal one; the charts therefore show the target "
        "lying on the limit line while the drive is pinned.",
    )
    _add_para(
        doc,
        "The secondary axis of a two-axis panel is always treated as a hard stop. Its angle is an "
        "arcsine bounded to ±90°, so a 360°-wide interval cannot exist and a rewind "
        "is physically meaningless.",
    )

    _add_heading(doc, "6.6 Rate limiting and the initial condition", 2)
    _add_para(
        doc,
        "Every limited axis is rate limited. At each step the drive may move at most "
        "max_slew_rate_deg_s multiplied by the actual time step, so a coarse time grid implicitly "
        "allows a larger per-step motion — the limit is a rate, not a per-sample cap. The "
        "move is then clamped into the legal interval a second time, which guarantees the "
        "reported angle never leaves the user's limits even transiently.",
    )
    _add_para(
        doc,
        "Both limited modes start the simulation with the drive already on its command rather "
        "than at the 0° rest position: actual[0] is set to target[0], which the branch "
        "selection and the clamp both guarantee lies inside the legal interval. Epoch is "
        "therefore a spacecraft that has been tracking all along, not one that has just "
        "deployed. Starting from rest instead would open every run with a slew of up to a full "
        "revolution — 180° taking three minutes in the default configuration — that belongs to "
        "the initial condition rather than to the orbit, and that shows up as the array visibly "
        "flipping in the first seconds of the Frames animation.",
    )
    _add_para(
        doc,
        "Two details follow from this. For a cable-wrap axis the starting branch is still the "
        "legal one nearest rest, so the drive begins unwound and the first rewind, if any, is "
        "driven by the orbit rather than by where the simulation happened to start. For a "
        "hard-stop axis whose Sun is already out of reach at epoch, the command is the clamped "
        "one, so the drive starts pinned at the nearer stop — that is a limit being hit, not a "
        "transient. Section 9.3 shows the first samples of the default case.",
    )

    _add_heading(doc, "6.7 Recombining the two axes", 2)
    _add_para(
        doc,
        "For a limited two-axis panel the two axes are processed independently — each gets "
        "its own unwrap, limit mode, clamp and rate cap — and the achievable angles are then "
        "substituted back into the same kinematics used to derive them:",
    )
    _add_code(
        doc,
        "n_body = cos(th2_act) cos(th1_act) n0_hat\n"
        "       + cos(th2_act) sin(th1_act) a2_hat\n"
        "       - sin(th2_act)              a1_hat\n"
        "n_eci  = R n_body",
    )
    _add_para(
        doc,
        "This is a per-axis clamp, not a joint-workspace optimisation. When one axis saturates, "
        "the model does not re-solve the other axis to recover as much cosine as the remaining "
        "freedom would allow; it keeps commanding the unconstrained ideal on the free axis. For an "
        "azimuth-over-elevation mount that is both the honest representation of how a simple drive "
        "controller behaves and close to optimal in practice, but it is a modelling choice worth "
        "knowing about when interpreting a heavily saturated case.",
    )
    _add_para(
        doc,
        "The same substitution is applied to a limited one-axis panel, using its single angle to "
        "rebuild the normal about the boom axis. Because this recombination is exact, the drawn "
        "geometry and the computed power can never "
        "disagree — the normal used for the cosine is the same vector the Frames tab renders.",
    )

    # ---------------------------------------------------------------- 7
    _add_heading(doc, "7. From orientation to power", 1)

    _add_heading(doc, "7.1 The incidence factor", 2)
    _add_para(
        doc,
        "Once the actual normal is in ECI, the incidence factor is a clipped dot product, and the "
        "per-panel power follows:",
    )
    _add_code(
        doc,
        "cos_inc = clip( n_eci . u_sun_eci , 0 , 1 )\n"
        "\n"
        "P = flux * area * cell_eff * packing * inherent * degradation\n"
        "      * (1 - temperature_loss) * cos_inc * illumination",
    )
    _add_para(
        doc,
        "The clip at zero is what makes a back-lit panel produce nothing rather than negative "
        "power. The illumination factor from the conical eclipse model multiplies in separately, "
        "so a panel that is perfectly pointed but in umbra produces nothing, and the gimbals keep "
        "tracking through eclipse rather than parking.",
    )

    _add_heading(doc, "7.2 What a limit actually costs", 2)
    _add_para(
        doc,
        "Writing the ideal angles as (θ₁, θ₂) and the achieved angles as "
        "(θ₁', θ₂'), the dot product of the actual normal with the Sun reduces "
        "to the spherical law of cosines:",
    )
    _add_code(
        doc,
        "cos_inc = cos(th2') cos(th2) cos(th1' - th1) + sin(th2') sin(th2)",
    )
    _add_para(
        doc,
        "Two special cases make the behaviour easy to reason about, and both are worth keeping in "
        "mind when choosing limits:",
    )
    _add_bullets(
        doc,
        [
            "Elevation-only error — if the azimuth is on target, cos_inc = "
            "cos(θ₂ − θ₂'). The loss depends only on how many degrees the "
            "secondary axis fell short. A 30° shortfall costs 13%; a 10° shortfall costs "
            "1.5%.",
            "Azimuth-only error — if the elevation is on target, cos_inc = 1 − "
            "cos²(θ₂)·(1 − cos Δθ₁). The penalty is scaled "
            "by cos² of the elevation angle, so azimuth errors are harmless near the "
            "singularity (θ₂ ≈ ±90°) and most expensive at "
            "θ₂ = 0. This is why the azimuth flip in Figure 3 costs so little.",
        ],
    )
    _add_para(
        doc,
        "Both identities were checked against the simulator over a full day of the default "
        "configuration and agree with the computed cos_inc to within 8×10⁻¹⁵, "
        "which is floating-point noise.",
    )

    # ---------------------------------------------------------------- 8
    _add_heading(doc, "8. What the algorithm reports", 1)
    _add_para(
        doc,
        "panel_frames_eci returns a dictionary per panel, and array_power_watts stacks it across "
        "panels onto the results DataFrame. These are the series the Frames tab plots and the "
        "“Inspect frame” table reads:",
    )
    _add_table(
        doc,
        ["Key", "Shape", "Meaning"],
        [
            ["normal_eci", "(N, 3)", "Actual panel normal in ECI, after all limits."],
            ["edge_eci", "(N, 3)", "In-plane direction used to draw the plate."],
            ["angle_deg", "(N,)", "1-axis: the signed gimbal angle. 2-axis: the total slew "
                                  "angle from the rest normal to the actual normal."],
            ["target_angle_deg", "(N,)", "The legal commanded angle (1-axis), or the total slew "
                                         "angle to the Sun (2-axis)."],
            ["angle1_deg / angle2_deg", "(N,)", "Actual primary and secondary axis angles. "
                                                "A 1-axis panel reports its angle as angle1 and "
                                                "zero for angle2."],
            ["target_angle1_deg / target_angle2_deg", "(N,)",
             "The corresponding commanded angles, after branch selection or clamping."],
            ["rewinding", "(N,)", "True while a cable-wrap axis is unwinding. Always False for "
                                  "hard stops."],
            ["cos_incidence", "(N,)", "The clipped cosine factor fed to the power model."],
        ],
        widths=[1.9, 0.7, 3.9],
    )
    _add_para(
        doc,
        "The difference between an angle series and its target series is exactly the pointing "
        "error introduced by the limits: they coincide whenever the drive is tracking freely and "
        "separate while it is slewing, rewinding or pinned at a stop.",
    )

    # ---------------------------------------------------------------- 9
    doc.add_page_break()
    _add_heading(doc, "9. Worked example with the default values", 1)
    _add_para(
        doc,
        "This section runs the algorithm end to end on the simulator's opening configuration, with "
        "the sidebar's default gimbal limits in force.",
    )

    _add_heading(doc, "9.1 Inputs", 2)
    _add_table(
        doc,
        ["Group", "Setting", "Default value"],
        [
            ["Mission", "Epoch / duration / step", "2026-03-20 12:00:00 UTC / 1.0 day / 30 s "
                                                   "(2881 samples)"],
            ["Orbit", "Perigee and apogee altitude", "1000 km circular (a = 7378.137 km, e = 0)"],
            ["", "Inclination / RAAN / argument of perigee / true anomaly",
             "80° / 0° / 0° / 0°"],
            ["Attitude", "Mode / yaw axis", "Nadir pointing / +X"],
            ["Array", "Number of wings, area", "2 wings on the ±X booms, 4.0 m² each"],
            ["", "Rest normal / mounting / primary axis",
             "+Z body / 2-axis tracking / (1, 0, 0) body"],
            ["", "Primary limit / secondary limit / max slew",
             "−180°…+180° / −60°…+60° / 1.0 °/s"],
            ["", "Cell efficiency, packing, inherent, temp loss", "0.30, 0.85, 0.77, 0.10"],
            ["", "Degradation / mission year / solar constant",
             "2.5 %/yr / year 0 / 1361 W/m²"],
        ],
        widths=[0.9, 2.6, 3.0],
    )
    _add_para(
        doc,
        "Two consequences of these defaults set up everything that follows. First, both wings "
        "share one boom axis and one rest normal, so they are kinematically identical and generate "
        "identical power; the ±X naming only affects how they are drawn. Second, the "
        "effective per-panel factor — efficiency × packing × inherent degradation "
        "× (1 − temperature loss) — is 0.176715, so at the epoch flux of 1372.40 "
        "W/m² a perfectly pointed 4 m² wing makes 970.09 W and the array peaks at 1940.19 "
        "W.",
    )

    _add_heading(doc, "9.2 Geometry at epoch", 2)
    _add_para(
        doc,
        "At t = 0 the satellite sits on the ECI +X axis at the ascending node. The propagator, "
        "ephemeris and attitude builder give:",
    )
    _add_code(
        doc,
        "r_eci      = (7378.137, 0, 0) km          v_eci = (0, 1.276338, 7.238474) km/s\n"
        "R_hat      = (1, 0, 0)                    W_hat = (0, -0.984808, 0.173648)\n"
        "u_sun_eci  = (0.999998, -0.001756, -0.000761)      distance = 0.995838 AU\n"
        "beta       = +0.0915 deg                  (range over the day: -1.761 .. +0.091 deg)\n"
        "\n"
        "R (body -> ECI) =  [  0        0       -1      ]\n"
        "                   [  0.173648 0.984808 0      ]\n"
        "                   [  0.984808 -0.173648 0     ]\n"
        "\n"
        "+X_body -> (0, 0.173648, 0.984808) = velocity direction\n"
        "+Y_body -> (0, 0.984808, -0.173648) = -W_hat\n"
        "+Z_body -> (-1, 0, 0)               = nadir",
    )
    _add_para(
        doc,
        "Rotating the Sun into the body frame gives ŝ = (−0.001054, −0.001597, "
        "−0.999998) — the Sun is almost exactly along −Zᵇ, which is zenith. "
        "That is the crux of the default geometry: the panel rest normal is +Zᵇ, the nadir "
        "face, so the Sun starts out behind the panels and the gimbals must swing nearly a half "
        "turn to see it. Projecting onto the triad:",
    )
    _add_code(
        doc,
        "p = s . n0_hat = -0.999998      q = s . a2_hat = +0.001597     r = s . a1_hat = -0.001054\n"
        "theta1_ideal = atan2(q, p) = +179.9085 deg\n"
        "theta2_ideal = -asin(r)    =   +0.0604 deg",
    )
    _add_para(
        doc,
        "The beta angle of 0.09° is the second decisive fact. The Sun lies essentially in the "
        "orbit plane, and the boom axis +Xᵇ is the along-track direction, which rotates "
        "through the orbit plane once per revolution. The Sun therefore sweeps past the boom axis "
        "twice per orbit — straight through the elevation singularity of section 5.4.",
    )

    _add_heading(doc, "9.3 The first samples", 2)
    _add_para(
        doc,
        "Both drives begin on command. The azimuth command is already at +179.9° at epoch and "
        "the drive is there too, so nothing has to catch up: both axes track exactly from the "
        "first sample and the incidence factor is 1.0000 throughout. The elevation command "
        "walks away from rest at about 0.057 °/s, far inside the 1 °/s cap, so it is never "
        "rate limited either:",
    )
    _add_table(
        doc,
        ["t (s)", "θ₁ command", "θ₁ actual", "θ₂ command",
         "θ₂ actual", "cos(inc)"],
        [
            ["0", "+179.909°", "+179.909°", "+0.060°", "+0.060°", "1.0000"],
            ["30", "+179.909°", "+179.909°", "+1.771°", "+1.771°", "1.0000"],
            ["60", "+179.910°", "+179.910°", "+3.481°", "+3.481°", "1.0000"],
            ["90", "+179.910°", "+179.910°", "+5.191°", "+5.191°", "1.0000"],
            ["120", "+179.910°", "+179.910°", "+6.901°", "+6.901°", "1.0000"],
            ["150", "+179.911°", "+179.911°", "+8.612°", "+8.612°", "1.0000"],
            ["180", "+179.911°", "+179.911°", "+10.322°", "+10.322°",
             "1.0000"],
        ],
        widths=[0.7, 1.2, 1.2, 1.2, 1.2, 0.9],
    )
    _add_para(
        doc,
        "It is worth seeing what the alternative would have cost. Had the drives started at "
        "the 0° rest position, the primary axis would have needed its full 180° of travel at "
        "1 °/s — exactly 180 s, six samples — before the incidence factor first reached "
        "1.0000, and the array would have been nearly edge-on to the Sun for the first two "
        "minutes of every run. Those six partly wasted samples are worth roughly 75 Wh over "
        "the day, which is essentially the whole difference between the “primary limit only” "
        "and “no limits” rows of the sensitivity table in section 9.6. The arithmetic is real, "
        "but it measures where the simulation happens to begin rather than anything about the "
        "orbit or the limits, which is why the model starts the drive on command instead.",
    )

    _add_heading(doc, "9.4 One orbit in steady state", 2)
    _add_para(
        doc,
        "Sampled every ten minutes through the second revolution, the pattern is clear: the "
        "elevation axis does nearly all of the work, sweeping ±90° once per orbit and "
        "spending the extremes pinned at its ±60° stops, while the azimuth axis sits "
        "near 0° or ±180° and flips between them as the Sun crosses the boom.",
    )
    _add_table(
        doc,
        ["t (s)", "Sun in body frame (ŝ)", "θ₁ ideal", "θ₁ actual",
         "θ₂ ideal", "θ₂ actual", "cos(inc)"],
        [
            ["6330", "(−0.016, +0.001, −1.000)", "−179.96°",
             "−179.96°", "+0.93°", "+0.93°", "1.0000"],
            ["6930", "(−0.576, +0.001, −0.818)", "−179.93°",
             "−179.93°", "+35.13°", "+35.13°", "1.0000"],
            ["7530", "(−0.936, +0.001, −0.353)", "−179.80°",
             "−179.80°", "+69.34°", "+60.00°", "0.9867"],
            ["8130", "(−0.972, +0.001, +0.234)", "−0.35°", "−0.35°",
             "+76.46°", "+60.00°", "0.9590"],
            ["8730", "(−0.672, +0.002, +0.740)", "−0.13°", "−0.13°",
             "+42.25°", "+42.25°", "1.0000  (eclipse)"],
            ["9930", "(+0.441, +0.002, +0.898)", "−0.14°", "−0.14°",
             "−26.16°", "−26.16°", "1.0000  (eclipse)"],
            ["10530", "(+0.869, +0.002, +0.495)", "−0.27°", "−0.27°",
             "−60.37°", "−60.00°", "1.0000"],
            ["11130", "(+0.997, +0.003, −0.080)", "−178.15°",
             "−104.54°", "−85.43°", "−60.00°", "0.8745"],
            ["11730", "(+0.780, +0.003, −0.626)", "−179.74°",
             "−179.74°", "−51.22°", "−51.22°", "1.0000"],
            ["12330", "(+0.293, +0.003, −0.956)", "−179.82°",
             "−179.82°", "−17.02°", "−17.02°", "1.0000"],
        ],
        widths=[0.55, 1.55, 0.85, 0.85, 0.85, 0.85, 1.0],
        font_size=8.0,
    )
    _add_para(
        doc,
        "The sample at t = 11130 s shows both mechanisms at once. The elevation command of "
        "−85.4° is beyond the −60° stop, and the azimuth is mid-flip at "
        "−104.5° against a command of −178.2° — a 73.6° lag imposed "
        "by the slew rate. Feeding those four angles into the spherical law of cosines from "
        "section 7.2 gives 0.500 × 0.0798 × cos 73.6° + 0.866 × 0.9968 = 0.8745, "
        "matching the simulator exactly. Note how little the 73.6° azimuth lag costs: "
        "because the elevation is at 60°, the azimuth penalty is scaled by cos²(60°) "
        "= 0.25.",
    )

    _add_figure(
        doc, figures, "trace",
        "Figure 3. Two orbits of the default configuration. Top: the azimuth axis librates between "
        "0° and ±180°, flipping as the Sun crosses the boom; the rate cap rounds "
        "each flip. Middle: the elevation command is a near-triangular sweep whose peaks are cut "
        "off by the ±60° stops. Bottom: the resulting incidence factor, which holds at 1.0 "
        "from epoch and dips shallowly at each saturation.",
        width=6.2,
    )

    _add_heading(doc, "9.5 Statistics over the day", 2)
    _add_table(
        doc,
        ["Quantity", "Value"],
        [
            ["Orbital period / eclipse", "105.18 min; about 35 min of eclipse per orbit "
                                         "(sunlit 66.1% of the day)"],
            ["Samples in sunlight", "1904 of 2881"],
            ["Secondary axis pinned at a ±60° stop", "50.5% of sunlit samples"],
            ["Largest secondary-axis shortfall", "29.84° (command reaches 89.84°)"],
            ["Azimuth rate the ideal command demands", "up to 5.92 °/s; exceeds the 1 °/s "
                                                       "cap on 45 samples per day"],
            ["Elevation rate demanded", "at most 0.057 °/s — never rate limited"],
            ["Mean cos(incidence) in sunlight", "0.9773 (minimum 0.8562)"],
            ["Mean / 95th percentile / worst off-Sun angle", "7.50° / 27.53° / "
                                                             "31.10°"],
            ["Cable-wrap rewinds", "none — the azimuth command never leaves "
                                   "±180°"],
        ],
        widths=[2.8, 3.7],
    )
    _add_para(
        doc,
        "That last line deserves emphasis. The default ±180° interval spans exactly "
        "360°, so it is a cable-wrap limit and rewinds are available — but the azimuth "
        "librates rather than winding up, so the unwrapped command never leaves the interval and "
        "no rewind is ever triggered. Changing the primary limit to ±270° produces "
        "the same energy for the same reason. To see the rewind machinery exercised on this "
        "orbit the interval has to be offset rather than widened: −360°…0° is "
        "also a legal cable-wrap limit and triggers 12 rewind samples per day.",
    )

    _add_heading(doc, "9.6 What the limits cost", 2)
    _add_para(
        doc,
        "Re-running the same default mission while changing only the limits isolates their effect "
        "on a day of generated energy:",
    )
    _add_table(
        doc,
        ["Configuration", "Energy (Wh/day)", "vs unlimited", "Mean cos(inc), sunlit"],
        [
            ["2-axis, no limits", "30 731.4", "—", "1.0000"],
            ["2-axis, ±180° primary only", "30 729.5", "−0.01%", "0.9999"],
            ["2-axis, ±180° and ±90°", "30 729.5", "−0.01%", "0.9999"],
            ["2-axis, ±180° and ±60° (sidebar default)", "30 032.5",
             "−2.27%", "0.9773"],
            ["2-axis, ±60° secondary only", "30 048.4", "−2.22%", "0.9778"],
            ["2-axis, ±90° primary and ±60° secondary", "18 471.5",
             "−39.89%", "0.6016"],
            ["1-axis about the same boom, unlimited", "16 654.3", "−45.8%", "0.5419"],
            ["1-axis, ±180° (cable wrap)", "16 624.2", "−45.9%", "0.5409"],
            ["1-axis, ±90° (hard stop)", "2 345.5", "−92.4%", "0.0769"],
            ["Fixed panel on the +Z face", "2 006.7", "−93.5%", "0.0659"],
        ],
        widths=[2.7, 1.2, 1.0, 1.6],
    )
    _add_para(
        doc,
        "The ordering is instructive. A ±90° secondary stop costs almost nothing here "
        "because the elevation command peaks at 89.84° and barely clips it; tightening that "
        "same stop to ±60° costs 2.3%. Turning the primary axis from a cable wrap into a "
        "±90° hard stop costs 40%, because the azimuth needs to reach ±180° "
        "and simply cannot. The one-axis rows show the geometric ceiling of section 5.2: with the "
        "Sun this close to the boom axis, |ŝ·â₁| averages 0.72 and no "
        "one-axis drive can average better than 0.54 on the cosine. The ±90° one-axis "
        "case collapses further still, because the rest normal faces nadir and the ideal command "
        "sits near ±180° — the exact opposite end of a ±90° travel range.",
    )
    _add_para(
        doc,
        "The slew rate matters much less than the travel limits, because only the brief azimuth "
        "flips are rate limited at all:",
    )
    _add_table(
        doc,
        ["Max slew rate", "0.2 °/s", "0.5 °/s", "1.0 °/s (default)",
         "2.0 °/s", "10 °/s"],
        [["Energy (Wh/day)", "29 103.9", "29 938.8", "30 032.5", "30 046.7", "30 048.4"]],
        widths=[1.5, 1.0, 1.0, 1.2, 1.0, 0.8],
    )
    _add_para(
        doc,
        "Going from 1 °/s to effectively infinite rate recovers only 16 Wh of the 699 Wh the "
        "default limits cost; the rest is the ±60° elevation stop, which no amount of "
        "slew rate can fix. Dropping to 0.2 °/s, on the other hand, costs a further 929 Wh, "
        "because each azimuth flip then takes minutes. Note how much smaller the top-end "
        "recovery is than it looks in a model that starts from rest: with no start-up slew to "
        "pay for, a faster drive has only the flips left to improve.",
    )
    _add_para(
        doc,
        "For completeness, the default mission as a whole generates 30 032.5 Wh against 19 206.7 "
        "Wh of load, shunting 7088.3 Wh and reaching a minimum state of charge of 42.1%. That is a "
        "positive power budget, though it fails the configured 30% depth-of-discharge limit — "
        "a battery sizing question rather than a pointing one.",
    )

    # ---------------------------------------------------------------- 10
    _add_heading(doc, "10. Assumptions, edge cases and limitations", 1)
    _add_bullets(
        doc,
        [
            "Kinematic, not dynamic — travel limits and a maximum rate are modelled; "
            "torque, inertia, controller lag, flexible-mode ringing and pointing jitter are not.",
            "Per-axis clamping — when a two-axis gimbal saturates on one axis, the other "
            "axis keeps following its unconstrained command rather than re-solving for the best "
            "reachable normal.",
            "Rewind scheduling — a cable-wrap rewind starts the moment continued tracking "
            "would cross a limit, with no attempt to defer it into an eclipse where it would be "
            "free.",
            "Already-tracking initial condition — limited drives start on their command "
            "rather than at 0°, so no run opens with a start-up slew. Acquisition and "
            "deployment sequences, and the energy they cost, are outside the model.",
            "No self-shadowing or structural blockage — a panel is lit whenever the "
            "spacecraft is lit and the cosine is positive; the bus, antennas and the opposite "
            "wing never cast shadows.",
            "No articulation feedback — the gimbal motion does not disturb the attitude, and "
            "the attitude solution never reacts to where the arrays are pointing.",
            "Sun-pointing attitude has a free phase — the rotation about the commanded "
            "Sun-pointing axis is resolved with a fixed ECI +Z hint vector. With the default "
            "panel normal this is harmless (the gimbals sit at rest), but for an off-axis panel "
            "with tight limits that arbitrary phase does influence the gimbal angles.",
            "Degenerate inputs are handled, not rejected — if the Sun lies parallel to a "
            "one-axis panel's rotation axis the projection vanishes and the code falls back to "
            "the rest normal; if a rotation axis is entered as the zero vector it is replaced by "
            "+Y body.",
            "Ephemeris accuracy — the low-precision solar series is good to roughly 0.01° "
            "for dates within a few decades of J2000, far finer than any pointing effect "
            "discussed here.",
            "Orbit model — secular J2 only. Over a single-day default run the node moves "
            "about one degree; over month-long runs the beta angle history, and hence the gimbal "
            "duty cycle, follows that precession.",
        ],
    )

    # ---------------------------------------------------------------- 11
    _add_heading(doc, "11. Reproducing these results", 1)
    _add_para(
        doc,
        "Every number in section 9 comes from the package itself. The following script reproduces "
        "the geometry series for the default configuration with the sidebar's default limits:",
    )
    _add_code(
        doc,
        "import numpy as np\n"
        "from energy_balance.attitude import body_to_eci\n"
        "from energy_balance.config import SimulationConfig, SolarArrayConfig\n"
        "from energy_balance.orbit import OrbitState, propagate_orbit\n"
        "from energy_balance.solar_array import panel_frames_eci\n"
        "from energy_balance.sun import sun_vector_eci\n"
        "\n"
        "base = SimulationConfig()                       # the opening preset\n"
        "panels = [p.model_copy(update={\"gimbal_min_deg\": -180.0, \"gimbal_max_deg\": 180.0,\n"
        "                               \"gimbal2_min_deg\": -60.0, \"gimbal2_max_deg\": 60.0})\n"
        "          for p in base.solar_array.panels]     # the sidebar defaults\n"
        "cfg = base.model_copy(update={\"solar_array\": SolarArrayConfig(panels=panels)})\n"
        "\n"
        "times = np.arange(0.0, 86400.0 + 1e-9, cfg.mission.time_step_s)\n"
        "state = OrbitState(cfg.orbit, cfg.mission.epoch)\n"
        "orbit = propagate_orbit(state, times)\n"
        "sun   = sun_vector_eci(cfg.mission.epoch, times)\n"
        "R     = body_to_eci(cfg.attitude, orbit[\"r_eci_km\"], orbit[\"v_eci_km_s\"],\n"
        "                    sun[\"u_sun_eci\"])\n"
        "\n"
        "frames = panel_frames_eci(cfg.solar_array.panels[0], R, sun[\"u_sun_eci\"], times)\n"
        "print(frames[\"angle1_deg\"][:8], frames[\"angle2_deg\"][:8], "
        "frames[\"cos_incidence\"][:8])",
    )
    _add_para(
        doc,
        "For the energy figures, run the whole simulation with run_simulation(cfg) and integrate "
        "the gen_w column over the time step. The per-panel geometry series are attached to the "
        "returned DataFrame as attributes (panel_angle1_deg, panel_angle2_deg, "
        "panel_cos_incidence, panel_rewinding, and the matching target series).",
    )

    return doc


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        figures = build_figures(Path(sys.argv[1]).resolve())
        doc = build_document(figures)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            figures = build_figures(Path(tmp))
            doc = build_document(figures)
    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
