"""Build the MPC / heuristic PDF from the campaign summary. Level: advanced."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs" / ".old" / "advisors" / "summary.json"
SOC_PNG = ROOT / "outputs" / ".old" / "advisors" / "campaign_soc_gen.png"
SOC_TS = ROOT / "outputs" / ".old" / "advisors" / "soc_timeseries.png"
OFF_PNG = ROOT / "outputs" / ".old" / "advisors" / "off_nadir_budget.png"
DECAY_PNG = ROOT / "outputs" / ".old" / "analysis" / "decay" / "min_max_drag_decay.png"
OUT = ROOT / "ACEnv" / "Reports" / "2026-08-18_detailed_mpc_heuristic.pdf"


def styles():
    base = getSampleStyleSheet()
    out = {
        "title": ParagraphStyle(
            "T",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Times-Bold", fontSize=13, spaceBefore=12, spaceAfter=6
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Times-Bold", fontSize=11, spaceBefore=9, spaceAfter=4
        ),
        "body": ParagraphStyle(
            "B",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12.5,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "C", parent=base["BodyText"], fontName="Times-Roman", fontSize=7.5, leading=9.5, alignment=TA_LEFT
        ),
        "head": ParagraphStyle(
            "CH", parent=base["BodyText"], fontName="Times-Bold", fontSize=7.5, leading=9.5
        ),
        "foot": ParagraphStyle(
            "F", parent=base["Normal"], fontName="Times-Italic", fontSize=8, textColor=colors.HexColor("#444444")
        ),
        "eq": ParagraphStyle(
            "E", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.5, leading=13, alignment=1, spaceAfter=8
        ),
    }
    return out


def P(text, st):
    return Paragraph(text, st)


def table(rows, widths):
    grid = Table(rows, colWidths=widths, repeatRows=1)
    grid.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#666666")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2EC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return grid


def fmt(x, nd=3):
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not (v == v):
        return "--"
    return f"{v:.{nd}f}"


def load_summary():
    if not SUMMARY.exists():
        return []
    return json.loads(SUMMARY.read_text())


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8)
    canvas.drawString(0.75 * inch, 0.5 * inch, "ARLAMX V2.0  ·  AstroChimera  ·  grok  ·  2026-08-18")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"{doc.page}")
    canvas.restoreState()


def build():
    st = styles()
    story = []
    story.append(P("Sampling MPC and heuristic advisors on ARLAMX V2.0", st["title"]))
    story.append(
        P(
            "Single-advisor AstroChimera session. The plant is the V2.0 C++ library "
            "(GGM03S degree 4, Sentman M-01, panel SRP, co-rotating MSIS 2.1). "
            "The spacecraft has no sun sensor. Heuristics close only on generated power. "
            "The MPC may use a sun ephemeris (clock + orbit), which is not a sun sensor.",
            st["body"],
        )
    )
    story.append(
        P(
            "<b>Geometry for every number:</b> SolarCat 1 % sealed-side mesh, 0.625 kg, "
            "i = 23°, e = 0.001, a = R<sub>E</sub>+500 km, F10.7 = 150, Ap = 4.",
            st["body"],
        )
    )

    story.append(P("1. What the advisors adjust", st["h1"]))
    story.append(
        P(
            "Both families emit one scalar-first unit quaternion q<sub>BN</sub> (q<sub>0</sub> &ge; 0) "
            "every 300 s advisor step. The plant tracks it with MRP-PD at 2 s. That quaternion "
            "is the only command. Body +Z is simultaneously the sail normal, the two-sided "
            "panel axis, and the antenna boresight. There is no array gimbal.",
            st["body"],
        )
    )

    story.append(P("2. Heuristic family", st["h1"]))
    story.append(P("2.1 SoC mode ladder", st["h2"]))
    story.append(
        P(
            "Mode selection uses only supercapacitor state of charge (Wertz 1978). "
            "Default thresholds t = (0.60, 0.50, 0.40, 0.30, 0.20).",
            st["body"],
        )
    )
    hc = st["cell"]
    hh = st["head"]
    story.append(
        table(
            [
                [P("<b>SoC</b>", hh), P("<b>Command</b>", hh)],
                [P("&ge; t0", hc), P("+Z at a visible station, else min-drag / nadir / last-good sun", hc)],
                [P("t1 … t0", hc), P("station if visible, else last-good sun, else search", hc)],
                [P("&lt; t1", hc), P("power search; no station", hc)],
            ],
            [1.3 * inch, 5.4 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        P(
            "Search steps grow as SoC falls (x1.5 below t2, x2 below t3). Below t4 the "
            "search latches on any sensed power.",
            st["body"],
        )
    )

    story.append(P("2.2 Power search without a sun vector", st["h2"]))
    story.append(
        P(
            "Sweep about body Y then X until <font face='Courier'>power_gen_norm</font> &ge; sense_floor. "
            "Then coordinate-descent hill-climb on ±X, ±Y (Nocedal &amp; Wright 2006): keep a step "
            "if generation rose by more than 10<sup>-3</sup>, else revert; halve the step after a "
            "failed cycle. Hold when generation &ge; gen_target or the step is &lt; 1 deg. "
            "A fruitless 360° sweep on both axes is treated as eclipse: freeze for "
            "eclipse_hold_steps (default 6 = 30 min). Body-frame updates are "
            "q' = q<sub>d</sub> x q<sub>BN</sub> (Hamilton product, left multiplication, passive convention).",
            st["body"],
        )
    )

    story.append(P("3. Sampling MPC", st["h1"]))
    story.append(
        P(
            "The attitude–power–drag map is non-convex and Sentman FMF has no cheap derivative. "
            "A receding-horizon sampler is used (Rawlings, Mayne &amp; Diehl 2017; "
            "Camacho &amp; Bordons 2007).",
            st["body"],
        )
    )
    story.append(P("3.1 Candidates and the 40° clip", st["h2"]))
    story.append(
        P(
            "Each step proposes: hold, sun-track from the analytic ephemeris (optional), "
            "ground-station track if elevation &ge; 10 deg, min-drag, nadir, and K = 8 random "
            "axis–angle perturbations. Every candidate is SLERP-clipped to 40° from the incumbent:",
            st["body"],
        )
    )
    story.append(
        P(
            "q(&tau;) = [sin((1-&tau;)&Omega;/2) q<sub>a</sub> + sin(&tau; &Omega;/2) q<sub>b</sub>] / sin(&Omega;/2), "
            "&Omega; = 2 arccos |&lt;q<sub>a</sub>, q<sub>b</sub>&gt;|, &tau; = min(1, 40&deg;/&Omega;).",
            st["eq"],
        )
    )
    story.append(P("3.2 Surrogate (H = 6 steps = 30 min)", st["h2"]))
    story.append(
        P(
            "The candidate is held inertially, matching the plant’s zero-order hold. "
            "Orbit: two-body RK4, 60 s, mu = 3.986004418x10<sup>14</sup> m<sup>3</sup>/s<sup>2</sup> "
            "(Vallado 2013 §1). J<sub>2</sub>/drag/SRP are omitted — they do not change ranking "
            "over 30 min. Eclipse: cylindrical shadow. Power: two-sided cosine "
            "P<sub>gen</sub> = 0.78 x 0.85 x |z_N · s| x 1[lit]. "
            "Loads: 7 mW always, 205 mW GPS when lit, 400 mW TX if z · g &gt; 0.7 and lit. "
            "Supercap 0.53 Wh. Drag: Sentman coefficients_only on the real 1 % panels.",
            st["body"],
        )
    )
    story.append(P("3.3 Objective", st["h2"]))
    story.append(
        P(
            "J(q) = sum<sub>k</sub> [ B(SoC<sub>k</sub>) + 1[pass,lit] (T(&theta;<sub>gs</sub>) + w<sub>align</sub> cos &theta;<sub>gs</sub>) "
            "- w<sub>cd</sub> C<sub>d,k</sub> ] - w<sub>slew</sub> &Delta;<sub>cmd</sub>/40&deg;.",
            st["eq"],
        )
    )
    story.append(
        P(
            "B(s) = -4(0.4-s)/0.4 if s &lt; 0.4; +1 if 0.4 &le; s &le; 0.6; exp(-6(s-0.6)) if s &gt; 0.6. "
            "T is the 2&deg;/5&deg;/10&deg; tier ladder (full / half / quarter / -0.5). "
            "Weights: power 1, low-SoC 4, GS 2, GS penalty 0.5, align 1, Cd 0.5, slew 0.2.",
            st["body"],
        )
    )
    story.append(
        P(
            "The physical plant still runs GGM03S + Sentman + SRP + MSIS. The surrogate lives "
            "only inside the planner. Variant <font face='Courier'>mpc_no_sun_eph</font> drops "
            "the ephemeris sun-track candidate so the planner has the same solar information "
            "as the heuristic.",
            st["body"],
        )
    )

    story.append(P("4. Eclipse and ground-pointing limits", st["h1"]))
    story.append(
        P(
            "Two-sided ±Z panels make nadir a <b>power-positive</b> communications attitude: "
            "at local noon r is nearly sunward, so the anti-sun face still illuminates and "
            "|z · s| ~ 1. The 2-day run measured mean sunlit generation 0.81 of peak at nadir. "
            "Tilting 0°/15°/30°/45°/60° toward velocity keeps min SoC at 0.528/0.523/0.517/0.511/0.504 "
            "and generation at 0.81/0.79/0.77/0.74/0.71 — a 60° ground-looking tilt is still "
            "eclipse-stable. One eclipse (~35 min) at the GPS+sensing load drops about 0.23 SoC. "
            "Heuristics that latch a dim false sun (sense_floor 0.02, or hold-on-any-power) "
            "brown out; floors 0.05–0.20 with a real climb target survive. "
            "Stations for i = 23°: Honolulu, Mexico City, Mumbai, Singapore, Quito, Darwin.",
            st["body"],
        )
    )

    story.append(P("5. Lifetime context (min / max drag)", st["h1"]))
    story.append(
        P(
            "Prescribed-attitude decay on the same 1 % plant, F10.7=150, Ap=4, corotating + SRP: "
            "min-drag reaches 250 km in <b>18.81 days</b>, max-drag in <b>10.85 days</b>. "
            "Start |r|-R<sub>E</sub> = 493.12 km is perigee of e = 0.001, not a first-step drop.",
            st["body"],
        )
    )
    if DECAY_PNG.exists():
        img = Image(str(DECAY_PNG), width=6.4 * inch, height=3.5 * inch)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(P("Figure 1. Min- and max-drag altitude. Oscillation is the eccentric + J2 radius, growing as drag pumps e.", st["foot"]))

    story.append(PageBreak())
    story.append(P("6. Two-day advisor campaign (1 % SolarCat)", st["h1"]))
    rows_json = load_summary()
    if rows_json:
        head = [
            P("<b>Policy</b>", hh),
            P("<b>min SoC</b>", hh),
            P("<b>mean SoC</b>", hh),
            P("<b>brown</b>", hh),
            P("<b>gen (lit)</b>", hh),
            P("<b>GS lock</b>", hh),
            P("<b>nadir 30°</b>", hh),
            P("<b>dh km</b>", hh),
            P("<b>ok</b>", hh),
        ]
        body = [head]
        for rec in rows_json:
            body.append(
                [
                    P(str(rec.get("name", "")), hc),
                    P(fmt(rec.get("soc_min"), 3), hc),
                    P(fmt(rec.get("soc_mean"), 3), hc),
                    P(str(int(rec.get("brownouts", 0))), hc),
                    P(fmt(rec.get("gen_mean_lit"), 3), hc),
                    P(fmt(rec.get("gs_lock_frac"), 3), hc),
                    P(fmt(rec.get("nadir_30_frac"), 3), hc),
                    P(fmt(rec.get("dalt_km"), 2), hc),
                    P("yes" if rec.get("survived") else "no", hc),
                ]
            )
        story.append(
            table(
                body,
                [
                    1.55 * inch,
                    0.65 * inch,
                    0.7 * inch,
                    0.5 * inch,
                    0.7 * inch,
                    0.6 * inch,
                    0.7 * inch,
                    0.55 * inch,
                    0.4 * inch,
                ],
            )
        )
        story.append(Spacer(1, 6))
        story.append(
            P(
                "Table 1. Two-day closed loop, 300 s advisor, 2 s MRP-PD. "
                "GS lock is the fraction of steps with a visible station and z · g &gt; 0.7. "
                "Nadir 30&deg; is the fraction of steps with +Z within 30&deg; of -r. "
                "dh is spherical-altitude change over the two days (more negative = more drag).",
                st["foot"],
            )
        )
    else:
        story.append(P("Campaign summary was not on disk when this PDF was built.", st["body"]))

    if SOC_PNG.exists():
        img = Image(str(SOC_PNG), width=6.4 * inch, height=2.9 * inch)
        img.hAlign = "CENTER"
        story.append(Spacer(1, 8))
        story.append(img)
        story.append(P("Figure 2. Minimum SoC and mean sunlit generation by policy.", st["foot"]))
    if SOC_TS.exists():
        img = Image(str(SOC_TS), width=6.4 * inch, height=3.4 * inch)
        img.hAlign = "CENTER"
        story.append(Spacer(1, 8))
        story.append(img)
        story.append(P("Figure 3. SoC time history. conserve and the 0.02-floor search go to zero; nadir / ground_first stay near 0.53.", st["foot"]))
    if OFF_PNG.exists():
        img = Image(str(OFF_PNG), width=5.6 * inch, height=3.3 * inch)
        img.hAlign = "CENTER"
        story.append(Spacer(1, 8))
        story.append(img)
        story.append(P("Figure 4. Off-nadir tilt toward velocity. Generation and min SoC stay above 0.50 out to 60°.", st["foot"]))

    story.append(P("6.1 How to read the table", st["h2"]))
    story.append(
        P(
            "A policy that <b>survives eclipse</b> has min SoC well above 0 and zero brownouts. "
            "On this bus that means entering umbra above ~0.35 SoC and generating at least 0.4 of peak "
            "when lit. <b>Ground pointing</b> shows up as a high nadir-30° fraction and/or a "
            "non-zero GS lock; both cost generation because +Z is then off the sun. "
            "The search-floor rows isolate how bright the panels must be before the heuristic "
            "believes it has found the sun — without ever reading a sun vector.",
            st["body"],
        )
    )

    story.append(P("7. Citations", st["h1"]))
    cites = [
        "Rawlings, J. B., Mayne, D. Q. &amp; Diehl, M. (2017). Model Predictive Control: Theory, Computation, and Design, 2nd ed.",
        "Camacho, E. F. &amp; Bordons, C. (2007). Model Predictive Control, 2nd ed.",
        "Wertz, J. R. (ed.) (1978). Spacecraft Attitude Determination and Control.",
        "Nocedal, J. &amp; Wright, S. (2006). Numerical Optimization, 2nd ed.",
        "Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications, 4th ed.",
        "Sentman, L. H. (1961). Free Molecule Flow Theory and its Application to the Determination of Aerodynamic Forces. LMSC-448514.",
        "Moe, K. &amp; Moe, M. M. (2005). Gas–surface interactions and satellite drag coefficients. Planet. Space Sci. 53(8), 793–801. (paywall — review by hand)",
        "Schaub, H. &amp; Junkins, J. L. (2018). Analytical Mechanics of Space Systems, 4th ed.",
        "Montenbruck, O. &amp; Gill, E. (2000). Satellite Orbits. Springer.",
        "Picone, J. M. et al. (2002). NRLMSISE-00. JGR Space Physics 107(A12).",
    ]
    for c in cites:
        story.append(P(c, st["body"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.7 * inch,
        title="ARLAMX V2.0 MPC and heuristic advisors",
        author="Grok / AstroChimera",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
