"""Six SolarCat flow attitudes + 1° AoA growth. STL is read as data, not copied."""
from pathlib import Path

import pytest

from arlamx_v2.validate_attitudes import (
    SOLAR_STL,
    all_within_budget,
    run_solarcat,
    write_outputs,
)

OUT = Path(__file__).resolve().parents[2] / "outputs" / "results" / "geometry"


@pytest.mark.skipif(not SOLAR_STL.exists(), reason="SolarCat STL not on disk")
def test_solarcat_1pct_six_attitudes():
    res = run_solarcat("1pct")
    write_outputs(res, OUT)
    assert res["wetted_rel"] <= 0.01, f"wetted {res['wetted_rel']:.3%}"
    failed = [r for r in res["rows"] if r["rel_err"] > 0.01]
    assert not failed, failed
    assert all_within_budget(res)
    names = {r["name"] for r in res["rows"]}
    assert names == {
        "1_side_flat",
        "2_side_vertex",
        "3_max_drag_pm_z",
        "4_tilt_x_45",
        "5_tilt_y_45",
        "6_aoa_0_to_1deg_growth",
    }


@pytest.mark.skipif(not SOLAR_STL.exists(), reason="SolarCat STL not on disk")
def test_solarcat_crude_10pct_is_looser():
    res = run_solarcat("10pct")
    assert res["tol"] == 0.10
    # Crude may use fewer panels than 1%.
    res1 = run_solarcat("1pct")
    assert res["report"].n_panels <= res1["report"].n_panels + 5
