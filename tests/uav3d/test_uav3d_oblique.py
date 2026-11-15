import json

import numpy as np

from uav3d import odm, plan


def _rec(tmp_path, angles_deg):
    """OpenSfM reconstruction whose shots look down and tilt about x by the given off-nadir angles."""
    shots = {}
    for i, a in enumerate(angles_deg):
        # world->camera rotation: pi about x looks straight down; tilting by a lowers it to pi - a
        shots[f"s{i}"] = {"rotation": [np.pi - np.radians(a), 0.0, 0.0], "translation": [0, 0, 0], "camera": "c"}
    p = tmp_path / "reconstruction.json"
    p.write_text(json.dumps([{"shots": shots, "cameras": {}, "points": {}}]), encoding="utf-8")
    return p


def test_view_angles_nadir_and_oblique(tmp_path):
    nadir = odm.view_angles(_rec(tmp_path, [0, 2, 3, 1]))
    assert nadir["median_off_nadir_deg"] < 3 and nadir["share_oblique"] == 0.0
    tilted = odm.view_angles(_rec(tmp_path, [45, 60, 30, 5]))
    assert abs(tilted["median_off_nadir_deg"] - 37.5) < 0.5 and tilted["share_oblique"] == 0.75
    ocfg = plan.defaults()["oblique"]
    assert odm.is_oblique(tilted, ocfg) and not odm.is_oblique(nadir, ocfg)
    # oblique capture is only flagged (out of scope); no ODM flags are changed
    assert "odm_flags" not in ocfg and "out of scope" in odm.OBLIQUE_WARNING
