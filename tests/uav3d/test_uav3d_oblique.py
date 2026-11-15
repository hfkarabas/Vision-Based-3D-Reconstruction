import json

import numpy as np
import trimesh

from uav3d import export, odm, plan


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
    assert "--use-3dmesh" in ocfg["odm_flags"]


def test_prune_keeps_the_textured_model(tmp_path):
    tex = tmp_path / "odm_texturing"
    tex.mkdir()
    for name in ("odm_textured_model_geo.obj", "odm_textured_model_geo.mtl",
                 "odm_textured_model_geo_material0000_map_Kd.png", "odm_textured_model_geo_unaligned.obj"):
        (tex / name).write_bytes(b"x" * (2 << 20))
    odm.prune(tmp_path)
    left = sorted(p.name for p in tex.iterdir())
    assert "odm_textured_model_geo.obj" in left and "odm_textured_model_geo_material0000_map_Kd.png" in left


def test_textured_model_is_shifted_to_the_scene_origin(tmp_path):
    project = tmp_path / "project"
    (project / "odm_georeferencing").mkdir(parents=True)
    (project / "odm_georeferencing" / "coords.txt").write_text("WGS84 UTM 33N\n500000 4000000\n", encoding="utf-8")
    (project / "odm_texturing").mkdir()
    obj = project / "odm_texturing" / "odm_textured_model_geo.obj"
    trimesh.creation.box(extents=[2, 2, 2]).export(obj)
    parts = export.textured_model(obj, origin=np.array([500010.0, 4000020.0, 0.0]))
    assert len(parts) == 1
    assert np.allclose(parts[0][1].bounds.mean(axis=0), [-10, -20, 0])
    out = tmp_path / "s.glb"
    info = export.write_scene(parts, {}, out)
    assert info["nodes"] == 1
