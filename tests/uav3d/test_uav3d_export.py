import json

import numpy as np
import trimesh

from uav3d import export

# A 10 x 10 x 5 m box as one CityJSON feature (scale 0.01, translate 1000/2000/0), LoD2.2 solid
# with roof/wall/ground semantics, parent building "b1".
BOX_VERTS = [[0, 0, 0], [1000, 0, 0], [1000, 1000, 0], [0, 1000, 0],
             [0, 0, 500], [1000, 0, 500], [1000, 1000, 500], [0, 1000, 500]]
FACES = [[[0, 3, 2, 1]], [[4, 5, 6, 7]], [[0, 1, 5, 4]], [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]]


def write_seq(path, status="ok"):
    header = {"type": "CityJSON", "version": "2.0", "CityObjects": {}, "vertices": [],
              "transform": {"scale": [0.01, 0.01, 0.01], "translate": [1000.0, 2000.0, 0.0]}}
    geom = {"type": "Solid", "lod": "2.2", "boundaries": [FACES],
            "semantics": {"surfaces": [{"type": "GroundSurface"}, {"type": "RoofSurface"}, {"type": "WallSurface"}],
                          "values": [[0, 1, 2, 2, 2, 2]]}}
    feature = {"type": "CityJSONFeature", "id": "b1", "vertices": BOX_VERTS, "CityObjects": {
        "b1": {"type": "Building", "attributes": {}, "children": ["b1-0"]},
        "b1-0": {"type": "BuildingPart", "parents": ["b1"], "geometry": [geom]}}}
    path.write_text(json.dumps(header) + "\n" + json.dumps(feature) + "\n" + json.dumps(feature).replace('"b1', '"b2') + "\n")
    return path


def test_triangulate_planar_polygon_with_hole():
    outer = np.array([[0, 0, 1], [4, 0, 1], [4, 4, 1], [0, 4, 1]], float)
    hole = np.array([[1, 1, 1], [1, 3, 1], [3, 3, 1], [3, 1, 1]], float)
    tri = export._triangulate([outer, hole])
    pts = np.vstack([outer, hole])
    area = sum(0.5 * np.linalg.norm(np.cross(pts[b] - pts[a], pts[c] - pts[a])) for a, b, c in tri)
    assert area == np.float64(12.0)  # 16 - 4


def test_building_meshes_are_closed_boxes_relative_to_origin(tmp_path):
    seq = write_seq(tmp_path / "t.city.jsonl")
    per = {"b1": {"status": "ok"}, "b2": {"status": "too_low"}}
    meshes = export.building_meshes([seq], per, origin=np.array([1000.0, 2000.0, 0.0]))
    assert list(meshes) == ["b1"]  # too_low buildings are not exported
    m = trimesh.Trimesh(meshes["b1"].vertices, meshes["b1"].faces)  # merge duplicated corners
    assert m.is_watertight and abs(m.volume) == np.float64(500.0)
    assert np.allclose(m.bounds, [[0, 0, 0], [10, 10, 5]])


def test_merge_cityjson_shifts_vertex_indices(tmp_path):
    seq = write_seq(tmp_path / "t.city.jsonl")
    n = export.merge_cityjson([seq], tmp_path / "all.city.json")
    model = json.loads((tmp_path / "all.city.json").read_text())
    assert n == 2 and len(model["vertices"]) == 16
    second = model["CityObjects"]["b2-0"]["geometry"][0]["boundaries"][0][0][0]
    assert min(second) >= 8  # indices of the second feature point into its own vertices


def test_scene_glb_is_y_up(tmp_path):
    box = trimesh.creation.box(extents=[2, 4, 6])
    box.apply_translation([0, 0, 3])  # z-up: sits on the ground, 6 m tall
    out = tmp_path / "s.glb"
    export.write_scene(None, {"b": box}, out)
    loaded = trimesh.load(out).dump(concatenate=True)
    assert np.allclose(loaded.extents, [2, 6, 4]) and loaded.bounds[0][1] == np.float64(0.0)
