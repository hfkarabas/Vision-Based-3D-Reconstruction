from uav3d import refine


def test_effective_size_follows_openmvs_min_resolution():
    assert refine.effective_size(7952, 4, 640) == (3, 994)   # ODM "medium" on 42 MP UseGeo images
    assert refine.effective_size(7952, 2, 640) == (2, 1988)
    assert refine.effective_size(4000, 1, 640) == (1, 2000)
    assert refine.effective_size(500, 2, 640) == (0, 500)


def test_dmap_size_reads_header(tmp_path):
    import struct
    p = tmp_path / "d.dmap"
    p.write_bytes(b"DR\x07\x00" + struct.pack("<IIII", 7952, 5304, 994, 663) + b"\x00" * 8)
    assert refine.dmap_size(p) == (994, 663)


def test_grid_covers_box_with_square_tiles():
    tiles = refine.grid(0, 0, 250, 100, 100)
    assert len(tiles) == 3 * 1
    assert [t.id for t in tiles] == [0, 1, 2]
    assert (tiles[-1].x0, tiles[-1].x1, tiles[-1].y1) == (200, 300, 100)


def test_groups_are_4_connected_components():
    tiles = {(t.ix, t.iy): t for t in refine.grid(0, 0, 400, 400, 100)}
    sel = [tiles[c] for c in [(0, 0), (1, 0), (1, 1), (3, 3), (2, 2)]]  # (2,2) touches (1,1) only diagonally
    comps = refine.groups(sel)
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 1, 3]


def test_multipolygon_shifts_by_offset():
    t = refine.grid(0, 0, 10, 10, 10)[0]
    wkt = refine.multipolygon([t], 500000, 4000000)
    assert wkt.startswith("MULTIPOLYGON(((500000.0 4000000.0, 500010.0 4000000.0")


def test_utm_offset_reads_coords_txt(tmp_path):
    (tmp_path / "odm_georeferencing").mkdir()
    (tmp_path / "odm_georeferencing" / "coords.txt").write_text("WGS84 UTM 33N\n498188 4379575\n1 2 3\n")
    assert refine.utm_offset(tmp_path) == (498188.0, 4379575.0)
    assert refine.utm_srs(tmp_path) == "EPSG:32633"


def test_merge_pipeline_replaces_coarse_inside_groups():
    tiles = refine.grid(0, 0, 200, 100, 100)
    p = refine.merge_pipeline("coarse.laz", [("a.ply", [tiles[0]]), ("b.ply", [tiles[1]])], (1000.0, 2000.0),
                              "EPSG:32633", refine.Path("out.laz"))
    assert all(s.get("override_srs", "EPSG:32633") == "EPSG:32633" for s in p)
    keep = next(s for s in p if s.get("tag") == "keep")
    assert keep["type"] == "readers.las" and not any(s.get("outside") for s in p)  # coarse is pre-cropped
    merge = next(s for s in p if s["type"] == "filters.merge")
    assert merge["inputs"] == ["keep", "a0", "a1"]
    shift = next(s for s in p if s.get("tag") == "t0")
    assert shift["matrix"] == "1 0 0 1000.0 0 1 0 2000.0 0 0 1 0 0 0 0 1"
    writer = p[-1]
    assert writer["offset_x"] == 1000.0 and writer["filename"].endswith("out.laz")

def test_merge_pipeline_without_coarse_writes_only_refined():
    t = refine.Tile(0, 0, 0, 0.0, 0.0, 10.0, 10.0)
    st = refine.merge_pipeline(None, [("a.ply", [t])], (100.0, 200.0), "EPSG:32633", "o.laz")
    assert [s["type"] for s in st][0] == "readers.ply"
    assert st[-2]["inputs"] == ["a0"]


def test_drop_tiles_removes_points_inside_chosen_tiles_only(tmp_path):
    import laspy
    import numpy as np

    h = laspy.LasHeader(point_format=3, version="1.2")
    h.offsets, h.scales = [1000, 2000, 0], [0.001] * 3
    las = laspy.LasData(h)
    # reconstruction-frame grid of 10 m tiles anchored at (0, 0); offset (1000, 2000) -> UTM
    xs = np.array([5.0, 15.0, 25.0, 5.0, 15.0]) + 1000
    ys = np.array([5.0, 5.0, 5.0, 15.0, 15.0]) + 2000
    las.x, las.y, las.z = xs, ys, np.zeros(5)
    las.classification = np.array([2, 2, 2, 3, 6], dtype=np.uint8)
    las.write(tmp_path / "c.laz")
    tiles = refine.grid(0, 0, 30, 20, 10)  # ids 0..2 bottom row, 3..5 top row
    r = refine.drop_tiles(tmp_path / "c.laz", [tiles[1], tiles[3]], (1000.0, 2000.0), tmp_path / "o.laz")
    out = laspy.read(tmp_path / "o.laz")
    assert r == {"kept": 3, "dropped": 2}
    assert sorted(zip(np.round(out.x - 1000), np.round(out.y - 2000))) == [(5, 5), (15, 15), (25, 5)]
