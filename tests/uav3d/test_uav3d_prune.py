from uav3d import odm


def test_prune_keeps_products_and_report_and_deletes_intermediates(tmp_path):
    big = b"x" * (2 << 20)
    files = {
        "odm_georeferencing/odm_georeferenced_model.laz": big, "odm_dem/dsm.tif": big, "odm_dem/dtm.tif": big,
        "odm_orthophoto/odm_orthophoto.tif": big, "opensfm/reconstruction.json": big,
        "opensfm/stats/overlap.tif": big, "odm_report/report.pdf": big, "images/a.jpg": big,
        "opensfm/undistorted/images/a.tif": big, "opensfm/undistorted/openmvs/depthmaps/d.dmap": big,
        "odm_texturing/model.obj": big, "opensfm/features/a.npz": big,
        "odm_meshing/log.txt": b"small",
    }
    for rel, data in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    r = odm.prune(tmp_path)
    left = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert left == {"odm_georeferencing/odm_georeferenced_model.laz", "odm_dem/dsm.tif", "odm_dem/dtm.tif",
                    "odm_orthophoto/odm_orthophoto.tif", "opensfm/reconstruction.json", "opensfm/stats/overlap.tif",
                    "odm_report/report.pdf", "images/a.jpg", "odm_meshing/log.txt"}
    assert r["missing_products_after_prune"] == [] and r["freed_gb"] > 0
    assert not (tmp_path / "opensfm" / "undistorted").exists()
