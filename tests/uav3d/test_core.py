import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from uav3d import ingest, odm, plan

SRC = Path(__file__).resolve().parents[2] / "src" / "uav3d"
CFG = {
    "ram_budget_fraction": 0.75,
    "pc_quality_by_ram_gb": [[12, "high"], [6, "medium"], [0, "low"]],
    "split": {"megapixel_images_per_ram_gb": 900, "overlap_m": 150},
    "resolution": {"orthophoto_gsd_factor": 1.0, "dem_gsd_factor": 2.0},
    "odm_flags": ["--dsm"],
}


def test_quality_follows_budget():
    assert plan.pc_quality(12, CFG) == "high"
    assert plan.pc_quality(8, CFG) == "medium"
    assert plan.pc_quality(2, CFG) == "low"


def test_split_only_when_load_exceeds_budget():
    small = plan.odm_plan({"count": 172, "mean_megapixels": 12.0}, 12, CFG)
    assert small["split"] is None and "--split" not in small["flags"]
    large = plan.odm_plan({"count": 7169, "mean_megapixels": 20.0}, 12, CFG)
    assert large["split"]["images_per_submodel"] == 540
    assert large["flags"][large["flags"].index("--split") + 1] == "540"


def test_resolution_scales_with_gsd_and_is_omitted_without_metric_gsd():
    assert plan.resolution_flags(0.025, CFG) == ["--orthophoto-resolution", "2.50", "--dem-resolution", "5.00"]
    assert plan.resolution_flags(None, CFG) == []


def test_gsd_from_reconstruction(tmp_path):
    rec = [{
        "cameras": {"c": {"projection_type": "brown", "focal_x": 0.8, "width": 4000, "height": 3000}},
        "shots": {f"s{i}": {"camera": "c", "rotation": [0, 0, 0], "translation": [0, 0, -100.0]} for i in range(3)},
        "points": {str(i): {"coordinates": [i, 0, 0.0]} for i in range(5)},
    }]
    path = tmp_path / "reconstruction.json"
    path.write_text(json.dumps(rec))
    assert odm.gsd_from_reconstruction(path)["gsd_m"] == pytest.approx(100 / (0.8 * 4000), abs=1e-4)


def test_ingest_finds_images_subfolder_and_reports_missing_gps(tmp_path):
    (tmp_path / "images").mkdir()
    for i in range(2):
        Image.new("RGB", (40, 30)).save(tmp_path / "images" / f"a{i}.JPG")
    files = ingest.find_images(tmp_path)
    s = ingest.summarise(files)
    assert s["count"] == 2 and s["gps_fraction"] == 0 and s["sizes"] == {"40x30": 2}
    with pytest.raises(FileNotFoundError):
        ingest.find_images(tmp_path / "images" / "none")


def test_prepare_links_without_touching_inputs(tmp_path):
    src = tmp_path / "in.jpg"
    Image.new("RGB", (4, 4)).save(src)
    project = odm.prepare([src], tmp_path / "work")
    assert (project / "images" / "in.jpg").read_bytes() == src.read_bytes()
    odm.prepare([src], tmp_path / "work")  # idempotent


def test_uav3d_does_not_import_legacy_or_reference_code():
    """ARCHITECTURE.md: no vision3d imports; only uav3d.evaluate may read reference data."""
    for path in SRC.rglob("*.py"):
        if "evaluate" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert not name.startswith("vision3d"), f"{path.name} imports legacy {name}"
                assert not name.startswith("uav3d.evaluate"), f"{path.name} imports reference code"
