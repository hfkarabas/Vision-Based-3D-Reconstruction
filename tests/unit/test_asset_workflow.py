import json
from pathlib import Path

import pytest

from vision3d.reconstruction import asset_workflow


def test_manifest_retries_transient_replace_failure(tmp_path, monkeypatch):
    original = Path.replace
    failures = []

    def replace(path, target):
        if not failures:
            failures.append(target)
            raise PermissionError("temporarily locked")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(asset_workflow, "sleep", lambda _: None)
    result = tmp_path / "workflow.json"
    asset_workflow._save_manifest(result, {"status": "success"})
    assert failures == [result] and json.loads(result.read_text()) == {"status": "success"}
    assert not result.with_suffix(".tmp").exists()


def test_asset_workflow_keeps_successful_sparse_output_on_dense_failure(tmp_path, monkeypatch):
    sparse = tmp_path / "saved"
    sparse.mkdir()
    (sparse / "metrics.json").write_text('{"status": "success", "registered_images": 5}')
    output = tmp_path / "run"

    def failed_mesh(*args, **kwargs):
        return {"status": "success", "mesh": {"status": "rejected", "reason": "no continuous depth"}}

    monkeypatch.setattr(asset_workflow, "reconstruct_multiview", failed_mesh)
    with pytest.raises(ValueError, match="no continuous depth"):
        asset_workflow.reconstruct_asset(output, sparse_dir=sparse)
    result = json.loads((output / "workflow.json").read_text())
    assert result["status"] == "failed" and result["failed_stage"] == "dense"
    assert result["stages"]["sparse"]["status"] == "success"
    assert result["assets"] == {} and (sparse / "metrics.json").is_file()
    with pytest.raises(FileExistsError):
        asset_workflow.reconstruct_asset(output, sparse_dir=sparse)


def test_asset_workflow_records_only_verified_outputs(tmp_path, monkeypatch):
    def sparse(input_path, output, **kwargs):
        assert input_path == "photos" and kwargs["auto_intrinsics"]
        output.mkdir()
        return {"status": "success", "registered_images": 5, "input_images": 7}

    def dense(_sparse, output, **kwargs):
        assert kwargs["profile"] == "uav-recovery" and kwargs["mesh"]
        output.mkdir()
        (output / "dense.ply").write_bytes(b"point data")
        return {"mesh": {"status": "success"}, "dense_points": 800}

    def surface(_dense, output, **kwargs):
        output.mkdir()
        (output / "mesh.ply").write_bytes(b"mesh data")

    def texture(_surface, output, **kwargs):
        assert kwargs["obj"]
        output.mkdir()
        for name in ("textured.glb", "textured.obj", "textured.mtl", "atlas.jpg"):
            (output / name).write_bytes(b"texture data")
        return {"texture": {"textured_area_fraction": .45}}

    for name, replacement in (("reconstruct_sequence", sparse), ("reconstruct_multiview", dense),
                              ("build_surface_asset", surface), ("build_texture_asset", texture)):
        monkeypatch.setattr(asset_workflow, name, replacement)
    run = tmp_path / "run"
    report = asset_workflow.reconstruct_asset(run, input_path="photos", profile="uav-recovery", obj=True)
    assert report == json.loads((run / "workflow.json").read_text())
    assert report["status"] == "success" and report["registered_images"] == 5
    assert report["photographic_area_fraction"] == .45 and len(report["assets"]) == 6
