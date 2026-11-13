"""Desktop authorization and provenance checks for native UAV scenes."""

import hashlib
import json

import pytest

from vision3d.demo.uav import scene_command


def result(tmp_path):
    run = tmp_path / "current run"
    scene = run / "scene"
    scene.mkdir(parents=True)
    source = tmp_path / "original photos"
    source.mkdir()
    blend = scene / "terrain_white_buildings.blend"
    blend.write_bytes(b"test BLEND")
    blender = tmp_path / "Blender Foundation" / "blender.exe"
    blender.parent.mkdir()
    blender.touch()
    record = dict(status="verified", blend=str(blend), blender=str(blender), input=str(source),
                  blend_sha256=hashlib.sha256(blend.read_bytes()).hexdigest())
    return run, source, record


def publish(run, record):
    (run / "run.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("status", ["verified", "partial"])
def test_only_a_verified_current_run_is_openable(tmp_path, status):
    run, photos, record = result(tmp_path)
    publish(run, dict(record, status=status))
    assert scene_command(run, photos) == [record["blender"], record["blend"]]
    with pytest.raises(ValueError, match="current photo folder"):
        scene_command(run, tmp_path / "new folder")
    publish(run, dict(record, status="failed"))
    with pytest.raises(ValueError, match="current photo folder"):
        scene_command(run, photos)


def test_old_or_modified_blend_is_rejected(tmp_path):
    run, photos, record = result(tmp_path)
    publish(run, dict(record, blend=str(tmp_path / "old.blend")))
    with pytest.raises(ValueError, match="outside the current run"):
        scene_command(run, photos)
    publish(run, dict(record, blend_sha256="0" * 64))
    with pytest.raises(ValueError, match="changed after Blender verification"):
        scene_command(run, photos)
