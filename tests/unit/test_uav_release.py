import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("stale_report", [False, True])
def test_release_manifest_binds_blender_validation_to_copied_files(tmp_path, monkeypatch, stale_report):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import build_uav_release as release

    source = tmp_path / "source"
    source.mkdir()
    (source / "textured.glb").write_bytes(b"glb")
    (source / "textured.obj").write_bytes(b"obj")
    (source / "atlas.jpg").write_bytes(b"atlas")
    files = {path.name: path for path in source.iterdir()}
    monkeypatch.setattr(release, "bundle_info", lambda _: (files, {"registered_images": 3}))

    def blender(_exe, scenes, output, *_args, **_kwargs):
        for name, bundle in scenes.items():
            for kind in ("glb", "obj"):
                path = output / name / kind
                path.mkdir(parents=True)
                asset = bundle / f"textured.{kind}"
                report = {"status": "passed", "asset": str(asset), "triangles": 100,
                          "bounded_intersection_probes": [], "atlas_sha256": release.sha256(bundle / "atlas.jpg"),
                          "asset_sha256": "invalid" if stale_report else release.sha256(asset)}
                (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return True

    monkeypatch.setattr(release, "validate_blender", blender)
    output = tmp_path / "release"
    args = (output, [("test-scene", source, "abcdef0")], Path("blender"), 640, None)
    if stale_report:
        with pytest.raises(ValueError, match="does not attest"):
            release.build(*args)
    else:
        release.build(*args)
    report = release.read(output / "manifest.json")
    assert report["status"] == ("failed" if stale_report else "passed")
    assert not (output / "manifest.tmp").exists()
    assert report["scenes"]["test-scene"]["assets"]["atlas.jpg"]["sha256"] == release.sha256(source / "atlas.jpg")
