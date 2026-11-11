"""Package verified UAV geometry, bind Blender reports to copied assets and write a manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from validate_blender_assets import run as validate_blender


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bundle_info(texture):
    appearance = read(texture / "metrics.json")
    surface = Path(appearance["surface_dir"])
    surface_metrics = read(surface / "metrics.json")
    dense = Path(surface_metrics["dense_dir"])
    depth = read(dense / "metrics.json")
    sparse = Path(depth["source_sparse"])
    cameras = read(sparse / "metrics.json")
    if any(result["status"] != "success" for result in (appearance, surface_metrics, depth, cameras)):
        raise ValueError(f"Expected a successful reconstruction chain: {texture}")
    if not appearance["texture"]["textured_area_fraction"] > 0:
        raise ValueError(f"Photographic texture is required: {texture}")
    if depth["registered_images"] < 3 or depth["registered_images"] != cameras["registered_images"]:
        raise ValueError(f"Camera counts disagree: {texture}")
    files = {name: texture / name for name in ("textured.glb", "textured.obj", "textured.mtl", "atlas.jpg", "metrics.json")}
    files.update({"mesh.ply": surface / "mesh.ply", "dense.ply": dense / "dense.ply",
                  "poses.json": dense / "poses.json", "intrinsics.json": dense / "intrinsics.json"})
    for path in files.values():
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Required release asset is missing or empty: {path}")
    for name in ("poses.json", "intrinsics.json"):
        if any(read(directory / name) != read(dense / name) for directory in (sparse, surface)):
            raise ValueError(f"The reconstruction stages disagree on {name}: {texture}")
    return files, {"source_texture_directory": str(texture.resolve()),
                   "source_sparse_directory": str(sparse.resolve()), "source_dense_directory": str(dense.resolve()),
                   "source_input_images": cameras["input_images"], "registered_images": depth["registered_images"],
                   "profile": depth["profile"], "dense_points": depth["dense_points"],
                   "min_effective_disparity_px": depth.get("min_effective_disparity_px"),
                   "conditioned_pair_selection": depth.get("prefer_conditioned_pairs", False),
                   "alternate_pairs_attempted": len((depth.get("depth_recovery") or {}).get("attempts", [])),
                   "sparse_projection": cameras["after_final_ba"],
                   "texture_area_fraction": appearance["texture"]["textured_area_fraction"],
                   "visibility_refinement": appearance.get("visibility_refinement"),
                   "surface_geometry": surface_metrics["geometry"]}


def build(output, scene_specs, blender, resolution, reference):
    output = output.resolve()
    scenes = {}
    for name, source, revision in scene_specs:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or name in scenes or name == "validation":
            raise ValueError(f"Invalid or duplicate release scene name: {name}")
        if not re.fullmatch(r"[a-f0-9]{7,40}", revision):
            raise ValueError(f"Expected a Git revision for {name}")
        files, metrics = bundle_info(Path(source))
        scenes[name] = (files, {**metrics, "producer_revision": revision})
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "validating", "scenes": {},
                "limitations": ["Scene coverage is partial. Meshes are open and may self-intersect.",
                                "Monocular scale and DSM vertical datum are unverified.",
                                "Blender compatibility does not establish Unity compatibility."]}
    manifest_path = output / "manifest.json"

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save()
    try:
        for name, (files, metrics) in scenes.items():
            destination = output / name
            destination.mkdir()
            assets = {}
            for filename, source in files.items():
                copied = destination / filename
                shutil.copyfile(source, copied)
                digest = sha256(copied)
                if digest != sha256(source):
                    raise ValueError(f"Copy differs from source: {source}")
                assets[filename] = {"path": copied.relative_to(output).as_posix(), "sha256": digest,
                                    "bytes": copied.stat().st_size}
            manifest["scenes"][name] = {**metrics, "assets": assets}
        save()
        if not validate_blender(blender, {name: output / name for name in scenes},
                                output / "validation", reference, resolution, timeout=300):
            raise ValueError("A Blender export or render regression gate failed")
        for name, entry in manifest["scenes"].items():
            entry["blender"] = {}
            for kind in ("glb", "obj"):
                path = output / "validation" / name / kind / "report.json"
                report = read(path)
                asset = entry["assets"][f"textured.{kind}"]
                if (report["status"] != "passed" or Path(report["asset"]).resolve() != output / asset["path"]
                        or report["asset_sha256"] != asset["sha256"]
                        or report["atlas_sha256"] != entry["assets"]["atlas.jpg"]["sha256"]):
                    raise ValueError(f"Blender report does not attest the copied export: {path}")
                entry["blender"][kind] = {"report": path.relative_to(output).as_posix(),
                                          "triangles": report["triangles"],
                                          "intersection_probes": report["bounded_intersection_probes"],
                                          "renders": [(path.parent / f"{view}.png").relative_to(output).as_posix()
                                                      for view in ("top", "oblique")]}
        manifest["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", nargs=3, required=True, metavar=("NAME", "TEXTURE_DIR", "GIT_REV"))
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if not args.blender.is_file() or not 64 <= args.resolution <= 2048:
        parser.error("Provide an existing Blender executable and a render resolution between 64 and 2048")
    build(args.output, args.scene, args.blender, args.resolution, args.reference)
