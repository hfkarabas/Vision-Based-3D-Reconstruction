"""Build a verified UAV Blender scene from a folder of original photographs."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile
import time
import traceback
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import ExifTags, Image
import psutil
import rasterio as rio
from rasterio.features import geometry_mask
from shapely.geometry import shape
import yaml

from omega_odm_products import observed_products, transfer
from omega_uav_pipeline import sha256
from omega_uav_products import raster
from omega_uav_foundation_trial import FOUNDATION_BUFFER_M
from omega_uav_publish import assemble_scene, built_ids, export_dem, stage_foundations


SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
SETTINGS = ROOT / "configs/omega_reusable_uav.json"
ODM_FLAGS = ("--dsm", "--dtm", "--dem-resolution", "50", "--orthophoto-resolution", "20",
             "--max-concurrency", "4", "--pc-quality", "low", "--feature-quality", "medium",
             "--skip-3dmodel", "--skip-report")
PRODUCTS = {
    "rgb": "odm_orthophoto/odm_orthophoto.tif",
    "dsm": "odm_dem/dsm.tif",
    "dtm": "odm_dem/dtm.tif",
    "cloud": "odm_georeferencing/odm_georeferenced_model.laz",
    "shots": "opensfm/reconstruction.json",
}
ODM_RUNTIME_FILES = (
    "VERSION", "run.py", "settings.yaml", "win32env.bat", "opendm/config.py",
    "stages/odm_georeferencing.py", "stages/odm_dem.py", "stages/odm_orthophoto.py",
    "SuperBuild/install/bin/DensifyPointCloud.exe", "SuperBuild/install/bin/pdal.exe",
    "SuperBuild/install/bin/renderdem.exe", "SuperBuild/install/bin/odm_orthophoto.exe",
)
CONTROL_OPTIONS = {"name", "project_path", "rerun", "rerun_all", "rerun_from"}
MIN_IMAGES = 3


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, data):
    payload = json.dumps(data, indent=2, allow_nan=False)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(payload)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def inventory(folder):
    if not folder.is_dir():
        raise ValueError("Select an existing folder of original UAV photos.")
    photos = sorted(path for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(photos) < MIN_IMAGES:
        raise ValueError(f"At least {MIN_IMAGES} JPEG/PNG photos are required. Found {len(photos)}.")
    records = {}
    positions = set()
    for path in photos:
        try:
            with Image.open(path) as image:
                if image.format not in {"JPEG", "MPO", "PNG"}:
                    raise ValueError(f"Unsupported decoded format {image.format}")
                image.load()
                gps = image.getexif().get_ifd(ExifTags.IFD.GPSInfo)
                try:
                    lat, lon = [sum(float(v) / 60**i for i, v in enumerate(gps[key]))
                                for key in (2, 4)]
                    lat = -lat if gps.get(1) == "S" else lat
                    lon = -lon if gps.get(3) == "W" else lon
                    if (gps.get(1) in {"N", "S"} and gps.get(3) in {"E", "W"}
                            and abs(lat) <= 90 and abs(lon) <= 180):
                        positions.add((lat, lon))
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    pass
        except (OSError, ValueError, SyntaxError) as exc:
            raise ValueError(f"Unreadable image {path.name}: {exc}") from exc
        records[path.name] = sha256(path)
    if len(positions) < MIN_IMAGES:
        raise ValueError("Metric reconstruction requires EXIF GPS from at least three distinct camera positions.")
    return records


def normalized_options(options):
    return {key: value for key, value in options.items()
            if key not in CONTROL_OPTIONS and not key.endswith("_is_set")}


def locate(variable, choices):
    if value := os.environ.get(variable):
        choices = [Path(value).expanduser()]
    for path in choices:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Set {variable} to its installed path. Checked: {', '.join(map(str, choices))}")


def tools():
    omega = ROOT / "outputs/mission-omega"
    legacy = omega / "building-recovery"
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    blender_root = program_files / "Blender Foundation"
    return SimpleNamespace(
        odm=locate("VISION3D_ODM", [Path("C:/ODM"), omega / "tools/odm",
                                 *sorted((omega / "reusable-uav").glob("*/tools/odm"))]),
        pdal=locate("VISION3D_PDAL", [omega / "tools/gis/Library/bin/pdal.exe"]),
        qgis=locate("VISION3D_QGIS_DEM", [omega / "tools/Qgis2threejs-master"]),
        blender=locate("VISION3D_BLENDER", [Path(p) for p in filter(None, [shutil.which("blender")])]
                       + sorted(blender_root.glob("Blender */blender.exe"), reverse=True)),
        model=locate("VISION3D_BUILDING_MODEL", [
            legacy / "terrain-validation/checkpoints/building_footprints_usa.pth",
        ]),
        hub_cache=locate("VISION3D_HUB_CACHE", [legacy / "runtime/hub"]),
        node=locate("VISION3D_NODE", [Path(p) for p in filter(None, [shutil.which("node")])]),
    )


def odm_environment(home, output):
    # Match the paths set by the native ODM win32env.bat. Pass arguments directly
    # to ODM's Python, including workspace paths with shell metacharacters.
    env = os.environ.copy()
    binary = home / "SuperBuild/install/bin"
    python = home / "venv/Scripts/python.exe"
    env.update(
        PATH=os.pathsep.join(map(str, (python.parent, binary, binary / "opensfm/bin", env["PATH"]))),
        GDAL_DATA=str(binary / "data/gdal"), GDAL_DRIVER_PATH=str(binary / "gdalplugins"),
        PDAL_DRIVER_PATH=str(binary), PROJ_LIB=str(binary / "data/proj"),
        PYTHONPYCACHEPREFIX=str(output / "odm-pycache"),
    )
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return python, env


def odm_metadata(home, output):
    settings = yaml.safe_load((home / "settings.yaml").read_text(encoding="utf-8-sig")) or {}
    settings = {key.replace("-", "_"): value for key, value in settings.items()}
    for key in ("sm_cluster", "align", "geo", "gcp", "boundary", "copy_to", "split_image_groups", "cameras"):
        if settings.get(key):
            raise ValueError(f"Clear {key} in ODM settings. This workflow uses only the selected photos locally.")
    python, env = odm_environment(home, output)
    flags = ["--project-path", str(output), "site", *ODM_FLAGS]
    probe = "from opendm.config import config; import json,sys; print(json.dumps(vars(config(sys.argv[1:]))))"
    result = subprocess.check_output([python, "-X", "utf8", "-c", probe, *flags],
                                     cwd=home, env=env, text=True, encoding="utf-8", timeout=60)
    effective = normalized_options(json.loads(result.splitlines()[-1]))
    return dict(version=(home / "VERSION").read_text().strip(),
                files={name: sha256(home / name) for name in ODM_RUNTIME_FILES},
                options=effective)


def odm_artifacts(project):
    return {key: project / relative for key, relative in PRODUCTS.items()}


def product_hashes(project):
    return {key: sha256(path) for key, path in odm_artifacts(project).items()}


def cache_key(inputs, metadata):
    return hashlib.sha256(json.dumps([inputs, metadata], sort_keys=True).encode()).hexdigest()


def cache_project(manifest, inputs, metadata):
    if not manifest.is_file():
        return None
    try:
        record = read(manifest)
        project = Path(record["project"])
        if (record["images"] == inputs and record["odm"] == metadata
                and normalized_options(read(project / "options.json")) == metadata["options"]
                and product_hashes(project) == record["artifacts"]):
            return project
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def register_cache(project, manifest, inputs, metadata, **provenance):
    if normalized_options(read(project / "options.json")) != metadata["options"]:
        raise ValueError("The effective ODM options do not match the current production profile.")
    staged = {path.name: sha256(path) for path in (project / "images").iterdir()
              if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if staged != inputs:
        raise ValueError("The ODM input copies differ from the selected originals.")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    save(manifest, dict(project=str(project), images=inputs, odm=metadata,
                        artifacts=product_hashes(project), **provenance))


def import_odm(project, log, command_file, tool, inputs, metadata, manifest):
    """Register a completed local ODM run using its original command and success log."""
    command = read(command_file)
    if len(command) < 4 or command[1] != "--project-path" or Path(command[0]).resolve() != tool / "run.bat":
        raise ValueError("The prior command did not use the selected ODM installation.")
    if Path(command[2]).resolve() / command[3] != project:
        raise ValueError("The prior command does not identify the supplied ODM project.")
    logged = log.read_text(encoding="utf-8", errors="replace")
    if f"Initializing ODM {metadata['version']} -" not in logged or "ODM app finished" not in logged:
        raise ValueError("The prior ODM log does not verify completion with the installed version.")
    completed = (project / "options.json").stat().st_mtime
    if any((tool / name).stat().st_mtime > completed for name in ODM_RUNTIME_FILES):
        raise ValueError("The installed ODM components are newer than the prior run.")
    register_cache(project, manifest, inputs, metadata, origin="verified local ODM run",
                   producer_log=str(log), producer_command=str(command_file))


def extent_from_products(artifacts, summary):
    with rio.open(artifacts["rgb"]) as rgb, rio.open(artifacts["dsm"]) as dsm, rio.open(artifacts["dtm"]) as dtm:
        crs = rgb.crs
        if (crs is None or not crs.is_projected or crs.linear_units_factor[1] != 1
                or crs != dsm.crs or crs != dtm.crs):
            raise ValueError("ODM did not produce an orthophoto, DSM and DTM in a common metric CRS.")
    if rio.crs.CRS.from_string(summary["srs"]["horizontal"]) != crs:
        raise ValueError("ODM point cloud and rasters have different projected CRS definitions.")
    bounds = summary["bounds"]
    return crs.to_string(), [math.floor(bounds["minx"]), math.floor(bounds["miny"]),
                             math.ceil(bounds["maxx"]), math.ceil(bounds["maxy"])]


def audit_ground(base, final, reports, features, solids):
    """Audit terrain changes near rejected candidates without modifying the terrain."""
    built = {row["building_id"] for row in solids if row["status"] == "built"}
    with rio.open(base / "dtm.tif") as src, rio.open(final / "dtm.tif") as result:
        first = src.read(1, masked=True).filled(np.nan)
        second = result.read(1, masked=True).filled(np.nan)
        changed = (np.isfinite(first) != np.isfinite(second)) | (
            np.isfinite(first) & np.isfinite(second) & (first != second))
        cell_m2 = abs(src.transform.a * src.transform.e)
        buffers = {item["properties"]["building_id"]: shape(item["geometry"]).buffer(FOUNDATION_BUFFER_M)
                   for item in features}
        def coverage(ids):
            return geometry_mask([buffers[key] for key in ids], first.shape, src.transform, invert=True) if ids else (
                np.zeros(first.shape, dtype=bool)
            )
        approved = coverage(built)
        affected = coverage(buffers.keys() - built)
        rejected_only = changed & affected & ~approved
        if (changed & ~approved).any():
            raise ValueError("The published terrain contains changes outside approved foundation buffers.")
        with rio.open(base / "observed.tif") as observed:
            direct = observed.read(1) > 0
        if (changed & direct).any():
            raise ValueError("A directly observed ground cell changed during foundation preparation.")
        valid = np.isfinite(second)
        grid = dict(crs=src.crs, transform=src.transform, width=src.width, height=src.height)
        for name, mask in (("observed", direct & valid), ("inferred", valid & ~direct),
                           ("unsupported", ~valid)):
            raster(reports / f"final_ground_{name}.tif", mask.astype("uint8"), grid)
    return dict(changed_cells=int(changed.sum()), changed_near_rejected_cells=int((changed & affected).sum()),
                changed_near_rejected_m2=float((changed & affected).sum() * cell_m2),
                changed_exclusively_near_rejected_cells=int(rejected_only.sum()),
                observed_cells=int((direct & valid).sum()), inferred_cells=int((valid & ~direct).sum()),
                unsupported_cells=int((~valid).sum()), cell_area_m2=cell_m2,
                note="Candidate buffers can overlap. This measures association, including shared buffers.")


def detect_roofs(run, local, ortho, crs):
    """Add WHU proposals that are substantially outside the Mask R-CNN detections."""
    detections = run.output / "detections"
    rgb = ortho / "orthophoto.tif"
    run.command([sys.executable, SCRIPTS / "omega_building_review.py", "alternate", "--rgb",
                 rgb, "--checkpoint", local.model, "--output", detections])
    primary_info = read(detections / "inference.json")
    primary = (gpd.read_file(detections / "geoai_polygons.gpkg").to_crs(crs) if primary_info["polygons"]
               else gpd.GeoDataFrame(geometry=[], crs=crs))

    settings = read(SETTINGS)["segmentation"]
    whu = detections / "whu"
    run.command([sys.executable, SCRIPTS / "omega_building_inference.py", "--rgb", rgb,
                 "--hub-cache", local.hub_cache, "--model", settings["repo"],
                 "--threshold", settings["threshold"], "--minimum-area", settings["minimum_area_m2"],
                 "--revision", settings["revision"], "--checkpoint-sha256", settings["sha256"], "--output", whu])
    whu_info = read(whu / "inference.json")
    secondary = (gpd.read_file(whu / "geoai_polygons.gpkg").to_crs(crs) if whu_info["vector_polygons"]
                 else gpd.GeoDataFrame(geometry=[], crs=crs))
    combined = append_roofs(primary.assign(detector="mask-rcnn"), secondary.assign(detector="whu"))
    report = dict(primary=primary_info, secondary=whu_info, whu_max_fraction_overlapping_primary=0.20,
                  whu_added=len(combined) - len(primary))
    if home := os.environ.get("VISION3D_UAGLNET_HOME"):
        root = Path(home).expanduser().resolve()
        deps = root / "deps"
        if not deps.is_dir():
            raise FileNotFoundError(f"UAGLNet dependencies are missing at {deps}.")
        env = os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join(filter(None, [str(deps), env.get("PYTHONPATH")])),
                   HF_HUB_OFFLINE="1", ALBUMENTATIONS_NO_TELEMETRY="1", NO_ALBUMENTATIONS_UPDATE="1")
        extra = detections / "uaglnet"
        run.command([sys.executable, SCRIPTS / "omega_uaglnet_inference.py", rgb, extra, "--home", root,
                     "--threshold", settings["threshold"], "--minimum-area", settings["minimum_area_m2"]], env=env)
        info = read(extra / "inference.json")
        count = len(combined)
        if info["polygons"]:
            proposals = gpd.read_file(extra / "geoai_polygons.gpkg").to_crs(crs)
            combined = append_roofs(combined, proposals.assign(detector="uaglnet"))
        report.update(uaglnet=info, uaglnet_added=len(combined) - count)
    run.record["inference"] = dict(report, candidates=len(combined))
    return combined


def append_roofs(primary, secondary):
    """Exclude proposals substantially overlapping a current footprint in either direction."""
    overlap_limit = 0.20
    overlap = secondary.geometry.intersection(primary.geometry.union_all()).area
    supplemental = secondary.loc[overlap <= overlap_limit * secondary.geometry.area]
    supplemental = supplemental.loc[supplemental.geometry.map(
        lambda polygon: not (primary.geometry.intersection(polygon).area >
                             overlap_limit * primary.geometry.area).any()
    ).astype(bool)]
    return gpd.GeoDataFrame(pd.concat([primary, supplemental], ignore_index=True), crs=primary.crs)



class Run:
    def __init__(self, output, photos):
        self.output = output
        self.photos = photos
        self.record = dict(status="running", input=str(photos), stages=[])
        with (output / "run.json").open("x", encoding="utf-8") as marker:
            json.dump(self.record, marker)
        self.set_stage("input", "Validate original photos and local tools")

    def set_stage(self, key, label):
        self.log = self.output / f"{key}.log"
        self.record.update(stage=label, log=str(self.log))
        self.record["stages"].append(dict(stage=key, label=label, log=str(self.log)))
        self.log.touch()
        self.publish()
        print(f"Stage: {label}", flush=True)

    def publish(self):
        save(self.output / "run.json", self.record)

    def command(self, command, **options):
        command = list(map(str, command))
        started = time.monotonic()
        with self.log.open("a", encoding="utf-8") as log:
            print(json.dumps(command), file=log, flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=3600, **options)
        self.record["stages"][-1].setdefault("commands", []).append(
            dict(argv=command, seconds=time.monotonic() - started, exit=result.returncode))
        self.publish()
        if result.returncode:
            with self.log.open("rb") as stream:
                stream.seek(max(0, self.log.stat().st_size - 16384))
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
            meaningful = [line.strip() for line in lines if "[ERROR]" in line or
                          line.startswith(("Error", "RuntimeError:", "ValueError:", "FileNotFoundError:"))]
            error = meaningful[-1] if meaningful else f"Process exited with status {result.returncode}."
            raise RuntimeError(error)

    def measured(self, output, name, command):
        self.command(command)


def pipeline(run, cache):
    output = run.output
    inputs = inventory(run.photos)
    local = tools()
    metadata = odm_metadata(local.odm, output)
    manifest = cache / (cache_key(inputs, metadata) + ".json")
    project = cache_project(manifest, inputs, metadata)
    available = psutil.virtual_memory().available
    free = shutil.disk_usage(output).free
    needed = max(5 * 1024**3, 4 * sum((run.photos / name).stat().st_size for name in inputs))
    if not project and (available < 2 * 1024**3 or free < needed):
        raise RuntimeError(f"ODM needs 2 GiB available RAM and {needed / 1024**3:.1f} GiB free disk "
                           "for this profile. Close other jobs or free disk space.")
    versions = dict(odm=metadata, geoai=version("geoai-py"), rasterio=version("rasterio"),
                    gdal_rasterio=rio.__gdal_version__, blender=sha256(local.blender),
                    building_checkpoint=sha256(local.model),
                    qgis_module=sha256(local.qgis / "web/src/layer/demlayer.ts"))
    run.record.update(images=inputs, tool_paths={key: str(path) for key, path in vars(local).items()},
                      provenance=versions, available_ram_bytes=available, free_disk_bytes=free)
    run.set_stage("odm", "OpenDroneMap · reuse verified products" if project else
                  "OpenDroneMap · reconstruct photos, orthophoto, DSM and DTM")
    if project is None:
        project = output / "odm/site"
        staged = project / "images"
        staged.mkdir(parents=True)
        for name, digest in inputs.items():
            target = staged / name
            shutil.copy2(run.photos / name, target)
            if sha256(target) != digest:
                raise OSError(f"Photo changed during staging: {name}")
        python, env = odm_environment(local.odm, output)
        run.command([python, "-X", "utf8", local.odm / "run.py", "--project-path",
                     project.parent, project.name, *ODM_FLAGS], env=env, cwd=local.odm)
        register_cache(project, manifest, inputs, metadata, origin="desktop alpha")
        reused = False
    else:
        reused = True
        run.log.write_text(f"Verified ODM products reused from {project}\n", encoding="utf-8")
    assets = odm_artifacts(project)
    run.record.update(odm_project=str(project), odm_reused=reused,
                      registered=len({name for item in read(assets["shots"]) for name in item["shots"]}))

    run.set_stage("rasters", "PDAL and GDAL · align imagery and terrain")
    summary_path = output / "cloud-summary.json"
    with summary_path.open("wb") as handle:
        subprocess.run([local.pdal, "info", "--summary", assets["cloud"]], stdout=handle,
                       check=True, timeout=60)
    crs, extent = extent_from_products(assets, read(summary_path)["summary"])
    run.record["provenance"]["pdal"] = read(summary_path)["pdal_version"]
    run.record.update(crs=crs, extent=extent, surface_settings=read(SETTINGS)["surface"])
    common = dict(settings=SETTINGS, extent=extent, crs=crs)
    observed = output / "observations"
    products = output / "products"
    observed_products(SimpleNamespace(**common, output=observed, cloud=assets["cloud"], pdal=local.pdal))
    transfer(SimpleNamespace(**common, output=products, rgb=assets["rgb"], dsm=assets["dsm"],
                             dtm=assets["dtm"], observed=observed / "classified_ground.tif"))
    ortho, ground = products / "orthophoto", products / "ground"
    ground_info = read(ground / "complete.json")
    if not ground_info["observed_cells"]:
        raise ValueError("ODM yielded no directly supported ground cells for the selected site.")

    run.set_stage("geoai", "GeoAI · detect roofs in the camera orthophoto")
    candidates = detect_roofs(run, local, ortho, crs)
    final = output / "scene"
    final.mkdir()
    footprints = final / "building_footprints_projected.json"
    shutil.copy2(ortho / "supported_photo.png", final / "terrain_photo.png")
    save(final / "grid.json", dict(crs=crs, bounds=extent, origin_enu=ground_info["origin"],
                                  texture_label=ground_info["texture_label"]))

    blend = final / "terrain_white_buildings.blend"
    blender_args = [local.blender, "-b", "-t", "4", "--python-exit-code", "1", "--python"]
    run.set_stage("foundations", "GDAL and Blender · validate temporary foundation surfaces")
    prepared, mesh, features, decisions = stage_foundations(
        run, local, blender_args, ground, ortho, observed / "dense_count.tif", candidates, final,
    )
    save(footprints, dict(type="FeatureCollection", features=features))
    if mesh is None:
        mesh = output / "mesh"
        export_dem(run, local, prepared, mesh)
    run.set_stage("blender", "Blender · recheck contact and export the approved scene")
    solids = assemble_scene(run, blender_args, final, mesh)
    if built_ids(solids) != built_ids(decisions):
        raise ValueError("Final mesh contact disagrees with the validated foundation trial.")
    reports = final / "reports"
    solids_path = reports / "blender_buildings.json"
    save(reports / "foundation_trial_decisions.json", decisions)
    save(reports / "ground_changes.json", audit_ground(ground, prepared, reports, features, solids))
    if features:
        run.command([sys.executable, SCRIPTS / "omega_building_review.py", "foundations", "--rgb",
                     ortho / "orthophoto.tif", "--projected-assessment", footprints, "--validation",
                     solids_path, "--output", final / "building_detection_overlay.png"])

    run.set_stage("verify", "Blender · reopen and verify this run’s native scene")
    reopened = reports / "blender_native_reopen.json"
    run.command([*blender_args, SCRIPTS / "omega_building_reopen.py", "--", "--scene", blend,
                 "--validation", reports / "blender_glb_reimport.json", "--render",
                 final / "preview/reopened.png", "--report", reopened, "--require-collections"])
    evidence = read(reopened)
    if Path(evidence["saved_scene"]) != blend:
        raise ValueError("Blender validated a different scene path.")
    buildings = evidence["buildings"]
    run.record.update(status="verified" if buildings else "partial", blend=str(blend),
                      blend_sha256=sha256(blend), blender=str(local.blender), accepted_buildings=buildings,
                      blender_version=evidence["blender"], mesh_provenance=read(mesh / "grid.json"),
                      published_dtm=str(prepared / "dtm.tif"))
    if not buildings:
        run.record["partial_reason"] = ("GeoAI returned no roof candidates." if not features else
                                        "No detected footprint passed the raster and Blender checks.")
    run.publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/demo-uav/odm-cache")
    parser.add_argument("--import-odm", type=Path)
    parser.add_argument("--producer-log", type=Path)
    parser.add_argument("--producer-command", type=Path)
    args = parser.parse_args()
    import_args = (args.import_odm, args.producer_log, args.producer_command)
    if any(import_args) and not all(import_args):
        parser.error("Importing ODM requires the project, producer log and producer command JSON.")
    output, photos = args.output.resolve(), args.images.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        parser.error("Choose an unused run directory.")
    os.environ.update(HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    run = Run(output, photos)
    try:
        if args.import_odm:
            inputs, local = inventory(photos), tools()
            metadata = odm_metadata(local.odm, output)
            manifest = args.cache.resolve() / (cache_key(inputs, metadata) + ".json")
            import_odm(args.import_odm.resolve(), args.producer_log.resolve(), args.producer_command.resolve(),
                       local.odm, inputs, metadata, manifest)
            run.record.update(status="cache_imported", cache_manifest=str(manifest))
            run.publish()
        else:
            pipeline(run, args.cache.resolve())
    except Exception as exc:
        with run.log.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        run.record.update(status="failed", error=str(exc))
        run.publish()
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
