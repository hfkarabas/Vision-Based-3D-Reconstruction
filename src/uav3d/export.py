"""Combine terrain and buildings into viewable scenes.

- Terrain: ODM's dem2mesh on the DTM (``-rtc``: vertices relative to the raster centre, so
  float32 PLY keeps cm precision), textured with the orthophoto.
- Buildings: Roofer CityJSONSeq, LoD2.2 (else LoD1.2) solids triangulated with earcut;
  roof and wall colours from CityJSON semantics; low-confidence buildings tinted.
- Outputs: scene.glb (glTF 2.0, Y-up, local origin), buildings.city.json (CityJSON 2.0,
  georeferenced), scene.json (CRS + origin to georeference the glb), optional scene.blend.
"""
from __future__ import annotations

import json
from pathlib import Path

import mapbox_earcut
import numpy as np
import rasterio
import trimesh
from PIL import Image
from rasterio.enums import Resampling

from uav3d.record import run_measured

ROOF, WALL, GROUND = [178, 74, 58, 255], [222, 216, 205, 255], [120, 120, 120, 255]
LOW_CONFIDENCE_TINT = np.array([255, 170, 0, 255])
EXPORTED_STATUSES = {"ok", "low_confidence"}
# glTF is Y-up, right-handed; map projected (east, north, up) -> (x, y, z) = (east, up, -north)
Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


class ExportError(RuntimeError):
    pass


def raster_center(path) -> np.ndarray:
    with rasterio.open(path) as ds:
        b = ds.bounds
        return np.array([(b.left + b.right) / 2, (b.bottom + b.top) / 2, 0.0])


def terrain_mesh(odm, dtm, ortho, work: Path, max_vertices: int, max_texture_px: int, log: Path,
                 n_threads: int | None = None) -> tuple:
    """(textured mesh in coordinates relative to the DTM centre, run info)."""
    ply = work / "terrain.ply"
    cmd = [str(odm.exe("dem2mesh")), "-inputFile", str(dtm), "-outputFile", str(ply),
           "-maxVertexCount", str(max_vertices), "-rtc"]
    if n_threads:
        cmd += ["-maxConcurrency", str(n_threads)]
    info = run_measured(cmd, log, env=odm.env())
    if info["returncode"] != 0 or not ply.is_file():
        raise ExportError(f"dem2mesh failed (exit {info['returncode']}); see {log}")
    mesh = trimesh.load(ply, process=False)
    center = raster_center(dtm)
    with rasterio.open(ortho) as ds:
        scale = max(1.0, max(ds.width, ds.height) / max_texture_px)
        shape = (max(1, int(ds.height / scale)), max(1, int(ds.width / scale)))
        bands = [1, 2, 3, 4] if ds.count >= 4 else [1, 2, 3]
        rgba = ds.read(bands, out_shape=(len(bands), *shape), resampling=Resampling.average)
        b = ds.bounds
    image = Image.fromarray(np.moveaxis(rgba, 0, -1).astype(np.uint8))
    xy = mesh.vertices[:, :2] + center[:2]
    uv = np.column_stack([(xy[:, 0] - b.left) / (b.right - b.left), (xy[:, 1] - b.bottom) / (b.top - b.bottom)])
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image.convert("RGB"),
                                                   metallicFactor=0.0, roughnessFactor=1.0)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.clip(uv, 0, 1), material=material)
    info.update(vertices=len(mesh.vertices), faces=len(mesh.faces), texture_px=list(image.size),
                center=center.tolist())
    return mesh, info


def _triangulate(ring_points: list[np.ndarray]) -> np.ndarray | None:
    """Earcut a planar 3D polygon (outer ring + holes) in its own plane; returns (k, 3) indices
    into the concatenated rings, or None for degenerate faces."""
    pts = np.vstack(ring_points)
    if len(pts) < 3:
        return None
    normal = np.zeros(3)
    for ring in ring_points[:1]:  # Newell's method on the outer ring
        nxt = np.roll(ring, -1, axis=0)
        normal += np.cross(ring, nxt).sum(axis=0)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal /= norm
    u = np.cross(normal, [0, 0, 1] if abs(normal[2]) < 0.9 else [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    local = pts - pts.mean(axis=0)
    flat = np.column_stack([local @ u, local @ v])
    ends = np.cumsum([len(r) for r in ring_points]).astype(np.uint32)
    tri = mapbox_earcut.triangulate_float64(flat, ends)
    return tri.reshape(-1, 3) if len(tri) else None


def _semantic_types(geom: dict) -> list[str | None]:
    sem = geom.get("semantics")
    if not sem:
        return []
    values = sem.get("values") or []
    if geom["type"] == "Solid":
        values = values[0] if values else []
    return [sem["surfaces"][i]["type"] if i is not None else None for i in values]


def building_meshes(jsonl_files: list[Path], per_building: dict, origin: np.ndarray) -> dict[str, trimesh.Trimesh]:
    """One mesh per exported building (LoD2.2, else LoD1.2), in coordinates relative to origin."""
    parts: dict[str, list[trimesh.Trimesh]] = {}
    for path in jsonl_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        scale = np.array(header["transform"]["scale"])
        translate = np.array(header["transform"]["translate"])
        for line in lines[1:]:
            feature = json.loads(line)
            verts = np.asarray(feature["vertices"], dtype=float) * scale + translate - origin
            for co in feature["CityObjects"].values():
                parent = (co.get("parents") or [None])[0]
                bid = parent if co["type"] == "BuildingPart" else None
                if bid is None or per_building.get(bid, {}).get("status") not in EXPORTED_STATUSES:
                    continue
                geoms = {g["lod"]: g for g in co.get("geometry", [])}
                geom = geoms.get("2.2") or geoms.get("1.2")
                if not geom:
                    continue
                surfaces = geom["boundaries"][0] if geom["type"] == "Solid" else geom["boundaries"]
                kinds = _semantic_types(geom)
                v_all, f_all, c_all = [], [], []
                for i, surface in enumerate(surfaces):
                    rings = [verts[np.asarray(r)] for r in surface]
                    tri = _triangulate(rings)
                    if tri is None:
                        continue
                    kind = kinds[i] if i < len(kinds) else None
                    colour = ROOF if kind == "RoofSurface" else GROUND if kind == "GroundSurface" else WALL
                    f_all.append(tri + sum(len(v) for v in v_all))
                    v_all.append(np.vstack(rings))
                    c_all.append(np.tile(colour, (len(tri), 1)))
                if not v_all:
                    continue
                mesh = trimesh.Trimesh(vertices=np.vstack(v_all), faces=np.vstack(f_all), process=False)
                colours = np.vstack(c_all).astype(float)
                if per_building[bid]["status"] == "low_confidence":
                    colours = 0.5 * colours + 0.5 * LOW_CONFIDENCE_TINT
                mesh.visual.face_colors = colours.astype(np.uint8)
                parts.setdefault(bid, []).append(mesh)
    return {bid: trimesh.util.concatenate(ms) for bid, ms in parts.items()}


def merge_cityjson(jsonl_files: list[Path], out: Path) -> int:
    """Concatenate CityJSONSeq features into one CityJSON 2.0 file (same transform)."""
    model, count = None, 0
    for path in jsonl_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        if model is None:
            model = {**header, "CityObjects": {}, "vertices": []}
        elif header["transform"] != model["transform"]:
            raise ExportError("CityJSONSeq tiles use different transforms; cannot merge.")
        for line in lines[1:]:
            f = json.loads(line)
            offset = len(model["vertices"])
            model["vertices"].extend(f["vertices"])
            for key, co in f["CityObjects"].items():
                for g in co.get("geometry", []):
                    g["boundaries"] = _shift(g["boundaries"], offset)
                model["CityObjects"][key] = co
                count += co.get("type") == "Building"
    out.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    return count


def _shift(boundaries, offset: int):
    if isinstance(boundaries, int):
        return boundaries + offset
    return [_shift(b, offset) for b in boundaries]


def write_scene(terrain: trimesh.Trimesh | None, buildings: dict[str, trimesh.Trimesh], out: Path) -> dict:
    scene = trimesh.Scene()
    if terrain is not None:
        scene.add_geometry(terrain, node_name="terrain", geom_name="terrain", transform=Z_UP_TO_Y_UP)
    for bid, mesh in sorted(buildings.items()):
        scene.add_geometry(mesh, node_name=f"building_{bid}", geom_name=f"building_{bid}", transform=Z_UP_TO_Y_UP)
    out.write_bytes(scene.export(file_type="glb"))
    return {"nodes": len(scene.graph.nodes_geometry), "bytes": out.stat().st_size}


BLENDER_SCRIPT = """
import bpy, math, sys
glb, blend, png = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
from mathutils import Vector
xs = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
cx = sum(p.x for p in xs) / len(xs); cy = sum(p.y for p in xs) / len(xs); cz = sum(p.z for p in xs) / len(xs)
span = max(max(p.x for p in xs) - min(p.x for p in xs), max(p.y for p in xs) - min(p.y for p in xs))
cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
bpy.context.scene.collection.objects.link(cam)
cam.location = (cx - 0.55 * span, cy - 0.75 * span, cz + 0.6 * span)
cam.rotation_euler = (math.radians(55), 0, math.radians(-35))
cam.data.clip_end = span * 10
bpy.context.scene.camera = cam
sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN'))
sun.rotation_euler = (math.radians(40), 0, math.radians(30))
bpy.context.scene.collection.objects.link(sun)
s = bpy.context.scene
s.render.engine = 'BLENDER_WORKBENCH'
s.display.shading.light = 'STUDIO'
s.display.shading.color_type = 'TEXTURE'
s.render.resolution_x, s.render.resolution_y = 1600, 1000
s.render.filepath = png
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=blend)
"""


def blender_scene(blender: Path, glb: Path, out_dir: Path, log: Path) -> dict:
    """Open the glb in Blender headless; save scene.blend and a preview render."""
    script = out_dir / "_blender_import.py"
    script.write_text(BLENDER_SCRIPT, encoding="utf-8")
    blend, png = out_dir / "scene.blend", out_dir / "preview.png"
    cmd = [str(blender), "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script),
           "--", str(glb), str(blend), str(png)]
    info = run_measured(cmd, log)
    script.unlink(missing_ok=True)
    if info["returncode"] != 0 or not blend.is_file():
        raise ExportError(f"Blender export failed (exit {info['returncode']}); see {log}")
    info.update(blend=str(blend), preview=str(png) if png.is_file() else None)
    return info


def export(odm, products: dict, buildings_summary: dict | None, cfg: dict, out_dir: Path, log_dir: Path,
           blender: Path | None, n_threads: int | None = None) -> dict:
    ecfg = cfg["export"]
    out_dir.mkdir(parents=True, exist_ok=True)
    terrain, tinfo = terrain_mesh(odm, products["dtm"], products["orthophoto"], out_dir,
                                  ecfg["terrain_max_vertices"], ecfg["texture_max_px"], log_dir / "dem2mesh.log",
                                  n_threads)
    origin = np.array(tinfo["center"])
    result = {"terrain": tinfo}
    meshes = {}
    if buildings_summary and buildings_summary.get("status") == "ok":
        files = [Path(f) for f in buildings_summary["files"]]
        meshes = building_meshes(files, buildings_summary["per_building"], origin)
        result["cityjson"] = str(out_dir / "buildings.city.json")
        result["cityjson_buildings"] = merge_cityjson(files, out_dir / "buildings.city.json")
    result["buildings_in_scene"] = len(meshes)
    glb = out_dir / "scene.glb"
    result["glb"] = {"path": str(glb), **write_scene(terrain, meshes, glb)}
    with rasterio.open(products["dtm"]) as ds:
        epsg = ds.crs.to_epsg() if ds.crs else None
    georef = {"crs": f"EPSG:{epsg}" if epsg else None, "origin_projected": origin.tolist(),
              "axes": "glTF Y-up: x = east, y = up, z = -north (metres, relative to origin_projected)"}
    (out_dir / "scene.json").write_text(json.dumps(georef, indent=1), encoding="utf-8")
    result["georeference"] = georef
    if blender and ecfg["blender"]:
        result["blender"] = blender_scene(blender, glb, out_dir, log_dir / "blender.log")
    (out_dir / "terrain.ply").unlink(missing_ok=True)
    return result
