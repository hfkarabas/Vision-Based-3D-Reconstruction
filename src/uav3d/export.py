"""Combine terrain and buildings into viewable scenes.

- Terrain: ODM's dem2mesh on the DTM (``-rtc``: vertices relative to the raster centre, so
  float32 PLY keeps cm precision), textured with the orthophoto.
- Buildings: Roofer CityJSONSeq, LoD2.2 (else LoD1.2; export.building_lod) solids triangulated with
  earcut; one material per building, flagged buildings (tower above the roof or low-confidence roof
  fit, see flagged) in export.flag_colour_rgba.
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

BUILDING = [215, 215, 215, 255]
FLAG = [230, 0, 200, 255]  # magenta: apart from terrain greens/greys and the grey buildings
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


def textured_model(obj_path, origin: np.ndarray) -> list[tuple[str, trimesh.Trimesh]]:
    """ODM's textured 3D model (odm_texturing/odm_textured_model_geo.obj, one texture atlas per
    material) shifted to coordinates relative to origin. The OBJ is in ODM's frame: UTM minus the
    coords.txt offset, z up."""
    from uav3d.refine import utm_offset

    project = Path(obj_path).parents[1]
    ox, oy = utm_offset(project)
    loaded = trimesh.load(obj_path, force="scene", process=False)
    shift = np.array([ox, oy, 0.0]) - origin
    parts = []
    for i, (name, geom) in enumerate(sorted(loaded.geometry.items())):
        m = geom.copy()
        m.apply_translation(shift)
        parts.append((f"mesh_{i:03d}", m))
    return parts


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


def pick_lod(per: dict, mode: str = "1.2", lod22_max_rmse_m: float | None = None) -> list[str]:
    """LoD preference for one building in the scene (the CityJSON keeps every LoD Roofer made).
    "1.2": the block (footprint x height) for every building; "2.2": roof planes for every building;
    "auto": roof planes only when they fit the points well (Roofer RMSE <= lod22_max_rmse_m)."""
    if mode == "2.2":
        return ["2.2", "1.2"]
    if mode == "auto":
        rmse = per.get("rmse_lod22_m")
        if rmse is not None and lod22_max_rmse_m is not None and rmse <= lod22_max_rmse_m:
            return ["2.2", "1.2"]
    return ["1.2", "2.2"]


def building_meshes(jsonl_files: list[Path], per_building: dict, origin: np.ndarray, lod_mode: str = "2.2",
                    lod22_max_rmse_m: float | None = None, flag_rgba=FLAG) -> dict[str, trimesh.Trimesh]:
    """One mesh per exported building (pick_lod), in coordinates relative to origin."""
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
                geom = next((geoms[l] for l in pick_lod(per_building[bid], lod_mode, lod22_max_rmse_m)
                             if l in geoms), None)
                if not geom:
                    continue
                surfaces = geom["boundaries"][0] if geom["type"] == "Solid" else geom["boundaries"]
                v_all, f_all = [], []
                for surface in surfaces:
                    rings = [verts[np.asarray(r)] for r in surface]
                    tri = _triangulate(rings)
                    if tri is None:
                        continue
                    f_all.append(tri + sum(len(v) for v in v_all))
                    v_all.append(np.vstack(rings))
                if not v_all:
                    continue
                mesh = trimesh.Trimesh(vertices=np.vstack(v_all), faces=np.vstack(f_all), process=False)
                parts.setdefault(bid, []).append(mesh)
    out = {}
    for bid, ms in parts.items():
        m = trimesh.util.concatenate(ms)
        colour = flag_rgba if flagged(per_building[bid]) else BUILDING
        # one glTF material per building: Blender shows its base colour (vertex colours it does not)
        m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name="flagged" if colour is flag_rgba else "building", baseColorFactor=colour,
            metallicFactor=0.0, roughnessFactor=1.0))
        out[bid] = m
    return out


def flagged(per: dict) -> bool:
    """Buildings whose model is not trustworthy: a tower above the roof (roof maximum far above its
    70th percentile, typically a tree crown), or a roof fit so poor that the outline probably holds
    trees (Roofer RMSE above the low-confidence threshold). See buildings._assess."""
    return bool(per.get("roof_spike")) or per.get("status") == "low_confidence"


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


def write_scene(terrain, buildings: dict[str, trimesh.Trimesh], out: Path) -> dict:
    """terrain: one mesh, or a list of (name, mesh) parts (the textured 3D model)."""
    scene = trimesh.Scene()
    if isinstance(terrain, trimesh.Trimesh):
        terrain = [("terrain", terrain)]
    for name, part in terrain or []:
        scene.add_geometry(part, node_name=name, geom_name=name, transform=Z_UP_TO_Y_UP)
    for bid, mesh in sorted(buildings.items()):
        scene.add_geometry(mesh, node_name=f"building_{bid}", geom_name=f"building_{bid}", transform=Z_UP_TO_Y_UP)
    out.write_bytes(scene.export(file_type="glb"))
    return {"nodes": len(scene.graph.nodes_geometry), "bytes": out.stat().st_size}


BLENDER_SCRIPT = """
import bpy, math, sys
glb, blend, png, hide = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if hide == '1':
    for o in objs:
        if o.name.startswith('building_'):
            o.hide_set(True)
            o.hide_render = True
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
for screen in bpy.data.screens:  # open with textures visible: Solid shading coloured by texture
    for area in screen.areas:     # ('MATERIAL' shading cannot be set in background mode)
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'TEXTURE'
                    space.clip_end = span * 10
bpy.ops.wm.save_as_mainfile(filepath=blend)
"""


def blender_scene(blender: Path, glb: Path, out_dir: Path, log: Path, hide_buildings: bool = False) -> dict:
    """Open the glb in Blender headless; save scene.blend and a preview render. hide_buildings: building
    objects stay in the file but hidden (eye icon in the outliner), e.g. over a textured 3D model."""
    script = out_dir / "_blender_import.py"
    script.write_text(BLENDER_SCRIPT, encoding="utf-8")
    blend, png = out_dir / "scene.blend", out_dir / "preview.png"
    cmd = [str(blender), "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script),
           "--", str(glb), str(blend), str(png), "1" if hide_buildings else "0"]
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
    model = products.get("textured_model")
    if model and Path(model).is_file():
        # oblique capture: ODM's textured 3D model replaces the DTM mesh (facades, towers); the building
        # models stay in the scene, hidden in Blender, since they would cover the same surfaces
        terrain = textured_model(model, origin)
        result["scene_surface"] = {"source": model, "parts": len(terrain),
                                   "faces": int(sum(len(m.faces) for _, m in terrain))}
    meshes = {}
    if buildings_summary and buildings_summary.get("status") == "ok":
        files = [Path(f) for f in buildings_summary["files"]]
        mode, thr = ecfg.get("building_lod", "2.2"), ecfg.get("lod22_max_rmse_m")
        meshes = building_meshes(files, buildings_summary["per_building"], origin, mode, thr,
                                 list(ecfg.get("flag_colour_rgba", FLAG)))
        result["flagged_buildings"] = sorted(b for b in meshes if flagged(buildings_summary["per_building"][b]))
        result["lod_by_building"] = {bid: pick_lod(per, mode, thr)[0]
                                     for bid, per in buildings_summary["per_building"].items() if bid in meshes}
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
        result["blender"] = blender_scene(blender, glb, out_dir, log_dir / "blender.log",
                                          hide_buildings="scene_surface" in result)
    (out_dir / "terrain.ply").unlink(missing_ok=True)
    return result
