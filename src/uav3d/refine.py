"""Regional re-densification (research contribution, Documentation/research/ADAPTIVE_SCHEDULER.md).

A region is a square tile in ODM's reconstruction frame (UTM minus the offset in
odm_georeferencing/coords.txt). OpenMVS computes a full depth map for every image of a scene, so an
ROI does not bound the cost; an image subset does. For a group of adjacent tiles, OpenSfM's own
OpenMVS exporter writes a scene with only the shots that observe >= k sparse points in the group
(external/opensfm_tiles.py, run with ODM's Python). That scene is densified with ODM's OpenMVS flags at
a finer level, shifted by ODM's UTM offset and cropped to the group, and replaces the coarse points
there. SfM, camera model and tools are those of the coarse run.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from uav3d.record import run_measured

HELPER = Path(__file__).resolve().parent / "external" / "opensfm_tiles.py"


class RefineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tile:
    id: int
    ix: int
    iy: int
    x0: float
    y0: float
    x1: float
    y1: float

    def wkt_ring(self, dx: float = 0.0, dy: float = 0.0) -> str:
        x0, y0, x1, y1 = (float(v) for v in (self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy))
        return f"(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


def grid(xmin: float, ymin: float, xmax: float, ymax: float, size: float) -> list[Tile]:
    """Square tiles of side `size` covering the box, anchored at (xmin, ymin)."""
    if size <= 0 or xmax <= xmin or ymax <= ymin:
        raise ValueError("grid needs a positive size and a non-empty box")
    nx, ny = math.ceil((xmax - xmin) / size), math.ceil((ymax - ymin) / size)
    return [Tile(iy * nx + ix, ix, iy, xmin + ix * size, ymin + iy * size,
                 xmin + (ix + 1) * size, ymin + (iy + 1) * size)
            for iy in range(ny) for ix in range(nx)]


def tile_index(xy, origin: tuple[float, float], size: float, shape: tuple[int, int]):
    """Flat tile id (iy * nx + ix) per XY point on a grid of `shape` (rows, cols); -1 outside the grid."""
    import numpy as np

    ix = np.floor((xy[:, 0] - origin[0]) / size).astype(np.int64)
    iy = np.floor((xy[:, 1] - origin[1]) / size).astype(np.int64)
    ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    return np.where(ok, iy * shape[1] + ix, -1)


def groups(selected: list[Tile]) -> list[list[Tile]]:
    """4-connected components of the selected tiles: adjacent tiles share images, so each component is
    densified as one scene and its shared images are processed once."""
    by_cell = {(t.ix, t.iy): t for t in selected}
    seen, out = set(), []
    for start in sorted(by_cell):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            cell = stack.pop()
            comp.append(by_cell[cell])
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (cell[0] + d[0], cell[1] + d[1])
                if nb in by_cell and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        out.append(sorted(comp, key=lambda t: t.id))
    return out


def multipolygon(tiles: list[Tile], dx: float = 0.0, dy: float = 0.0) -> str:
    return "MULTIPOLYGON(" + ", ".join(t.wkt_ring(dx, dy) for t in tiles) + ")"


def utm_offset(project: Path) -> tuple[float, float]:
    """ODM's (east, north) offset: reconstruction/OpenMVS coordinates + offset = UTM."""
    lines = (project / "odm_georeferencing" / "coords.txt").read_text(encoding="utf-8").splitlines()
    east, north = lines[1].split()[:2]
    return float(east), float(north)


def utm_srs(project: Path) -> str:
    """EPSG code of ODM's output frame from coords.txt ("WGS84 UTM 33N" -> EPSG:32633)."""
    head = (project / "odm_georeferencing" / "coords.txt").read_text(encoding="utf-8").splitlines()[0].split()
    if len(head) < 3 or head[1].upper() != "UTM":
        raise RefineError(f"unexpected coords.txt header {head!r}")
    zone, hemi = int(head[2][:-1]), head[2][-1].upper()
    return f"EPSG:{(32600 if hemi == 'N' else 32700) + zone}"


def subsets(odm, project: Path, tiles: list[Tile], k: int, work: Path, log: Path,
            export: list[tuple[str, list[Tile]]] = ()) -> dict:
    """Sparse points and observing shots per tile; optionally export one OpenMVS scene per named group."""
    work.mkdir(parents=True, exist_ok=True)
    req = {"opensfm": str(project / "opensfm"), "odm_home": str(odm.home), "k": int(k),
           "tiles": [[t.id, t.x0, t.y0, t.x1, t.y1] for t in tiles],
           "export": [{"name": name, "tiles": [t.id for t in g], "out": str(work / name)} for name, g in export]}
    path = work / "subsets.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    info = run_measured([str(odm.python), "-X", "utf8", str(HELPER), str(path)], log, env=odm.env())
    if info["returncode"] != 0:
        raise RefineError(f"tile subsets failed (exit {info['returncode']}); see {log}")
    result = json.loads((work / "subsets.result.json").read_text(encoding="utf-8"))
    result["measure"] = info
    return result


def undistorted_max_size(project: Path) -> int:
    """ODM's --max-resolution for OpenMVS: the larger side of the undistorted images."""
    from PIL import Image

    images = project / "opensfm" / "undistorted" / "images"
    first = next((p for p in sorted(images.iterdir()) if p.suffix.lower() in (".tif", ".jpg", ".png")), None)
    if first is None:
        raise RefineError(f"no undistorted images in {images} (run kept ODM intermediates?)")
    with Image.open(first) as im:
        return max(im.size)


def effective_size(max_resolution: int, level: int, min_resolution: int = 640) -> tuple[int, int]:
    """(level, image side in px) OpenMVS actually uses: max_resolution >> level, but the level is lowered
    until the side reaches min_resolution (OpenMVS Image::RecomputeMaxResolution). With ODM's
    --min-resolution 640, a 7952 px image at "level 4" is densified at level 3 (994 px)."""
    size = max_resolution >> level
    while level > 0 and size < min_resolution:
        level -= 1
        size = max_resolution >> level
    return level, size


def dmap_size(path: Path) -> tuple[int, int]:
    """(width, height) of an OpenMVS .dmap depth map (header: 'DR', type, padding, image w/h, depth w/h)."""
    import struct

    with open(path, "rb") as fh:
        head = fh.read(20)
    if head[:2] != b"DR":
        raise RefineError(f"not an OpenMVS depth map: {path}")
    return struct.unpack("<II", head[12:20])


def densify(odm, scene: Path, level: int, max_resolution: int, n_threads: int, log: Path,
            min_resolution: int = 640, filter_points: bool = True) -> dict:
    """OpenMVS densification of one exported scene with ODM's flags (stages/openmvs.py) at `level`.
    CPU only: the bundled OpenMVS has no kernels for sm_120 GPUs and crashes on them."""
    work = scene.parent
    dmaps = work / "depthmaps"
    dmaps.mkdir(exist_ok=True)
    ini = work / "Densify.ini"
    ini.write_text("Optimize = 7\nMin Views Filter = 1\n", encoding="utf-8")
    exe = str(odm.exe("DensifyPointCloud"))
    cmd = [exe, str(scene), "--resolution-level", str(level), "--dense-config-file", str(ini),
           "--max-resolution", str(max_resolution), "--min-resolution", str(min_resolution),
           "--max-threads", str(n_threads), "--number-views-fuse", "2", "--sub-resolution-levels", "2",
           "--archive-type", "3", "-w", str(dmaps), "-v", "0", "--cuda-device", "-2"]
    info = run_measured(cmd, log, cwd=str(work), env=odm.env())
    dense = work / "scene_dense.mvs"
    if info["returncode"] != 0 or not dense.is_file():
        raise RefineError(f"DensifyPointCloud failed on {scene} (exit {info['returncode']}); see {log}")
    stats = {"wall_s": info["wall_s"], "peak_rss_gb": info["peak_rss_gb"]}
    if filter_points:
        f = run_measured([exe, "--filter-point-cloud", "-20", "-i", str(dense), "--max-threads", str(n_threads),
                          "-v", "0", "--cuda-device", "-2"], log, cwd=str(work), env=odm.env())
        stats["filter_wall_s"] = f["wall_s"]
        stats["peak_rss_gb"] = max(stats["peak_rss_gb"], f["peak_rss_gb"])
    ply = next((p for p in (work / "scene_dense_dense_filtered.ply", work / "scene_dense.ply") if p.is_file()), None)
    if ply is None:
        raise RefineError(f"no dense PLY next to {scene}")
    stats["ply"] = str(ply)
    return stats


SPLIT_MIN_IMAGES = 40  # below this a scene is fused whole (fine level: ~0.15 GB per image)


def _omvs(odm, args, log: Path, cwd: Path, n_threads: int) -> dict:
    info = run_measured([str(odm.exe("DensifyPointCloud")), *map(str, args), "-v", "0", "--cuda-device", "-2"],
                        log, cwd=str(cwd), env=odm.env(), cpus=n_threads)
    if info["returncode"] != 0:
        raise RefineError(f"DensifyPointCloud {[str(x) for x in args[:2]]} failed (exit {info['returncode']}); see {log}")
    return {"wall_s": info["wall_s"], "peak_rss_gb": info["peak_rss_gb"]}


def densify_lowmem(odm, scene: Path, work: Path, level: int, max_resolution: int, n_threads: int, log: Path,
                   sub_scene_area: int = 660000, rec: dict | None = None, save=lambda: None) -> dict:
    """ODM's low-memory OpenMVS path (stages/openmvs.py, used there when OpenMVS runs out of memory):
    (1) depth maps of every image once, without fusion (--fusion-mode 1); (2) split into sub-scenes of
    bounded sampling area (--sub-scene-area, in pixels at the chosen level, so finer levels give more
    sub-scenes); (3) fuse and filter each sub-scene from the existing depth maps. Peak memory is bounded by
    the sub-scene, not by the block. Resumable through ``rec`` (records of finished steps) and the files.
    Returns ``rec`` with "plys" (one filtered PLY per sub-scene, ODM's reconstruction frame)."""
    rec = {} if rec is None else rec
    work.mkdir(parents=True, exist_ok=True)
    local = work / "scene.mvs"
    if Path(scene).resolve() != local.resolve() and not local.is_file():
        import shutil
        shutil.copy2(scene, local)  # image paths inside are absolute
    dmaps = work / "depthmaps"
    dmaps.mkdir(exist_ok=True)
    (work / "Densify.ini").write_text("Optimize = 7\nMin Views Filter = 1\n", encoding="utf-8")
    (work / "subscene.ini").write_text("Optimize = 0\nEstimation Geometric Iters = 0\nMin Views Filter = 1\n",
                                       encoding="utf-8")
    common = ["--resolution-level", level, "--max-resolution", max_resolution,
              "--sub-resolution-levels", 2, "--number-views-fuse", 2, "--max-threads", n_threads,
              "--archive-type", 3, "-w", dmaps]
    if "depthmaps" not in rec:
        rec["depthmaps"] = _omvs(odm, [local, *common, "--min-resolution", 640, "--dense-config-file",
                                       work / "Densify.ini", "--fusion-mode", 1], log, work, n_threads)
        rec["depthmaps"]["count"] = len(list(dmaps.glob("*.dmap")))
        save()
    subs = sorted(work.glob("scene_[0-9][0-9][0-9][0-9].mvs"))
    if rec.get("split", {}).get("whole_scene"):
        subs = [local]
    elif not subs:
        n_img = rec["depthmaps"]["count"]
        if n_img <= SPLIT_MIN_IMAGES:
            # small scenes fit in memory whole; OpenMVS's splitter also crashed on them (0xC00000FD)
            rec["split"] = {"whole_scene": True, "reason": f"{n_img} images <= {SPLIT_MIN_IMAGES}"}
            subs = [local]
        else:
            rec["split"] = _omvs(odm, [local, "--sub-scene-area", sub_scene_area, "--max-threads", n_threads,
                                       "-w", dmaps], log, work, n_threads)
            subs = sorted(work.glob("scene_[0-9][0-9][0-9][0-9].mvs"))
        save()
    rec.setdefault("subscenes", {})
    plys = []
    for s in subs:
        done = s.with_name(s.stem + "_dense_dense_filtered.ply")
        if not done.is_file():
            e = _omvs(odm, [s, *common, "--dense-config-file", work / "subscene.ini", "--postprocess-dmaps", 0,
                            "--geometric-iters", 0], log, work, n_threads)
            f = _omvs(odm, ["--filter-point-cloud", -20, "-i", s.with_name(s.stem + "_dense.mvs"),
                            "--max-threads", n_threads], log, work, n_threads)
            rec["subscenes"][s.stem] = {"fuse": e, "filter": f}
            save()
        if done.is_file():
            plys.append(done)
    rec["plys"] = [str(p) for p in plys]
    rec["subscene_count"] = len(subs)
    return rec


def drop_tiles(laz_in, tiles: list[Tile], offset: tuple[float, float], out_laz: Path,
               chunk: int = 5_000_000) -> dict:
    """Copy a UTM point cloud without its points inside `tiles` (reconstruction frame; + offset = UTM).
    Done here, on the tile grid itself, because PDAL's filters.crop with a multipolygon and outside=true keeps
    every point (each point is outside some member polygon)."""
    import laspy
    import numpy as np

    if not tiles:
        raise ValueError("drop_tiles needs at least one tile")
    size = tiles[0].x1 - tiles[0].x0
    gx = tiles[0].x0 - tiles[0].ix * size + offset[0]
    gy = tiles[0].y0 - tiles[0].iy * size + offset[1]
    mask = np.zeros((max(t.iy for t in tiles) + 1, max(t.ix for t in tiles) + 1), bool)
    for t in tiles:
        mask[t.iy, t.ix] = True
    kept = dropped = 0
    with laspy.open(laz_in) as r, laspy.open(out_laz, mode="w", header=r.header) as w:
        for pts in r.chunk_iterator(chunk):
            ix = np.floor((np.asarray(pts.x) - gx) / size).astype(np.int64)
            iy = np.floor((np.asarray(pts.y) - gy) / size).astype(np.int64)
            ok = (ix >= 0) & (iy >= 0) & (ix < mask.shape[1]) & (iy < mask.shape[0])
            inside = np.zeros(len(ix), bool)
            inside[ok] = mask[iy[ok], ix[ok]]
            w.write_points(pts[~inside])
            kept += int((~inside).sum())
            dropped += int(inside.sum())
    return {"kept": kept, "dropped": dropped}


def merge_pipeline(coarse_laz: str | None, refined: list[tuple[str, list[Tile]]], offset: tuple[float, float],
                   srs: str, out_laz: Path, refined_class: int = 1) -> list:
    """PDAL pipeline: the coarse cloud (already without the refined tiles, see drop_tiles) + each group's
    refined points (shifted to UTM, cropped to the group), written as one LAZ in ODM's frame. OpenMVS PLYs
    carry no SRS, so ODM's is given. Without a coarse cloud only the refined points are written."""
    ox, oy = offset
    stages: list = []
    merged = []
    if coarse_laz:
        stages += [{"type": "readers.las", "filename": coarse_laz, "override_srs": srs, "tag": "keep"}]
        merged.append("keep")
    for i, (ply, g) in enumerate(refined):
        stages += [{"type": "readers.ply", "filename": ply, "override_srs": srs, "tag": f"r{i}"},
                   {"type": "filters.transformation", "inputs": [f"r{i}"],
                    "matrix": f"1 0 0 {ox} 0 1 0 {oy} 0 0 1 0 0 0 0 1", "tag": f"t{i}"},
                   {"type": "filters.crop", "inputs": [f"t{i}"], "polygon": multipolygon(g, ox, oy), "a_srs": srs,
                    "tag": f"c{i}"},
                   {"type": "filters.assign", "inputs": [f"c{i}"], "value": f"Classification = {refined_class}",
                    "tag": f"a{i}"}]
        merged.append(f"a{i}")
    stages += [{"type": "filters.merge", "inputs": merged, "tag": "all"},
               {"type": "writers.las", "inputs": ["all"], "filename": str(out_laz), "compression": "laszip",
                "minor_version": 4, "dataformat_id": 3, "offset_x": ox, "offset_y": oy, "offset_z": 0,
                "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001, "extra_dims": "all"}]
    return stages


def merge(odm, coarse_laz: str | None, refined: list[tuple[str, list[Tile]]], project: Path, out_laz: Path,
          log: Path) -> dict:
    offset = utm_offset(project)
    kept, dropped = None, None
    if coarse_laz:
        kept = Path(out_laz).with_name(Path(out_laz).stem + ".coarse_kept.laz")
        dropped = drop_tiles(coarse_laz, [t for _, g in refined for t in g], offset, kept)
    pipeline = merge_pipeline(str(kept) if kept else None, refined, offset, utm_srs(project), out_laz)
    path = out_laz.with_suffix(".pipeline.json")
    path.write_text(json.dumps(pipeline, indent=1), encoding="utf-8")
    info = run_measured([str(odm.exe("pdal")), "pipeline", str(path)], log, env=odm.env())
    if info["returncode"] != 0 or not out_laz.is_file():
        raise RefineError(f"PDAL merge failed (exit {info['returncode']}); see {log}")
    if kept:
        kept.unlink()
    return {"laz": str(out_laz), "wall_s": info["wall_s"], "peak_rss_gb": info["peak_rss_gb"], "coarse": dropped}
