"""Fine dense pass of a whole kept ODM project (tuning data for the adaptive scheduler).

Re-densifies the full scene one OpenMVS level finer than ODM's run with ODM's own low-memory path.
ODM asks for a level but passes --min-resolution 640, which makes OpenMVS lower the level (42 MP UseGeo:
"medium" = level 4 requested, level 3 = 994 px used); the script refuses a "fine" pass that is not finer
than ODM's depth maps.
Low-memory path
(stages/openmvs.py, used there when OpenMVS runs out of memory): (1) depth maps of every image once,
without fusion (--fusion-mode 1); (2) the scene is split into sub-scenes (--sub-scene-area);
(3) each sub-scene is fused from the existing depth maps and filtered; (4) the PLYs are merged into
fine.laz in ODM's frame. Fusing the whole scene at once would not fit in memory (~0.15 GB per image at
level 3). Per-tile quality gains (fine vs coarse against LiDAR) are computed afterwards from fine.laz.
Resumable: OpenMVS reuses existing depth maps; finished sub-scene PLYs are skipped.

usage: python bench/fine_pass.py <odm project> <out> [--level 3] [--threads 10] [--sub-scene-area 660000]
"""
import argparse
import json
import shutil
from pathlib import Path

import laspy

from uav3d import refine, tools
from uav3d.record import run_measured


def omvs(o, args, log, cwd):
    info = run_measured([str(o.exe("DensifyPointCloud")), *map(str, args), "-v", "0", "--cuda-device", "-2"],
                        log, cwd=str(cwd), env=o.env())
    if info["returncode"] != 0:
        raise refine.RefineError(f"DensifyPointCloud {args[:2]} failed (exit {info['returncode']}); see {log}")
    return {"wall_s": info["wall_s"], "peak_rss_gb": info["peak_rss_gb"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--level", type=int, default=3, help="OpenMVS --resolution-level of the fine pass")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--sub-scene-area", type=int, default=660000, help="ODM's value (stages/openmvs.py)")
    ap.add_argument("--scene", type=Path, default=None, help="OpenMVS scene to use instead of ODM's (tests)")
    a = ap.parse_args()

    project, out = a.project.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "fine_pass.log"
    o = tools.odm()
    rec_path = out / "fine_pass.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else {}
    rec.update(project=str(project), level=a.level, threads=a.threads, sub_scene_area=a.sub_scene_area)

    def save():
        rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")

    scene = out / "scene.mvs"
    if not scene.is_file():  # image paths in ODM's scene.mvs are absolute; the copy reads the same images
        shutil.copy2(a.scene or project / "opensfm" / "undistorted" / "openmvs" / "scene.mvs", scene)
    dmaps = out / "depthmaps"
    dmaps.mkdir(exist_ok=True)
    (out / "Densify.ini").write_text("Optimize = 7\nMin Views Filter = 1\n", encoding="utf-8")
    (out / "subscene.ini").write_text("Optimize = 0\nEstimation Geometric Iters = 0\nMin Views Filter = 1\n",
                                      encoding="utf-8")
    max_res = refine.undistorted_max_size(project)
    level, side = refine.effective_size(max_res, a.level)
    coarse_dmap = next((project / "opensfm" / "undistorted" / "openmvs" / "depthmaps").glob("*.dmap"), None)
    coarse_side = max(refine.dmap_size(coarse_dmap)) if coarse_dmap else None
    rec.update(effective_level=level, side_px=side, coarse_side_px=coarse_side)
    if a.scene is None and coarse_side is not None and side <= coarse_side:
        raise SystemExit(f"fine pass would not be finer: {side} px vs ODM's {coarse_side} px (level {a.level} "
                         f"-> effective {level}); use a lower --level")
    common = ["--resolution-level", a.level, "--max-resolution", max_res,
              "--sub-resolution-levels", 2, "--number-views-fuse", 2, "--max-threads", a.threads,
              "--archive-type", 3, "-w", dmaps]

    if "depthmaps" not in rec:
        rec["depthmaps"] = omvs(o, [scene, *common, "--min-resolution", 640, "--dense-config-file",
                                    out / "Densify.ini", "--fusion-mode", 1], log, out)
        rec["depthmaps"]["count"] = len(list(dmaps.glob("*.dmap")))
        save()
    subs = sorted(out.glob("scene_[0-9][0-9][0-9][0-9].mvs"))
    if not subs:
        rec["split"] = omvs(o, [scene, "--sub-scene-area", a.sub_scene_area, "--max-threads", a.threads,
                                "-w", dmaps], log, out)
        subs = sorted(out.glob("scene_[0-9][0-9][0-9][0-9].mvs"))
        save()
    rec.setdefault("subscenes", {})
    plys = []
    for s in subs:
        done = s.with_name(s.stem + "_dense_dense_filtered.ply")
        if not done.is_file():
            e = omvs(o, [s, *common, "--dense-config-file", out / "subscene.ini", "--postprocess-dmaps", 0,
                         "--geometric-iters", 0], log, out)
            f = omvs(o, ["--filter-point-cloud", -20, "-i", s.with_name(s.stem + "_dense.mvs"),
                         "--max-threads", a.threads], log, out)
            rec["subscenes"][s.stem] = {"fuse": e, "filter": f}
            save()
        if done.is_file():
            plys.append(done)
        print(f"{s.stem}: {'ok' if done.is_file() else 'no PLY'}", flush=True)

    coarse = project / "odm_georeferencing" / "odm_georeferenced_model.laz"
    ox, oy = refine.utm_offset(project)
    h = laspy.open(coarse).header
    pad = 50.0  # the fine cloud may extend slightly beyond the coarse one
    whole = refine.Tile(0, 0, 0, h.mins[0] - ox - pad, h.mins[1] - oy - pad, h.maxs[0] - ox + pad, h.maxs[1] - oy + pad)
    rec["merge"] = refine.merge(o, str(coarse), [(str(p), [whole]) for p in plys], project, out / "fine.laz", log)
    rec["plys"] = [str(p) for p in plys]
    save()
    print(json.dumps({k: v for k, v in rec.items() if k != "subscenes"}))
    return 0 if plys and len(plys) == len(subs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
