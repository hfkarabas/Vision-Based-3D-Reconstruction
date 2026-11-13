"""uav3d command line: ``doctor`` and ``run``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uav3d import __version__, buildings, footprints, ingest, odm, plan, tools
from uav3d.record import RunRecord

TERRAIN = ("point_cloud", "dsm", "dtm", "orthophoto")


def cmd_doctor(_args) -> int:
    ok = True
    for name, good, detail in tools.doctor():
        ok &= good
        print(f"[{'ok' if good else 'MISSING'}] {name}: {detail}")
    return 0 if ok else 1


def _previous_odm(out: Path) -> dict | None:
    """Products and GSD of an earlier successful ODM phase in the same output folder."""
    try:
        prev = json.loads((out / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    prods = prev.get("products") or {}
    if not all(prods.get(k) and Path(prods[k]).is_file() for k in TERRAIN):
        return None
    gsd = next((s["gsd"] for s in prev.get("stages", []) if s.get("gsd")), None)
    return {"products": prods, "gsd": gsd, "plan": prev.get("plan"), "ingest": prev.get("ingest")}


def run_odm(rec: RunRecord, o: tools.Odm, images, summary, gcp, p, cfg, out: Path) -> tuple[dict, dict | None]:
    workdir = out / "odm"
    project = odm.prepare(images, workdir)
    log = out / "logs" / "odm.log"
    extra = ["--gcp", str(gcp)] if gcp else []
    gsd = None
    if p["split"] is None:
        s = rec.stage("odm:sfm")
        s.update(odm.run(o, workdir, [*p["flags"], *extra, "--end-with", "opensfm"], log))
        gsd = odm.gsd_from_reconstruction(project / odm.PRODUCTS["reconstruction"])
        gsd["metric"] = summary["gps_fraction"] > 0 or gcp is not None
        s["gsd"] = gsd
        rec.save()
        res = plan.resolution_flags(gsd["gsd_m"] if gsd["metric"] else None, cfg)
        s = rec.stage("odm:dense", resolution_flags=res)
        s.update(odm.run(o, workdir, [*p["flags"], *extra, *res, "--rerun-from", "openmvs"], log))
    else:
        # Split-merge runs all stages per submodel; ODM caps resolutions by its own GSD estimate.
        s = rec.stage("odm:split")
        s.update(odm.run(o, workdir, [*p["flags"], *extra], log))
    rec.save()
    prods = odm.products(project)
    missing = [k for k in TERRAIN if not prods[k]]
    if missing:
        raise odm.OdmError(f"ODM finished but did not produce: {', '.join(missing)}")
    return prods, gsd


def run_buildings(rec: RunRecord, prods: dict, gsd: dict | None, cfg: dict, out: Path) -> None:
    bdir = out / "buildings"
    bdir.mkdir(parents=True, exist_ok=True)
    s = rec.stage("buildings:footprints")
    metric_gsd = gsd["gsd_m"] if gsd and gsd.get("metric") else None
    gpkg = bdir / "roofprints.gpkg"
    s.update(footprints.prepare(prods["dsm"], prods["point_cloud"], metric_gsd, cfg, gpkg))
    rec.save()
    if not s["kept"]:
        # Not a failure: an explicit, reasoned "no buildings" outcome (protocol G1).
        rec.set("buildings", {"status": "none",
                              "reason": "no OSM or point-cloud roofprints covered by the reconstruction"})
        return
    s = rec.stage("buildings:roofer")
    s.update(buildings.run(tools.roofer(), Path(prods["point_cloud"]), gpkg, bdir / "roofer",
                           prods["dsm"], cfg, out / "logs" / "roofer.log"))
    rec.set("buildings", {"status": "ok", **s["summary"]})


def cmd_run(args) -> int:
    out = Path(args.out).resolve()
    images_dir = Path(args.images).resolve()
    cfg = plan.defaults()
    previous = _previous_odm(out) if args.reuse_odm else None
    rec = RunRecord(out / "run.json", uav3d=__version__, input=str(images_dir),
                    defaults_status=cfg["status"])
    stage = "setup"
    try:
        if previous:
            stage = "odm:reused"
            rec.set("ingest", previous["ingest"])
            rec.set("plan", previous["plan"])
            rec.stage(stage, gsd=previous["gsd"])
            prods, gsd = previous["products"], previous["gsd"]
        else:
            stage = "tools"
            o = tools.odm()
            rec.set("tools", {"odm": o.version, "odm_home": str(o.home)})
            stage = "ingest"
            images = ingest.find_images(images_dir)
            summary = ingest.summarise(images)
            gcp = Path(args.gcp).resolve() if args.gcp else ingest.find_gcp(images_dir, images)
            summary["gcp"] = str(gcp) if gcp else None
            rec.set("ingest", summary)
            if summary["gps_fraction"] < 1.0 and not gcp:
                rec.set("warning", "Not all images carry EXIF GPS and no GCP file; outputs may be in a local frame.")
            stage = "plan"
            p = plan.odm_plan(summary, plan.ram_budget_gb(args.budget_ram_gb, cfg), cfg)
            rec.set("plan", p)
            stage = "odm"
            prods, gsd = run_odm(rec, o, images, summary, gcp, p, cfg, out)
        rec.set("products", prods)
        if not args.skip_buildings:
            stage = "buildings"
            run_buildings(rec, prods, gsd, cfg, out)
        rec.finish("ok")
        print(f"ok: {out / 'run.json'}")
        return 0
    except Exception as exc:  # fail loud: record the stage and reason (protocol G2)
        if isinstance(exc, odm.OdmError) and rec.data["stages"]:
            rec.data["stages"][-1].update(exc.result)
        rec.finish("failed", f"{stage}: {type(exc).__name__}: {exc}")
        print(f"FAILED at {stage}: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uav3d", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="verify external tools").set_defaults(func=cmd_doctor)
    r = sub.add_parser("run", help="reconstruct terrain and buildings from a folder of UAV images")
    r.add_argument("images", help="folder with images (or with an images/ subfolder)")
    r.add_argument("--out", required=True, help="output folder")
    r.add_argument("--budget-ram-gb", type=float, default=None,
                   help="RAM budget; default: ram_budget_fraction of installed RAM")
    r.add_argument("--gcp", help="ODM-format GCP file (default: auto-detect next to the images)")
    r.add_argument("--reuse-odm", action="store_true",
                   help="reuse ODM products of an earlier successful run in --out")
    r.add_argument("--skip-buildings", action="store_true")
    r.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
