"""uav3d command line: ``doctor`` and ``run``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uav3d import __version__, ingest, odm, plan, tools
from uav3d.record import RunRecord


def cmd_doctor(_args) -> int:
    ok = True
    for name, good, detail in tools.doctor():
        ok &= good
        print(f"[{'ok' if good else 'MISSING'}] {name}: {detail}")
    return 0 if ok else 1


def cmd_run(args) -> int:
    out = Path(args.out).resolve()
    images_dir = Path(args.images).resolve()
    cfg = plan.defaults()
    rec = RunRecord(out / "run.json", uav3d=__version__, input=str(images_dir),
                    defaults_status=cfg["status"])
    stage = "setup"
    try:
        stage = "tools"
        o = tools.odm()
        rec.set("tools", {"odm": o.version, "odm_home": str(o.home)})

        stage = "ingest"
        images = ingest.find_images(images_dir)
        summary = ingest.summarise(images)
        rec.set("ingest", summary)
        if summary["gps_fraction"] < 1.0:
            rec.set("warning", "Not all images carry EXIF GPS; outputs may be in a local frame.")

        stage = "plan"
        budget = plan.ram_budget_gb(args.budget_ram_gb, cfg)
        p = plan.odm_plan(summary, budget, cfg)
        rec.set("plan", p)

        workdir = out / "odm"
        project = odm.prepare(images, workdir)
        log = out / "logs" / "odm.log"

        if p["split"] is None:
            stage = "odm:sfm"
            s = rec.stage(stage)
            s.update(odm.run(o, workdir, [*p["flags"], "--end-with", "opensfm"], log))
            gsd = odm.gsd_from_reconstruction(project / odm.PRODUCTS["reconstruction"])
            metric = summary["gps_fraction"] > 0
            s["gsd"] = {**gsd, "metric": metric}
            rec.save()

            stage = "odm:dense"
            res = plan.resolution_flags(gsd["gsd_m"] if metric else None, cfg)
            s = rec.stage(stage, resolution_flags=res)
            s.update(odm.run(o, workdir, [*p["flags"], *res, "--rerun-from", "openmvs"], log))
        else:
            # Split-merge runs all stages per submodel; ODM caps resolutions by its own GSD estimate.
            stage = "odm:split"
            s = rec.stage(stage)
            s.update(odm.run(o, workdir, p["flags"], log))
        rec.save()

        stage = "products"
        prods = odm.products(project)
        rec.set("products", prods)
        missing = [k for k in ("point_cloud", "dsm", "dtm", "orthophoto") if not prods[k]]
        if missing:
            raise odm.OdmError(f"ODM finished but did not produce: {', '.join(missing)}")
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
    r = sub.add_parser("run", help="reconstruct terrain products from a folder of UAV images")
    r.add_argument("images", help="folder with images (or with an images/ subfolder)")
    r.add_argument("--out", required=True, help="output folder")
    r.add_argument("--budget-ram-gb", type=float, default=None,
                   help="RAM budget; default: ram_budget_fraction of installed RAM")
    r.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
