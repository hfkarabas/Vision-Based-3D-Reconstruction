"""Summarise a smoke-suite folder (one run.json per dataset) against DATASET_PROTOCOL.md G1-G4.

Usage: python bench/summarize_smoke.py outputs/smoke
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

TERRAIN = ("point_cloud", "dsm", "dtm", "orthophoto")


def _flag(flags: list[str], name: str) -> str | None:
    return flags[flags.index(name) + 1] if name in flags[:-1] else None


def row(run_dir: Path) -> dict:
    r = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    stages = r.get("stages", [])
    gsd = next((s["gsd"] for s in stages if s.get("gsd")), {}) or {}
    images = (r.get("ingest") or {}).get("count")
    shots = gsd.get("shots")
    b = r.get("buildings") or {}
    products = r.get("products") or {}
    return {
        "dataset": run_dir.name,
        "status": r.get("status"),
        "error": (r.get("error") or "")[:160],
        "images": images,
        "registered": shots,
        "registered_pct": round(100 * shots / images, 1) if images and shots else None,
        "gsd_cm": round(gsd["gsd_m"] * 100, 2) if gsd.get("gsd_m") else None,
        "pc_quality": _flag((r.get("plan") or {}).get("flags", []), "--pc-quality"),
        "split": bool((r.get("plan") or {}).get("split")),
        "wall_min": round(sum(s.get("wall_s") or 0 for s in stages) / 60, 1),
        "peak_ram_gb": max((s.get("peak_rss_gb") or 0 for s in stages), default=None),
        "budget_ram_gb": (r.get("plan") or {}).get("budget_ram_gb"),
        "terrain_products": sum(bool(products.get(k)) for k in TERRAIN),
        "buildings_status": b.get("status"),
        "buildings": b.get("buildings"),
        "buildings_ok": (b.get("by_status") or {}).get("ok"),
        "buildings_low_conf": (b.get("by_status") or {}).get("low_confidence"),
        "scene": any(s["name"] == "export" and s.get("glb") for s in stages),
    }


def main(root: Path) -> None:
    rows = [row(d) for d in sorted(root.iterdir()) if (d / "run.json").is_file()]
    if not rows:
        print("no runs yet")
        return
    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if r["status"] in ("ok", "failed")]
    ok = [r for r in done if r["status"] == "ok"]
    g1 = 100 * len(ok) / len(done) if done else 0
    g2 = all(r["error"] for r in done if r["status"] == "failed")
    g3 = all((r["peak_ram_gb"] or 0) <= (r["budget_ram_gb"] or 0) for r in ok)
    g4 = all((r["registered_pct"] or 0) >= 95 for r in ok if r["registered_pct"] is not None)
    cols = ["dataset", "status", "images", "registered_pct", "gsd_cm", "pc_quality", "wall_min", "peak_ram_gb",
            "terrain_products", "buildings_status", "buildings", "buildings_ok", "buildings_low_conf", "scene", "error"]
    lines = ["# Smoke suite summary", "",
             f"- G1 finished without error: {len(ok)}/{len(done)} ({g1:.0f}%; target >= 90%)",
             f"- G2 every failure names stage and reason: {'yes' if g2 else 'NO'}",
             f"- G3 peak RAM within budget (successful runs): {'yes' if g3 else 'NO'}",
             f"- G4 >= 95% images registered (successful runs): {'yes' if g4 else 'NO'}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
