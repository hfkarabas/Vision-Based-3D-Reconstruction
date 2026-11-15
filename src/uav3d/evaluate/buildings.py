"""Building detection against hand labels (labels/buildings/<scene>.gpkg).

Matches the requirement "a building is there, roughly this wide": a labelled building and a
predicted one match when their footprints overlap with IoU >= iou_min (one-to-one, greedy
by IoU). Labels drawn as touching rectangles are merged into one building first.

Counted only inside the labelled sample areas: a label counts if it intersects an area, a
prediction if its centroid lies in one. Predictions matching an "uncertain" label are neither
true nor false positives, and uncertain labels never count as misses.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

PREDICTED = {"ok", "low_confidence"}  # statuses exported as buildings


def load_labels(gpkg: Path, crs) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    raw = gpd.read_file(gpkg, layer="buildings")
    areas = gpd.read_file(gpkg, layer="area").to_crs(crs)
    if raw.empty:
        return gpd.GeoDataFrame({"uncertain": []}, geometry=[], crs=crs), areas
    raw["uncertain"] = raw["uncertain"].fillna(0).astype(int)
    merged = shapely.unary_union(list(raw.geometry))
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    labels = gpd.GeoDataFrame(geometry=parts, crs=raw.crs)
    # a merged building is uncertain only if every rectangle it consists of was marked uncertain
    hits = gpd.sjoin(raw, labels, predicate="intersects")
    labels["uncertain"] = hits.groupby("index_right")["uncertain"].min().reindex(labels.index).fillna(0).astype(int)
    return labels.to_crs(crs), areas


def load_predictions(run_dir: Path) -> gpd.GeoDataFrame:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    per = (run.get("buildings") or {}).get("per_building", {})
    gpkg = run_dir / "buildings" / "roofprints.gpkg"
    if not gpkg.is_file():
        raise FileNotFoundError(f"no roofprints in {run_dir}")
    fp = gpd.read_file(gpkg)
    fp["status"] = [per.get(i, {}).get("status", "missing") for i in fp["fp_id"]]
    fp["height_m"] = [per.get(i, {}).get("height_m") for i in fp["fp_id"]]
    return fp[fp["status"].isin(PREDICTED)].reset_index(drop=True)


def match(labels: gpd.GeoDataFrame, preds: gpd.GeoDataFrame, iou_min: float) -> list[tuple[int, int, float]]:
    pairs = []
    if len(labels) and len(preds):
        for li, pi in gpd.sjoin(labels, preds, predicate="intersects")["index_right"].items():
            a, b = labels.geometry[li], preds.geometry[pi]
            inter = a.intersection(b).area
            iou = inter / (a.area + b.area - inter)
            if iou >= iou_min:
                pairs.append((li, pi, iou))
    used_l, used_p, out = set(), set(), []
    for li, pi, iou in sorted(pairs, key=lambda t: -t[2]):
        if li not in used_l and pi not in used_p:
            used_l.add(li)
            used_p.add(pi)
            out.append((li, pi, iou))
    return out


def evaluate(run_dir: Path, label_gpkg: Path, iou_min: float = 0.3) -> dict:
    preds = load_predictions(run_dir)
    labels, areas = load_labels(label_gpkg, preds.crs)
    region = shapely.unary_union(list(areas.geometry))
    labels = labels[labels.intersects(region)].reset_index(drop=True)
    preds = preds[preds.centroid.within(region)].reset_index(drop=True)
    matches = match(labels, preds, iou_min)
    matched_l = {li for li, _, _ in matches}
    matched_p = {pi for _, pi, _ in matches}
    uncertain = set(labels.index[labels["uncertain"] == 1])
    tp = sum(1 for li, _, _ in matches if li not in uncertain)
    fn = sum(1 for li in labels.index if li not in matched_l and li not in uncertain)
    ignored = {pi for li, pi, _ in matches if li in uncertain}
    fp = sum(1 for pi in preds.index if pi not in matched_p and pi not in ignored)
    area_err = [preds.geometry[pi].area / labels.geometry[li].area - 1 for li, pi, _ in matches if li not in uncertain]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"labels": len(labels) - len(uncertain), "predictions": len(preds) - len(ignored), "tp": tp, "fp": fp,
            "fn": fn, "precision": None if precision is None else round(precision, 3),
            "recall": None if recall is None else round(recall, 3), "f1": round(f1, 3),
            "median_iou": round(float(np.median([m[2] for m in matches])), 3) if matches else None,
            "median_area_error": round(float(np.median(area_err)), 3) if area_err else None,
            "iou_min": iou_min, "predicted_statuses": sorted(PREDICTED)}


TUNE_SCENES = {  # labelled tune scenes: name -> default run folder
    "sheffield": "outputs/uav3d/sheffield",
}


def main(argv: list[str] | None = None) -> int:
    """python -m uav3d.evaluate.buildings [scene=run_dir ...]  (default: all tune scenes)"""
    import sys

    repo = Path(__file__).resolve().parents[3]
    args = argv if argv is not None else sys.argv[1:]
    scenes = dict(a.split("=", 1) for a in args) if args else TUNE_SCENES
    rows = {}
    for name, run in scenes.items():
        rows[name] = evaluate(repo / run, repo / "labels" / "buildings" / f"{name}.gpkg")
        r = rows[name]
        print(f"{name:14s} labels {r['labels']:3d} preds {r['predictions']:4d} | TP {r['tp']:3d} FP {r['fp']:4d} "
              f"FN {r['fn']:3d} | P {r['precision']} R {r['recall']} F1 {r['f1']} | IoU {r['median_iou']} "
              f"area err {r['median_area_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
