"""Derive tool settings from the data and the resource budget (ARCHITECTURE.md section 5).

Every value comes from the input summary, the measured GSD or the machine budget, scaled by
coefficients in configs/defaults.json. Nothing here may depend on which dataset is running.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import psutil

from uav3d.tools import REPO


def defaults(path: Path | None = None) -> dict:
    return json.loads((path or REPO / "configs" / "defaults.json").read_text(encoding="utf-8"))


def ram_budget_gb(requested: float | None, cfg: dict) -> float:
    total = psutil.virtual_memory().total / 2**30
    budget = requested if requested else total * cfg["ram_budget_fraction"]
    if budget > total:
        raise ValueError(f"RAM budget {budget:.1f} GB exceeds installed RAM {total:.1f} GB.")
    return round(budget, 1)


def pc_quality(budget_gb: float, cfg: dict) -> str:
    for threshold, quality in cfg["pc_quality_by_ram_gb"]:
        if budget_gb >= threshold:
            return quality
    return cfg["pc_quality_by_ram_gb"][-1][1]


def odm_plan(summary: dict, budget_gb: float, cfg: dict) -> dict:
    """ODM flags shared by all phases, plus whether to split."""
    load = summary["count"] * summary["mean_megapixels"]           # megapixel-images
    capacity = cfg["split"]["megapixel_images_per_ram_gb"] * budget_gb
    flags = [*cfg["odm_flags"],
             "--pc-quality", pc_quality(budget_gb, cfg),
             "--max-concurrency", str(os.cpu_count() or 1)]
    split = None
    if load > capacity:
        per_model = max(20, math.floor(capacity / summary["mean_megapixels"]))
        split = {"images_per_submodel": per_model, "overlap_m": cfg["split"]["overlap_m"]}
        flags += ["--split", str(per_model), "--split-overlap", str(cfg["split"]["overlap_m"])]
    return {"budget_ram_gb": budget_gb, "load_mpix_images": round(load), "capacity_mpix_images": round(capacity),
            "split": split, "flags": flags}


def resolution_flags(gsd_m: float | None, cfg: dict) -> list[str]:
    """ODM takes cm/pixel. Without a metric GSD (no georeferencing) ODM's own defaults apply."""
    if not gsd_m:
        return []
    r = cfg["resolution"]
    to_cm = lambda f: f"{gsd_m * f * 100:.2f}"  # noqa: E731
    return ["--orthophoto-resolution", to_cm(r["orthophoto_gsd_factor"]),
            "--dem-resolution", to_cm(r["dem_gsd_factor"])]
