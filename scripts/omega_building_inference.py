"""Run GeoAI's pretrained building segmentation on a bounded, registered RGB raster."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time


def reuse(source, rgb, output):
    import numpy as np
    import rasterio

    with rasterio.open(source / "inference_rgb.tif") as original, rasterio.open(rgb) as target:
        if (original.crs, original.transform, original.shape) != (target.crs, target.transform, target.shape):
            raise ValueError("Cached inference has different coordinates or raster dimensions.")
        for _, window in original.block_windows(1):
            if not np.array_equal(original.read(window=window), target.read(window=window)):
                raise ValueError("Cached inference was computed from different RGB pixels.")
    for name in ("building_masks.tif", "building_probabilities.tif", "geoai_polygons.gpkg"):
        shutil.copy2(source / name, output / name)
    report = json.loads((source / "inference.json").read_text(encoding="utf-8"))
    report["reused_from"] = str(source)
    report["identical_rgb_grid_verified"] = True
    (output / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--hub-cache", type=Path)
    checkpoint.add_argument("--reuse", type=Path)
    parser.add_argument("--model", default="giswqs/whu-building-unetplusplus-efficientnet-b4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--minimum-area", type=float, default=20.0)
    parser.add_argument("--max-pixels", type=int, default=8_000_000)
    parser.add_argument("--revision")
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rgb = args.rgb.resolve()
    if args.reuse:
        reuse(args.reuse.resolve(), rgb, out)
        return

    # Use only the verified, revision-pinned Hub snapshot prepared by the operator.
    os.environ.update(
        HF_HUB_CACHE=str(args.hub_cache.resolve()),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    import geoai
    from huggingface_hub import hf_hub_download
    import rasterio
    import torch

    with rasterio.open(rgb) as src:
        if src.count != 3 or src.width * src.height > args.max_pixels:
            raise ValueError("Expected a bounded three-band RGB input.")
        if not src.crs or not src.crs.is_projected or src.crs.linear_units_factor[1] != 1:
            raise ValueError("The RGB raster must have a projected metre-based CRS.")

    mask = out / "building_masks.tif"
    probs = out / "building_class_probabilities.tif"
    model_path = Path(hf_hub_download(repo_id=args.model, filename="model.pth", revision=args.revision,
                                     local_files_only=True))
    with model_path.open("rb") as checkpoint:
        model_hash = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if args.checkpoint_sha256 and model_hash != args.checkpoint_sha256:
        raise ValueError("The cached WHU checkpoint does not match the configured SHA-256.")
    started = time.perf_counter()
    if args.revision:
        # The Hub convenience wrapper resolves the default branch. Pass the pinned
        # local files to the same GeoAI implementation used by that wrapper.
        config_path = hf_hub_download(repo_id=args.model, filename="config.json",
                                      revision=args.revision, local_files_only=True)
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        infer = geoai.timm_semantic_segmentation
        model = dict(
            model_path=str(model_path), encoder_name=config.get("encoder_name", "resnet50"),
            architecture=config.get("architecture", "unet"), num_channels=config.get("num_channels", 3),
            num_classes=config.get("num_classes", 2), use_timm_model=config.get("use_timm_model", False),
            timm_model_name=config.get("timm_model_name"),
        )
    else:
        infer = geoai.timm_segmentation_from_hub
        model = dict(repo_id=args.model)
    infer(
        input_path=str(rgb),
        output_path=str(mask),
        **model,
        window_size=512,
        overlap=256,
        batch_size=args.batch_size,
        device=args.device,
        probability_path=str(probs),
        probability_threshold=args.threshold,
    )
    with rasterio.open(probs) as src:
        if src.count != 2:
            raise ValueError("Expected background and building probability bands.")
        profile = dict(src.profile, count=1, compress="deflate", nodata=None)
        with rasterio.open(out / "building_probabilities.tif", "w", **profile) as dst:
            dst.write(src.read(2), 1)
            dst.set_band_description(1, "Class 1 building probability")
    with rasterio.open(mask) as src:
        mask_pixels = int((src.read(1) == 1).sum())
    polygons = geoai.orthogonalize(input_path=str(mask), epsilon=args.epsilon) if mask_pixels else None
    if polygons is not None:
        polygons = polygons.explode(index_parts=False, ignore_index=True)
        polygons = polygons.loc[
            polygons.geometry.is_valid & (polygons.geometry.area >= args.minimum_area)
        ]
        if not polygons.empty:
            polygons[["geometry"]].to_file(out / "geoai_polygons.gpkg", driver="GPKG", index=False)
    report = dict(
        input=str(rgb),
        model=args.model,
        model_revision=model_path.parent.name,
        model_checkpoint_sha256=model_hash,
        geoai_version=importlib.metadata.version("geoai-py"),
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        device=args.device,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        seconds=time.perf_counter() - started,
        window_size=512,
        overlap=256,
        batch_size=args.batch_size,
        threshold=args.threshold,
        polygon_epsilon_m=args.epsilon,
        minimum_area_m2=args.minimum_area,
        building_mask_pixels=mask_pixels,
        vector_polygons=len(polygons) if polygons is not None else 0,
    )
    (out / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
