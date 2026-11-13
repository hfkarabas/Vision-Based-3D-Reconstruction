"""Apply the published UAGLNet checkpoint to a local RGB orthophoto."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import albumentations as albu
import geoai
from monai.inferers import sliding_window_inference
import numpy as np
import rasterio
from safetensors.torch import load_file
import torch
import ttach


UPSTREAM_REVISION = "1f1d7dd96f77442587d08d1b2dee9a03cca2eff2"
UPSTREAM_SHA256 = "0d9fc6fa80658d47485bc1558ad7cacc7a2aedcd4561f82cea1375200f7a233f"
CHECKPOINT_SHA256 = "c93eb66050067f1b40af976fce04031175d294934436d33c59977652f28dfc32"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgb", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--minimum-area", required=True, type=float)
    args = parser.parse_args()
    root = args.home.resolve()
    source = root / "upstream/geoseg/models/UAGLNet.py"
    checkpoint = root / "model.safetensors"
    for path, expected in ((source, UPSTREAM_SHA256), (checkpoint, CHECKPOINT_SHA256)):
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"UAGLNet SHA-256 mismatch for {path.name}.")
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("uaglnet_upstream", source)
    upstream = module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    with rasterio.open(args.rgb) as src:
        if (src.count != 3 or src.dtypes != ("uint8",) * 3 or not src.crs
                or not src.crs.is_projected or src.crs.linear_units_factor[1] != 1
                or src.width * src.height > 8_000_000):
            raise ValueError("Expected a bounded three-band uint8 RGB orthophoto in a metric CRS.")
        rgb = src.read().transpose(1, 2, 0)
        valid = src.dataset_mask() > 0
        profile = dict(src.profile, count=1, compress="deflate", nodata=None)

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    model = upstream.UAGLNet()
    model.load_state_dict(load_file(checkpoint), strict=True)
    model.to(device).eval()
    transforms = ttach.Compose([ttach.HorizontalFlip(), ttach.VerticalFlip()])
    model = ttach.SegmentationTTAWrapper(model, transforms)
    image = albu.Normalize()(image=rgb)["image"]
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    def probabilities(batch):
        return model(batch).softmax(dim=1)

    with torch.inference_mode():
        output = sliding_window_inference(tensor, (512, 512), 1, probabilities,
                                          overlap=0.5, mode="gaussian", sw_device=device, device="cpu")
    # Upstream WHU labels assign class 0 to buildings.
    building_probability = output[0, 0].numpy()
    if not np.isfinite(building_probability).all():
        raise ValueError("UAGLNet returned non-finite probabilities.")
    mask = ((building_probability >= args.threshold) & valid).astype(np.uint8)
    mask_path = args.output / "building_masks.tif"
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.write_mask(valid.astype(np.uint8) * 255)
    with rasterio.open(args.output / "building_probabilities.tif", "w", **dict(profile, dtype="float32")) as dst:
        dst.write(np.where(valid, building_probability, 0), 1)
        dst.write_mask(valid.astype(np.uint8) * 255)
    polygon_count = 0
    if mask.any():
        polygons = geoai.orthogonalize(input_path=str(mask_path), epsilon=0.6)
        polygons = polygons.explode(index_parts=False, ignore_index=True)
        polygons = polygons.loc[polygons.geometry.is_valid & (polygons.geometry.area >= args.minimum_area)]
        polygon_count = len(polygons)
        if polygon_count:
            polygons[["geometry"]].to_file(args.output / "geoai_polygons.gpkg", index=False)
    report = dict(model="ldxxx/UAGLNet_WHU", upstream=UPSTREAM_REVISION,
                  source_sha256=UPSTREAM_SHA256,
                  model_revision="9fcb9e7", checkpoint_sha256=CHECKPOINT_SHA256,
                  input=str(args.rgb.resolve()), device=str(device), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                  seconds=time.perf_counter() - start, window_size=512, overlap=0.5, tta="horizontal+vertical",
                  threshold=args.threshold, minimum_area_m2=args.minimum_area, polygon_epsilon_m=0.6,
                  polygons=polygon_count, mask_pixels=int(mask.sum()))
    (args.output / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
