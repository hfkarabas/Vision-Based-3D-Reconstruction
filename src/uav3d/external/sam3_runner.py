"""Run SAM 3 text-prompted segmentation over image tiles. Executed with the ML venv's Python
(.tools/ml-venv), not imported by uav3d: it may only depend on torch, sam3, numpy and Pillow.

Usage: python sam3_runner.py <manifest.json>
manifest: {"prompts": [...], "tiles": [{"png": path, "npz": path}, ...]}
Each npz: masks (n, h, w) bool, scores (n,), prompt_index (n,) int.
"""
import json
import sys
import time

import numpy as np
import torch
from PIL import Image


def main(manifest_path: str) -> int:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    t0 = time.time()
    processor = Sam3Processor(build_sam3_image_model())
    print(f"loaded in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)
    for i, tile in enumerate(manifest["tiles"]):
        img = Image.open(tile["png"]).convert("RGB")
        masks, scores, which = [], [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(img)
            for k, prompt in enumerate(manifest["prompts"]):
                out = processor.set_text_prompt(state=state, prompt=prompt)
                if len(out["scores"]):
                    m = out["masks"].float().cpu().numpy().reshape(-1, img.height, img.width) > 0.5
                    masks.append(m)
                    scores.append(out["scores"].float().cpu().numpy())
                    which.append(np.full(len(m), k))
        np.savez_compressed(tile["npz"],
                            masks=np.concatenate(masks) if masks else np.zeros((0, img.height, img.width), bool),
                            scores=np.concatenate(scores) if scores else np.zeros(0, "float32"),
                            prompt_index=np.concatenate(which) if which else np.zeros(0, int))
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(manifest['tiles'])} tiles", flush=True)
    print(f"done in {time.time() - t0:.1f}s; gpu peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
