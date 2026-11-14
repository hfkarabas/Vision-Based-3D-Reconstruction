"""Prepare a UseGeo dataset for `uav3d run` (dataset preparation, not production code).

UseGeo images carry no EXIF GPS; camera positions come from the authors' aerial
triangulation (Camera_Inputs/eors.txt). They are written as an ODM geo.txt next to hard-linked
images, so the generic geo.txt support of the pipeline georeferences the run in the same
projected frame as the LiDAR reference.

The files do not state the CRS. Easting ~498 km sits on a UTM central meridian, so any UTM
zone keeps ODM's output in the identical frame; EPSG:32633 is used as a label only.
Usage: python bench/prepare_usegeo.py datasets/usegeo/dataset-1
"""
import os
import sys
from pathlib import Path

LABEL_CRS = "EPSG:32633"


def main(root: Path) -> None:
    src = root / "Camera_Inputs" / "Metashape_outputs_images"
    out = root / "uav3d_input"
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    eors = root / "Camera_Inputs" / "eors.txt"
    if not eors.is_file():  # dataset-2 ships only the orientation table (same leading columns)
        eors = next(root.glob("Image_orientations_dataset*.xyz"))
    for line in eors.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, x, y, z, omega, phi, kappa = line.split()[:7]
        img = src / name
        if not img.is_file():
            raise FileNotFoundError(img)
        dst = out / "images" / name
        if not dst.exists():
            os.link(img, dst)
        # positions only: omega/phi/kappa are photogrammetric angles, not ODM yaw/pitch/roll
        rows.append(f"{name} {x} {y} {z}")
    (out / "images" / "geo.txt").write_text(LABEL_CRS + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} images -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
