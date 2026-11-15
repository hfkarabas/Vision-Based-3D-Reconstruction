# Vision-Based 3D Reconstruction

Terrain and building models from nadir UAV photos on a single workstation, within a fixed memory budget.

Graduation project, Computer Engineering. Research topic: *Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA
Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon* (quality-feedback regional adaptive 3D
reconstruction of large UAV image blocks under limited compute).

The pipeline (`src/uav3d`) is a thin orchestration layer over proven open-source tools. Its own code covers
integration, terrain clean-up, building footprint fusion, the adaptive refinement scheduler and evaluation.

## What it does

Input: a folder of nadir (straight-down) mapping-flight photos with GPS in EXIF, or a GCP / position file.

1. **Check and plan:** reads the photos, estimates the memory load, splits the block if it would not fit.
2. **Photogrammetry (OpenDroneMap):** camera positions, dense point cloud, surface and terrain models, orthophoto.
3. **Terrain clean-up (PDAL):** removes outliers and water-reflection artefacts, separates the ground, builds a
   clean terrain model (DTM).
4. **Buildings:** footprints from OpenStreetMap, the point cloud and image evidence (SAM 3, optional); roof
   models with Roofer (LoD1.2 and LoD2.2). Unreliable models are flagged.
5. **Adaptive refinement (optional, research):** re-densifies one level finer only the 25 m tiles with the
   highest predicted gain, within an image budget.
6. **Export:** one textured scene with terrain and buildings.

Oblique (sideways) photo sets are out of scope; they are detected and recorded as a warning.

## Output

| File | Content |
|---|---|
| `scene/scene.glb` | Textured terrain and buildings in one file (any glTF viewer) |
| `scene/scene.blend` | The same scene for Blender; unreliable buildings in magenta |
| `scene/preview.png` | Preview image |
| `scene/buildings.city.json` | Buildings as CityJSON (GIS) |
| `scene/scene.json` | Coordinate system and scene origin |
| `terrain/dtm.tif`, `terrain/dsm.tif` | Terrain and surface height rasters |
| `odm/project/...` | ODM products: orthophoto, point cloud |
| `run.json` | Every step: duration, peak RAM, warnings, and the failing step if any |

## Install (Windows)

Requirements: 64-bit Windows, [uv](https://docs.astral.sh/uv/), about 2 GB of disk for the environment and tools
(another 5 GB for SAM 3). Blender is optional (only for the `.blend` file and the preview).

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1          # Python env + ODM 3.6.2 + Roofer 1.0.0 (SHA-256 checked)
powershell -ExecutionPolicy Bypass -File setup.ps1 -WithML  # optional: SAM 3 building detection
.venv\Scripts\python.exe -m uav3d doctor                     # check every tool
```

SAM 3 weights are gated: accept the licence at https://huggingface.co/facebook/sam3 and log in with
`.tools\ml-venv\Scripts\hf.exe auth login` before the first run.

## Run

**Desktop app:** double-click `run_app.bat`. Choose the photo folder and an output folder, press **Run**. The app
checks the tools, shows each step with its time and peak RAM, and opens the result (Blender, 3D viewer, folder).

**Command line:**

```powershell
.venv\Scripts\python.exe -m uav3d run <photo folder> --out <output folder> --threads 10
```

| Option | Effect |
|---|---|
| `--threads N` | worker threads for all tools (default: all logical CPUs) |
| `--budget-ram-gb G` | memory budget (default: a fraction of installed RAM) |
| `--gcp file` | ODM-format ground control points (default: found next to the photos) |
| `--refine-budget B` | adaptive refinement within B × the block's images (0–1, e.g. 0.5) |
| `--skip-buildings`, `--skip-export` | leave out the building or the scene step |
| `--reuse-odm` | reuse the photogrammetry of an earlier run in the output folder |
| `--keep-odm-intermediates` | keep ODM's depth maps (large; only for re-densifying regions) |

Coefficients live in `configs/defaults.json`; nothing in the pipeline is specific to a dataset.

## Repository

| Path | Content |
|---|---|
| `src/uav3d/` | Pipeline, desktop app (`app.py`), evaluation (`evaluate/`) |
| `tests/uav3d/` | Tests (`.venv\Scripts\python.exe -m pytest`) |
| `configs/` | Defaults, frozen gain model of the scheduler, dataset registry |
| `bench/` | Experiment scripts: hold-out runs, per-tile scores, replay, hypothesis analysis |
| `labels/buildings/` | Hand-drawn building labels (Sheffield) |
| `paper/` | Report and conference paper (LaTeX) |
| `Documentation/` | Architecture and research notes |
| `datasets/`, `outputs/` | Local data and results (not in Git) |

## Data

Experiments use only public nadir datasets: six OpenDroneMap sample sets (generality), Sheffield Cross (building
labels) and the three UseGeo strips with their LiDAR reference (UseGeo-1 for tuning, UseGeo-2/3 held out).
See `configs/dataset_registry.json` and `Documentation/research/DATASET_PROTOCOL.md`.

## Tools

OpenDroneMap 3.6.2 (OpenSfM, OpenMVS, PDAL, GDAL), Roofer 1.0.0 (3DBAG), SAM 3 (Meta), OpenStreetMap, Blender.
Versions and checksums are pinned in `tools.lock.json`.
