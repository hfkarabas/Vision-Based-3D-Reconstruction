# Mission 0 research foundation

Status: **baseline checkpoint; mission gates remain open**, 12 November 2026. Both complete independent SfM + MVS pipelines and two surface/texturing routes have finished. Both full montages have been inspected by the agent; explicit user acceptance is pending. No novel reconstruction algorithm or production refactor was implemented.

## Reproducible evidence

All new experiments are isolated in `outputs/research-baseline/`. Original datasets, old models, native source, and release bundles are preserved. The starting revision is `9e008ddb6350da2b67d578fbc342795bf49e4b68`. `reports/initial-git-status.txt` records unrelated deleted/modified academic deliverables and untracked captures, reference material, and mission launch files. Those changes must not enter the mission commit.

Read together: [pipeline audit](PIPELINE_AUDIT.md), [literature review](LITERATURE_REVIEW.md), [prior work comparison](PRIOR_WORK_COMPARISON.md), [research hypothesis](RESEARCH_HYPOTHESIS.md), and [architecture](PROPOSED_ARCHITECTURE.md). The historical sources are `FINAL_RELEASE_REPORT.md`, `AGENT_PROGRESS.md`, `docs/AIRZOO_SURFACE_VALIDATION.md`, `docs/AIRZOO_GEOMETRIC_RECOVERY.md` (III-B evidence), `docs/AIRZOO_MISSION_III_C.md`, and `docs/AIRZOO_VISUAL_RECOVERY.md` (III-E). III-D is recorded in the progress log and final release report; no separately named III-D markdown report was found in the initial file inventory.

## Inputs actually available

| Scene | Source folder relative to repository | Originals | Resolution |
|---|---|---:|---|
| Rural | `datasets/airzoo/rural/Scene2/query` | 190 JPG | 4032 × 3024 |
| Urban | `datasets/airzoo/urban/Scene1/query` | 161 JPG | 4032 × 3024 |

These are DJI M4T WideCamera photographs with embedded EXIF/XMP, RTK position, exposure time and gimbal angles. SHA-256, dimensions, byte size and metadata for every photograph are in `reports/dataset-audit.json`. Full contact sheets are `reports/{rural,urban}-contact-sheet.jpg`. Historical `*-colmap*/images` directories contain decoded/resized benchmark derivatives; they are not the original collection. Do not accidentally treat the upstream rendered `airzoo_real_recon` split as measured UAV imagery: that split is not present locally. The [upstream dataset card](https://huggingface.co/datasets/RingoWRW97/AirZoo-Real) distinguishes these products.

Available ancillary files include MRK, NAV, OBS and RTK records, DOM/DSM TIFFs, world files, XML metadata and satellite reference maps; exact paths and sizes are inventoried rather than inferred. Other local datasets are Middlebury motorcycle stereo, OpenCV basketball, Sceaux, Temple, and a Tanks and Temples Caterpillar video. Their presence does not establish UAV provenance or local laser-ground-truth availability. `reports/all-datasets.json` records their file inventories.

The acquisition maps use XMP latitude/longitude in a local equirectangular diagnostic frame, with gimbal yaw arrows. A 15 m proximity diagnostic finds 11 Rural and 13 Urban position clusters. Camera-position extents are approximately 531 × 298 m and 382 × 478 m; these are **camera** footprints, not reconstructed terrain footprints. Rural pitch ranges from −90° to −24.1°, Urban from −72.9° to −25.4°. The contact sheets show oblique viewpoints and repeated observations from stations. Filenames are not a valid neighbor-selection rule. Proximity alone does not prove geometric connectivity; exhaustive verified matching and recovered tracks provide that evidence.

## Coordinate and reference audit

Both XML sidecars declare EPSG:4547, CGCS2000 / 3-degree Gauss-Kruger CM 114E, in metres. Native EPSG axis order is northing/easting; array and export interfaces must explicitly specify easting/northing, for example `always_xy=True`. [CRS definition](https://epsg.io/4547).

MRK fields explicitly label height `Ellh`. XMP has `AltitudeType=RtkAlt`; these observations cannot be assumed to use the DSM vertical datum. No independent checkpoints or calibrated laser surface have been established for these scenes. DOM and DSM are supplied reference products, with unresolved acquisition/processing lineage; they are not automatically independent ground truth.

The TIFF header audit reveals a conflict: Urban DOM declares geographic WGS84 keys while its affine transform has metre-sized projected coordinates and the XML says EPSG:4547. The Rural DOM and both DSMs have no inspected GeoTIFF CRS keys. Both world files use a **positive** row increment; preserve this convention, rather than assuming a north-up negative increment. Rural rasters are 21615 × 17949, Urban 28349 × 18324. Sidecars specify approximately 0.05642 m and 0.05270 m pixels. Do not silently rewrite these files. Resolve spatial orientation by comparing landmarks with source imagery before interpreting DSM residuals.

Independent SfM is an arbitrary similarity frame. Position metadata is used for acquisition diagnostics and post-estimation camera plausibility only. The incremental mapper receives no native poses, no native intrinsics, no position-hint file and no pose-prior mapper. EXIF supplies an initial focal estimate; COLMAP refines its shared SIMPLE_RADIAL camera. GPS records stored by COLMAP's reader are not positional constraints in the chosen incremental mapper command. Any fit to camera positions must be labelled in-sample registration, not independent accuracy evaluation. Offset-adjusted DSM residuals are relative elevation agreement, not absolute geospatial accuracy or full 3D accuracy.

## Environment and initial baseline

The existing portable executable is `outputs/airzoo-benchmark/geometric-root-cause-recovery/colmap-cuda/bin/colmap.exe`: COLMAP 4.2.1, commit bd1fcf6, CUDA build. RTX 5070 has 12227 MiB VRAM, driver 617.42. Python reports 14 logical CPUs and 16,782,508,032 bytes total RAM. CIM hardware queries are denied in the sandbox; no system settings were changed. CPU-only PyCOLMAP 4.2.1 is available in the historical project-local `colmap-python` directory, outside the default Python import path. Its `has_cuda=False` does not describe the separate CUDA CLI. GPU SIFT execution succeeded with that CLI.

Initial SfM uses every original JPG, a 2000-pixel feature-extraction limit, exhaustive matching, four CPU threads and shared SIMPLE_RADIAL intrinsics. Fresh databases contain no imported observations or native geometry. Both scenes completed: Urban has 161/161 cameras, 38,215 sparse points, 225,723 observations, mean track length 5.9067 and mean reprojection error 1.3069 original-image pixels; its verified graph has 2,389 edges in one component. Rural completed with 190/190 cameras, 63,234 points, 340,503 observations, mean track length 5.3848 and mean reprojection error 1.3390 pixels in original image coordinates. Its database verified graph is one component of 190. Rural intrinsics are f=2852.1346 px, principal point (2016,1512), radial k=0.02985. Camera RTK similarity residual median is 0.395 m, P95 0.796 m; this is an in-sample diagnostic. The mapper logged CHOLMOD step warnings and nevertheless completed; retain warnings and inspect final geometry rather than declaring convergence from exit code alone.

Dense pilot: all cameras in the largest independently estimated model; 900-pixel undistorted images; six automatically selected source views; geometric PatchMatch consistency, default iteration count; four CPU threads for fusion. PatchMatch uses a 2 GB cache option. Fusion also receives cache_size=2, but installed use_cache defaults false, so it is **not a 2 GB fusion memory limit**. This is a reduced image-resolution run, not a reduced camera subset. Delaunay and depth-10 Poisson are compared. Current installed COLMAP includes `mesh_texturer`; the [official CLI](https://colmap.github.io/cli.html) documents it. Stage attribution reflects the installed version. OpenMVS remains a separate comparison, not a prerequisite for this available route.

Exact argument arrays, logs, stage runtimes and sampled process RSS are persisted beside each stage. Dense-stage GPU readings, where collected, are whole-device samples and include unrelated applications. Earlier SfM stages did not capture VRAM peaks. No unsupported peak measurement is substituted.

## Mission acceptance gates

Additional reference validity finding: both DSMs contain finite values over their whole sampled rectangular raster, including the DOM's blank margins, without an inspected nodata tag. New DSM evaluation therefore uses a conservative visible-DOM mask (nearest RGB sampling and one-mask-pixel erosion, about metre-scale boundary uncertainty). This is a reference-footprint convention, not an independent completeness certificate. Historical DSM numbers did not use this new mask and must retain their original labels.

1. Independent full-scene SfM: Rural completed; Urban also completed (161/161).
2. Dense fusion and surfaces: complete, 1,577,923 Rural / 1,292,342 Urban fused points. Supported surface area is not certified. Raw Delaunay has extensive peripheral unsupported-looking triangles and is rejected as a usable complete asset.
3. Photographic meshes, OBJ/MTL/atlas, GLB and independent Blender import: complete for Delaunay and Poisson in both scenes. Native validation return codes are all zero. Successful import does not establish visual acceptance.
4. Complete top, oblique, side, neutral, wireframe and photographic montages: complete and inspected by the agent. Poisson is the preferred review candidate; Rural is coherent and recognizable, Urban recognizable but substantially incomplete. No user acceptance recorded.
5. Comparable native-versus-established metrics: historical results are controls, not a new equal-input benchmark.
6. Literature: close overlap established; comprehensive screening and precise novelty determination remain separate gates.

The seven final decision questions must be answered from completed reports. The currently demonstrated research problem is native fragmentation and incomplete photographic coverage. Whether a new budgeted scheduler solves a limitation of established backends is **not yet demonstrated**. Do not start it until an acceptable independent textured environment exists.

## Completed baseline evidence and final decision questions

The compact [measured benchmark report](research/BASELINE_RESULTS.md) and per-scene JSON in `Documentation/research/` preserve lightweight evidence outside ignored output folders. Full logs, command arrays, resource samples, source hashes, depth maps, clouds and raw meshes remain in `outputs/research-baseline/`. Review `reports/rural-visual-review.jpg` and `reports/urban-visual-review.jpg`; the last three rows show Poisson at its own full bounds, while the earlier mesh rows share the union of both raw surface bounds. Camera/cloud panels have their own labelled full bounds. No favorable crop or selected camera subset is hidden.

1. **Can established software recover a meaningful recognizable scene?** Yes, recognizable photographic geometry is demonstrated: Rural fields, roads and low structures; Urban gardens, road junctions and high-rise buildings. This is not complete scene recovery. Rural Poisson has 404,968 triangles and 99.94% in its largest edge-connected component; Urban has 620,132 and 93.87%. Holes, small components and positive self-intersection probes remain. Human acceptance is pending.
2. **Does it outperform the native pipeline?** Practical full-scene registration and visual coherence improve markedly relative to historical fragmented native releases. An equal-input causal superiority claim is **not established**: native releases use 40 selected images versus 190/161 originals here, and different dense resolution/footprint/reference masks. The new all-point centred DSM P95 is 1.688 m Rural and 25.834 m Urban; the Urban tail prevents a blanket accuracy claim. These are offset-adjusted height-reference residuals, not independent accuracy. New matched-input/common-footprint experiments remain required.
3. **What remains valuable?** Keep desktop/job orchestration, provenance, deterministic evaluation, export validation and the archived native comparison. Replace production SfM/MVS/fusion/surfaces with established backends; see the component-by-component [audit](PIPELINE_AUDIT.md). No demonstrated native scientific novelty is inferred from engineering effort.
4. **What gap remains?** Native deficiencies are demonstrated. A limitation of established software that a new scheduler uniquely addresses is not yet demonstrated. Broad partial-model/quality/budget-driven selection strongly overlaps prior work; see [comparison](PRIOR_WORK_COMPARISON.md).
5. **What proposed contribution?** Only a hypothesis: calibrated marginal valid-area gain with correlated evidence, measured action cost, feedback and abstention. The broad direction is mostly established; this narrower candidate is potentially differentiated but unresolved. It is not an implemented or certified original method.
6. **What would prove/disprove it?** Held-out scenes with measured reference, quality-budget curves, full/uniform/random/geometric-greedy and direct-prior-work selectors, dependence/feedback/cost/abstention ablations; all overhead included. Predeclared 25/50/75% budgets and a proposed 5 percentage-point valid-completeness gain at 50% with at most 1 point precision loss. See [hypothesis](RESEARCH_HYPOTHESIS.md). Failure against a strong simple selector rejects the claim.
7. **Shortest responsible graduation route?** Review these assets; resolve Urban holes/outliers and reference validity; perform one matched-input comparison; finish close-competitor review and obtain measured-reference data; integrate the established backend through a small adapter; only then decide on a bounded research experiment. Do not write another MVS engine. See [architecture and milestones](PROPOSED_ARCHITECTURE.md).

OpenMVS was not run: no executable was found, and a source build would need an unverified toolchain/dependency installation. The comparison remains open, not a loss or win for either backend. Literature contains 44 deduplicated screened records, not the requested aspirational 200–300. The initial targeted review is substantive for overlap detection but insufficient for an exhaustive novelty claim.

The Urban all-point DSM tail is not synonymous with wrong geometry: facades can legitimately lie below a rooftop DSM. A separately reported fixed 1 m upper-envelope diagnostic gives centred P95 6.386 m over 269,220 occupied Urban cells and 2.555 m over 393,868 Rural cells. It is sensitive to high outliers and sampling density; it does not replace the all-point statistic. Minimum-three-point variants cover smaller footprints and are separately labelled. None certifies 3D precision, completeness or absolute accuracy.

Local checkpoint limitation: `git add` failed because this session cannot create `.git/index.lock` in the read-only Git directory. Verified files remain in the working tree; **no local commit and no push** occurred. Unrelated pre-existing changes remain untouched.

## Replay

From the repository root:

```powershell
.venv/Scripts/python.exe -B scripts/research_baseline.py audit
.venv/Scripts/python.exe -B scripts/research_baseline_diagnostics.py acquisition
.venv/Scripts/python.exe -B scripts/research_baseline.py sfm --scene rural
.venv/Scripts/python.exe -B scripts/research_baseline_diagnostics.py sfm --scene rural
.venv/Scripts/python.exe -B scripts/research_baseline.py dense --scene rural
.venv/Scripts/python.exe -B scripts/evaluate_research_baseline.py --scene rural
.venv/Scripts/python.exe -B scripts/export_research_texture.py --scene rural
.venv/Scripts/python.exe -B scripts/export_research_texture.py --scene rural --method poisson
```

Repeat scene arguments for Urban. The converter intentionally refuses to overwrite an existing asset directory. Blender commands and full replay sequence are in `outputs/research-baseline/reports/REPRODUCE.md`. Successful reconstruction stages are skipped when their success JSON exists; use `--output-root` with a fresh directory for independent replay or parameter changes. All datasets/assets stay outside Git. Keep lightweight command and metric evidence with the local checkpoint.
