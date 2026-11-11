# UAV reconstruction demonstration

Start the installed app from the repository root.

```powershell
.venv\Scripts\python.exe -m vision3d.demo
```

## Live reconstruction

1. Choose **Files…** or **Folder…** and select overlapping UAV photos. A calibrated camera JSON is optional. Automatic calibration records its assumptions in `intrinsics.json`.
2. Set **Multi-view** to `uav-recovery` to demonstrate the benchmarked recovery option. Select **Colored mesh** and **Photo GLB + OBJ**. **GLB export** separately requests a vertex-colored export.
3. Choose an output location and start reconstruction. The log follows validation, camera registration, dense stereo, surface optimization and texture export. Registration can be partial. Inspect the reported camera count and completed stage.
4. When **Photographic GLB and OBJ ready** appears, open **Open UV GLB**. Left drag orbits, the wheel zooms, and I/J/K/L pan. Press F or R to fit and reset the view. The PLY viewer also offers camera overlays where the pose metadata is available.
5. Choose **Open Output Folder** to locate the photo bundle. Keep `textured.obj`, `textured.mtl` and `atlas.jpg` together. The GLB embeds its atlas.

The 12-image desktop validation completed in 53.15 seconds with four registered cameras. The 40-photo no-cache CLI validation took 376.48 seconds. Use a short overlapping set or show a saved result during a time-limited presentation.

## Saved demonstration assets

Use **Open UV GLB** to choose `outputs/final-release/release-candidate/urban/textured.glb` for a compact first preview. Rural and Rural-80 assets are in sibling folders. Introduce these as saved results. To show that the files import independently, use the top and oblique Blender renders under `validation/<scene>/glb/` and the reports in the same folders. Point out real coverage gaps in the Rural render.

The release manifest lists local asset paths, byte hashes and verification reports. The full-size assets are kept in the local output directory. If the release directory is absent on another machine, rebuild it from local reconstruction results using the commands in the [release evaluation](FINAL_RELEASE_REPORT.md).

## CLI demonstration

```powershell
.venv\Scripts\python.exe -m vision3d.cli reconstruct-asset --input datasets/uav-photos --profile uav-recovery --obj --output outputs/uav-live-demo
.venv\Scripts\python.exe -B scripts/verify_uav_workflow.py --run outputs/uav-live-demo --output outputs/uav-live-check.json
```

Point `--input` at the folder of photos you intend to use. For a failed late export, the existing dense directory can be reused in a new run.

```powershell
.venv\Scripts\python.exe -m vision3d.cli reconstruct-asset --dense-dir outputs/uav-live-demo/dense --obj --output outputs/uav-resumed
```

In the desktop, **Cancel** stops the active job and keeps completed stages. An incomplete output directory can contain logs or partial files. The CLI records completion in `workflow.json`.
