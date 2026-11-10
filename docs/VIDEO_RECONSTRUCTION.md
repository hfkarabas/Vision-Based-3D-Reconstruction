# Video reconstruction

## Desktop workflow

Launch the existing app from PowerShell:

```powershell
& 'C:\projects\Vision-Based-3D-Reconstruction\run_demo.bat'
```

1. Choose **Video…** and select an MP4. The input line displays decoded dimensions, FPS and nominal duration.
2. Optionally choose **Camera JSON**. Supply one shared pinhole camera model at the decoded video resolution, including the orientation applied by OpenCV. If omitted, the generated model uses a centered principal point, focal length equal to the image diagonal, square pixels and zero distortion. This is an estimate.
3. **Video settings…** changes the default four scored samples per second and the 40-keyframe cap. Choose **Yes** on the overlap-aware prompt to prioritize verified continuity during budget reduction. **No** keeps the default time-uniform selection. Leave **Enhanced matching** off for the standard SfM schedule. Enable it to add the existing experimental MAGSAC, descriptor ratio and seed probing.
4. Select a dense profile such as **fast**, and optionally **Colored mesh** and **GLB export**, then start. **Cancel** stops the active worker. Sparse results already completed remain available if the dense stage is cancelled or fails.
5. Use **Switch Sparse / Dense**, **Open 3D View** or **Open Mesh**. The run contains a keyframe selection report, native-resolution PNGs, camera JSON, sparse metrics and any dense outputs.

The existing SfM and MVS pipelines consume the exported PNGs. Two usable views use the pair engine. Three or more use incremental SfM and local optimization. Multi-view dense fusion requires at least three registered views and consistent depth evidence. Photo, folder and calibrated demo workflows are unchanged.

## CLI reproduction

Run from the checkout and choose new output directories. `--camera` during extraction preserves a supplied calibration in the generated camera file. Without it, use the diagonal prior or pass `--fov-degrees` for a known horizontal field-of-view assumption.

```powershell
.venv\Scripts\python.exe -m vision3d.cli keyframes --video 'C:\path\orbit.mp4' --output outputs/video-frames --sample-hz 4 --max-keyframes 40 --strategy overlap-aware
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input outputs/video-frames/frames --camera outputs/video-frames/camera.json --max-images 50 --window 4 --long-gap 8 --recovery-pairs 12 --local-ba --refine-points --output outputs/video-sparse
.venv\Scripts\python.exe -m vision3d.cli dense-sequence --sparse-dir outputs/video-sparse --output outputs/video-dense --profile fast --mesh --glb
```

For exactly two retained frames, invoke `reconstruct-pair` with their PNG paths instead. For CLI enhanced matching use `--verifier magsac --probe-seeds --ratio 0.80`. Replace `--recovery-pairs 12` with `--exhaustive` for up to 30 frames, or `--recovery-pairs 30` above that. The default video schedule uses temporal offsets 1–4 and 8 plus up to 12 appearance-ranked nonlocal candidates. Discovery still requires geometric verification. Explicit masks are supported in the dense CLI through `--object-masks`, using the extracted filenames as keys.

The video input is read only. A successfully extracted run records its source hash and maps stable PNG names to nominal frame indices and times. Failed or cancelled stages can leave diagnostic logs or partial files. A valid result requires completion metrics. Preserve the extracted PNGs while using sparse outputs to run MVS because image hashes are verified.

## Limits

MP4 support depends on the local OpenCV codec backend. Clips need valid FPS, at least two decoded frames, stable dimensions and a declared duration of at most ten minutes. Frame timestamps use frame index divided by nominal FPS. Variable-frame-rate presentation timestamps are not recovered. Digital zoom, electronic stabilization, rolling shutter, moving objects and pure rotation can bias geometry or prevent registration. Full-resolution PNG output uses more disk space than the input. The frame budget is 2–50 to respect the existing SfM resource limit. See [selection details](VIDEO_KEYFRAME_SELECTION.md) and [evaluation](PHOTO_VS_VIDEO_EVALUATION.md).
