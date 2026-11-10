# Darth Vader controlled experiments

All rows use the original 30 photos, the default seed, unchanged geometry/PnP acceptance gates and local BA plus fixed-camera point refinement. Rows use 1600 px longest side, reciprocal SIFT ratio 0.75 and EXIF estimates unless specified. Runtime is engine time on the local Windows desktop. Cache warmth and concurrent work differ, so use the timing column for practical cost, not controlled speed ratios. A multi-view track has at least three observations.

| Experiment | Edges / components | Multi-view tracks | Cameras / landmarks | Median / RMS px | Median ray angle | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Reproduced baseline | 14 / 16 | 15 | 2 / 73 | 0.140 / 0.274 | 13.40° | 32.02 |
| 30 appearance-ranked discovery attempts | 17 / 13 | 120 | 2 / 204 | 0.144 / 0.220 | 10.79° | 24.08 |
| Local MAGSAC | 19 / 11 | 43 | 2 / 82 | 0.215 / 0.352 | 12.89° | 7.49 |
| Exhaustive RANSAC | 27 / 10 | 204 | 2 / 204 | 0.144 / 0.220 | 10.79° | 66.85 |
| Exhaustive MAGSAC | 34 / 6 | 262 | 2 / 224 | 0.138 / 0.267 | 9.61° | 9.24 |
| Exhaustive MAGSAC, track-count seed | 34 / 6 | 262 | 2 / 224 | 0.138 / 0.267 | 9.61° | 8.12 |
| Same with 3000 px assumed focal | 34 / 6 | 262 | 2 / 228 | 0.141 / 0.265 | 10.70° | 8.58 |
| Exhaustive MAGSAC, 2400 px, track-count seed | 27 / 13 | 240 | 2 / 250 | 0.183 / 0.288 | 9.94° | 162.91 |
| Exhaustive MAGSAC, ratio 0.80, track-count seed | 38 / 5 | 311 | 2 / 258 | 0.186 / 0.286 | 9.37° | 70.45 |
| Exhaustive MAGSAC, PnP-probed seed | 34 / 6 | 262 | 3 / 405 | 0.386 / 0.620 | 7.90° | 7.96 |
| Same at 2400 px | 27 / 13 | 240 | 3 / 205 | 0.216 / 0.455 | 22.23° | 81.23 |
| Exhaustive MAGSAC, ratio 0.80, PnP-probed seed | 38 / 5 | 311 | 5 / 645 | 0.384 / 0.686 | 8.49° | 11.20 |

Exhaustive evaluated 352 additional pairs. The first 435-pair run had 28 cached descriptor evaluations and took about 67 s total. The MAGSAC rerun reused all 435 and took 9.24 s while still recomputing geometric verification. Loosening only the reciprocal descriptor ratio was tested to allow less distinctive views to reach robust geometry. It did not change any geometry, PnP or spatial acceptance threshold. Alone it did not grow the map. Combining it with seed probes selected 12/23 instead of 1/26. Cameras 22, 13 and 21 passed the unmodified PnP gates with 30, 22 and 21 final inliers, and with 17.7%, 18.0% and 8.7% image coverage. The remaining 25 views lacked sufficient 2D–3D support at termination. A 23-image verified graph component did not imply a 23-image reconstructable track component.

Reprojection error increased with the five-view map and is computed over a different observation set. This dataset has no reference poses or surface. The result demonstrates additional geometrically accepted support and a partial reconstruction, with absolute pose and surface accuracy unverified.

## Dense validation

| Input and mode | Result |
|---|---|
| Three-view probed result, Fast | Three depth maps, only 20 corroborated samples, rejected |
| Five-view result, Balanced | Three references exhausted the bounded disparity search, two maps remained with zero independent samples, rejected |
| Five-view result, Fast, scene-wide | Four fused references, 50,140 corroborated samples, 25,006 fused points, 46,851 faces, 2.19 s |
| Same five poses, Fast with manual object polygons | Three fused references, 27,223 corroborated samples, 20,628 fused points, 36,065 faces, 1.97 s |

Both Fast runs validated colored PLY and GLB by reloading. Coordinates use the sparse world frame in PLY, with the documented axis change in GLB. Open3D renders show scattered surface fragments. The unmasked run includes pieces of the platform and room. Masking improves focus on the helmet and stand but does not recover a complete or clean model. The polygons were manually traced in oriented image coordinates from five thumbnail views and applied after sparse pose estimation. This retains background support for registration. The source polygon file is saved in `outputs/vader-investigation/object-masks.json` and copied beside the masked result.

## Reproduce

```powershell
$images = 'C:\Users\furka\OneDrive\Desktop\vader'
$runs = 'outputs/vader-replay'
$common = @('--input', $images, '--auto-intrinsics', '--local-ba', '--refine-points', '--feature-cache', "$runs/features", '--pair-cache', "$runs/pair-cache")
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence @common --output "$runs/baseline"
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence @common --exhaustive --verifier magsac --probe-seeds --ratio 0.80 --output "$runs/enhanced"
.venv\Scripts\python.exe -m vision3d.cli dense-sequence --sparse-dir "$runs/enhanced" --output "$runs/dense" --profile fast --mesh --glb
.venv\Scripts\python.exe -m vision3d.cli dense-sequence --sparse-dir "$runs/enhanced" --output "$runs/dense-object" --profile fast --mesh --glb --object-masks outputs/vader-investigation/object-masks.json
.venv\Scripts\python.exe scripts/sfm_diagnostics.py --run "$runs/enhanced" --output "$runs/diagnostics" --keypoints --pair 1 26 --pair 12 23
```

Use fresh run directories. A baseline with only two cameras exits 3 and retains its partial outputs. The other ablations are obtained by setting `--recovery-pairs 30`, `--continuation-seed` in place of `--probe-seeds`, `--max-dimension 2400`, or replacing `--auto-intrinsics` with `--approximate-intrinsics --focal-px 3000` as indicated. [OpenCV calibration and fundamental estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
