# TASK 02 — calibrated sparse two-view reconstruction

Date: 2026-11-04, Europe/Istanbul. Status: a genuine calibrated real-image pair produced a validated colored sparse point cloud. This is **two-view reconstruction**, not full multi-view SfM.

## Audit, scope and architecture

Started from clean commit `0774c6594e94f19c0fb05d403ec7a630379174f0`. Reviewed the existing input, preprocessing, SIFT/ORB, matcher, CLI, tests, configuration and TASK01 report. Reused all image preprocessing and feature/matching machinery; the `analyze` command remains operational. No new dependencies were needed; Python 3.12.14 and the pinned environment from TASK01 remain in use.

Added:

- `geometry/camera.py`: camera validation, JSON loading, explicit approximations, pixel-center scaling and undistortion.
- `geometry/two_view.py`: robust essential estimation, rotation-only/homography diagnostics, pose recovery, triangulation and rejection gates.
- `reconstruction/pair.py`: image-pair orchestration, metadata/tracks, original-image color sampling, Open3D PLY export/reload and static preview.
- `configs/motorcycle.json`: per-view published intrinsic calibration, containing no reference poses or baseline magnitude.
- `scripts/motorcycle_experiment.py`: optional checksum-verified two-image download, reproducible reconstruction and separate reference-pose evaluation.
- New unit/integration tests, extended CLI and README.

No multi-view reconstruction, bundle adjustment, dense cloud, mesh, neural network, web interface, Unity integration or CUDA processing was added.

## Camera coordinates and calibration assumptions

OpenCV pinhole convention: x right, y down, z forward, image pixels `(u,v)` measured from the top-left pixel center. World coordinates equal camera 1 coordinates. Projection matrices in normalized coordinates are `P1=[I|0]` and `P2=[R|t]`; `X2=R X1+t`. World-to-camera poses are exported as 4x4 matrices. Camera 2 center in the world is `-R.T @ t`, not `t`. Translation has unit norm, so the baseline is 1 and cloud coordinates have arbitrary scale. The sign is selected by cheirality; no physical translation magnitude can be recovered from this monocular geometry.

The camera model stores width/height, fx/fy/cx/cy, optional standard OpenCV distortion coefficients, source, approximation flag and assumptions. It exposes `K`. Dimensions must be integers >=2, focal lengths positive, values finite, and principal points inside the image. Distortion vectors accept OpenCV lengths 4, 5, 8, 12 or 14 (or empty); fisheye calibration is unsupported. JSON may specify one camera or a camera1/camera2 pair. An optional redundant K is checked for agreement. This milestone loads and validates calibration; it does not implement checkerboard calibration acquisition.

Calibration must describe the original **EXIF-oriented** image and match its dimensions. Raw sensor-coordinate calibration is explicitly unsupported rather than silently transformed; converting it requires matching image orientation, optical axes and distortion conventions. Preprocessing applies EXIF before feature extraction. Each resize records `sx,sy` and `T`, with `u'=sx*(u+0.5)-0.5`; the camera model applies `K'=T K`. Distortion coefficients remain unchanged. Rounding of dimensions is reflected independently in sx/sy. The model supports separate intrinsics and image dimensions for the two views.

Approximation is opt-in through CLI `--approximate-intrinsics` with exactly one original-pixel focal length or horizontal FOV. JSON explicitly marked `approximate: true` is also supported, with required assumptions. Guesses assume square pixels, a centered principal point and zero distortion. They warn and are recorded in metadata. A passed quality gate does not establish that guessed calibration is accurate.

## Estimation and threshold units

Use the existing SIFT/ORB extractor and reciprocal Lowe-ratio matches. The existing fundamental-matrix result is preserved as a diagnostic. Essential estimation deliberately starts from reciprocal descriptor matches rather than only fundamental inliers: a fundamental model fit directly to distorted pixels can incorrectly exclude valid lens-distorted observations.

Each camera uses `cv2.undistortPoints` with its own K and distortion, producing normalized observations. `findEssentialMat` receives identity intrinsics and RANSAC threshold `essential_threshold_px / min(fx1,fy1,fx2,fy2)`. The default is a conservative shared virtual-pixel tolerance, **not** a promise of identical pixel error bounds under different focal lengths or nonlinear distortion. Homography and rotation-only diagnostics use the same virtual coordinate convention. Actual post-triangulation reprojection errors are separately computed in each processed distorted image with `cv2.projectPoints`, and filtered in real pixels.

Essential candidates are checked for finite values, shape, rank-two structure and equal leading singular values. `recoverPose` chooses cheirality-consistent R/t from robust inliers; if multiple essential candidates are returned, the greatest pose-inlier count wins. R is checked for orthonormality and determinant +1 before triangulation. The translation direction is normalized. Seed 0, one OpenCV thread and a 5000-iteration RANSAC cap are recorded; exact reproducibility is scoped to the same software/build.

Reference: [OpenCV calibration and two-view geometry documentation](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html).

## Degeneracy and triangulation gates

Defaults are immutable `Quality` parameters, recorded in each result. No thresholds were weakened to obtain the real cloud.

| Gate | Default |
|---|---|
| Minimum distinct, non-collinear input correspondences | 30 |
| Minimum essential inliers / pose inliers | 20 / 20 |
| Essential RANSAC virtual-pixel tolerance | 1.0 px |
| Robust estimation confidence | 0.999 |
| Homography RANSAC tolerance | 2 times normalized essential tolerance |
| Reject homography support among essential inliers | >=90% |
| Reject rotation-only bearing alignment support | >=90% within 2 virtual pixels |
| Minimum pose cheirality ratio | 60% of essential inliers |
| Homogeneous denominator magnitude | >1e-12 |
| Positive depth in both cameras | >1e-9 |
| Maximum distance from either camera | <1000 unit-baseline lengths |
| Maximum reprojection error | <=2.0 px in **each** processed distorted image |
| Minimum angle between triangulation rays | >=1.0 degree per retained point |
| Minimum valid filtered cloud | 20 points and >=50% of pose-triangulated points |

The rotation-only model uses SVD alignment of unit bearings, with its residual/support recorded. Homography overlap flags planar or rotational ambiguity. A zero physical baseline cannot be measured by checking recovered `||t||`: essential decomposition already normalizes it. Rotation-only support, homography dominance, cheirality, point bounds and measured parallax provide the observable safeguards instead.

`cv2.triangulatePoints` operates on undistorted normalized observations and consistent projection matrices. Homogeneous conversion rejects near-zero denominators; non-finite, negative-depth, excessive-distance, reprojection and low-angle points are filtered separately. Tracks retain each pose candidate's point, errors, angle and filter flags, along with the original correspondence index. Only retained points are exported. Rejection exits with code 3 and writes diagnostics without a PLY; input/configuration/I/O failures exit 2; success exits 0. A failed run may leave partial diagnostics, and output directories must be new.

## Real-world data and permission

Middlebury **2014 Motorcycle**: a nonplanar, static photographed scene, distributed as a quarter-resolution rectified stereo pair by scikit-image. Downloaded on 2026-11-04 from its official pinned v0.25.2 repository, only **two images, 1,285,074 bytes total**. No full dataset archive, disparity map, reference pose file, or training data was downloaded.

| Image | Size | SHA-256 |
|---|---:|---|
| motorcycle_left.png | 644701 bytes; 741x500 | db18e9c4157617403c3537a6ba355dfeafe9a7eabb6b9b94cb33f6525dd49179 |
| motorcycle_right.png | 640373 bytes; 741x500 | 5fc913ae870e42a4b662314bc904d1786bcad8e2f0b9b67dba5a229406357797 |

Sources:

- [Left image](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/motorcycle_left.png), [right image](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/motorcycle_right.png).
- [scikit-image's dataset description and quarter-resolution calibration](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py#L981).
- [Middlebury source calibration](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Motorcycle-perfect/calib.txt).
- [Middlebury usage permission and citation requirements](https://vision.middlebury.edu/stereo/data/): the provider permits image/disparity use and publication and requests the relevant paper citation. This is a provider permission statement, not a claimed Creative Commons license.

Citation: D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang and P. Westling, *High-resolution stereo datasets with subpixel-accurate ground truth*, GCPR 2014. The scene was created at Middlebury College. Images remain ignored and uncommitted.

Calibration used: fx=fy=994.978, cy=254.877, cx1=311.193, cx2=342.279. These are published rectified-image values; zero distortion is appropriate for the distributed rectified images. The original physical baseline is **not used** to reconstruct or scale the cloud. Rectified reference `R=I`, translation direction `(-1,0,0)` is evaluated only after pose recovery in the experiment script, never passed to the reconstruction routine. This benchmark validates known-intrinsic geometry; it does not establish performance on arbitrary unrectified phone images.

## Reproduction and outputs

From the installed repository:

```powershell
$env:MPLCONFIGDIR="$PWD\outputs\matplotlib-cache"  # optional for restricted cache access
.venv\Scripts\python.exe scripts/motorcycle_experiment.py --download --output outputs/motorcycle-run --preview
.venv\Scripts\python.exe scripts/motorcycle_experiment.py --output outputs/motorcycle-small --max-dimension 600
.venv\Scripts\vision3d.exe reconstruct-pair --image1 datasets/middlebury-motorcycle/motorcycle_left.png --image2 datasets/middlebury-motorcycle/motorcycle_right.png --camera configs/motorcycle.json --detector orb --output outputs/motorcycle-orb-run
```

The script verifies hashes even for cached input. Downloads use two fixed URLs, bounded reads and hash checks before saving. It writes `evaluation.json` separately from the processing metrics. The regular CLI requires no benchmark-specific knowledge.

Actual artifact locations:

- `outputs/motorcycle-final/`: canonical SIFT cloud, preview and evaluation.
- `outputs/motorcycle-resized-final/`: independently resized SIFT cloud and evaluation.
- `outputs/motorcycle-orb-final/`: ORB cloud, full geometry metrics and reference evaluation.
- `outputs/motorcycle-identical-calibrated/`: same-photo negative control, one camera model for both views; rejected as rotation-dominant, exit 3, no PLY.

Each successful directory contains `sparse.ply`, `poses.json`, `intrinsics.json`, `manifest.json`, `metrics.json` and `correspondences.npz`. The experiment adds `evaluation.json`; `--preview` adds `sparse_preview.png`. Original inputs are hashed/referenced, not copied. RGB colors sample the original EXIF-oriented first image after inversion of the resize pixel-center transform. Open3D exports and reloads each PLY, checking point coordinates, count and color quantization. No placeholder geometry is exported.

The PLY vertex ordering is indexed by NPZ `ply_observation_indices`. `pixels1/2` hold processed distorted coordinates; `feature_indices` join the detector keypoints. `essential_mask` and `pose_mask` cover all reciprocal matches. `pose_indices` maps the triangulated array into those matches; `valid_pose_points`, reprojection errors, angles, positive-depth and bound flags explain filtering. Original observations can be recovered with the inverse manifest resize transform. Invalid candidate coordinates may be NaN in NPZ, but exported PLY points are finite.

## Observed results

The first real pair succeeded without threshold changes; no failed pair was replaced to force output. Repeated SIFT runs preserved all counts and geometry. The canonical run includes preview/export/reload; timings are single observations, not a controlled hardware comparison.

| Measurement | SIFT original | SIFT resized | ORB original |
|---|---:|---:|---:|
| Processed dimensions | 741x500 | 600x405 | 741x500 |
| Keypoints image 1 / image 2 | 2650 / 2588 | 2060 / 2042 | 5987 / 6000 |
| Raw nearest descriptor matches | 2650 | 2060 | 5987 |
| Ratio-filtered matches | 985 | 807 | 1553 |
| Mutual matches / essential input | 894 | 740 | 1238 |
| Fundamental inliers (diagnostic) | 850 | 726 | 1154 |
| Essential inliers | 859 | 724 | 1123 |
| Recovered-pose inliers | 859 | 724 | 1123 |
| Triangulated / finite / positive-depth | 859 / 859 / 859 | 724 / 724 / 724 | 1123 / 1123 / 1123 |
| Valid filtered points | **859** | **724** | **1123** |
| Homography overlap among essential inliers | 0.394645 | 0.399171 | 0.615316 |
| Median reprojection error, px | **0.086653** | **0.070623** | ~1.17e-13 |
| Median triangulation angle, degrees | **3.859673** | **4.443978** | **4.447359** |
| Runtime, seconds | 1.609892 | 0.941163 | 1.171107 |
| Peak working set, bytes | 149504000 | 112336896 | 235581440 |
| Reference rotation error, degrees | 0.249754 | 0.259860 | ~0 (rectified sample) |
| Reference translation-direction error, degrees | 0.476587 | 0.752342 | ~0 (rectified sample) |

Median reprojection error pools both views of retained points; each point independently passes both <=2px gates. ORB's tiny median is expected for many exact same-row matches in this already rectified sample; it is **not** evidence of negligible 3D error or general camera accuracy. The uncalibrated F diagnostic may have large scale-dependent coefficients in such stereo configurations and does not drive pose estimation. The canonical rotation-only support is 0.001164 and cheirality/valid ratios are both 1.0. Camera translation has unit norm.

Runtime starts before image loading and includes feature/matching, estimation, output construction, Open3D import/export/reload and optional preview; it excludes interpreter startup and the last metrics write. Memory is the Windows process-lifetime peak working set, includes interpreter/import overhead and is null if unavailable elsewhere. No GPU acceleration is used. The requested minimum laptop was not available for separate testing. No scale or absolute depth accuracy was measured.

The canonical preview was visually inspected: colored points occupy several depth layers with two unit-separated camera centers, and there is no synthetic fill or mesh. View the local artifact at `outputs/motorcycle-final/sparse_preview.png`. Open3D can display the PLY with `visualization.draw_geometries` on a graphical desktop. Static preview generation uses Matplotlib to support headless Windows execution; no application UI was built.

## Automated validation

Final command: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-delivery`. Result: **45 passed, 0 failed, 0 skipped in 1.73 seconds**, with no warnings. `uv pip check` confirmed all 75 installed distributions remain compatible. The original `analyze` CLI was run on the same two real photographs and processed both successfully. Both module and installed executable reconstruction entry points were exercised.

Tests cover camera matrices, JSON consistency and malformed input, resize/undistortion equivalence, explicit guesses, invalid numerical values, synthetic nonplanar geometry with distortion and unequal cameras, noisy/outlier essential estimation, rotation and signed translation direction, triangulation accuracy, positive depth, finite point handling, reprojection errors, extreme distances, insufficient/duplicate observations, pure rotation, low parallax and planar rejection. Integration tests rasterize textured patches at several depths, execute the full reconstruction CLI after resizing, validate tracks and PLY, and check rejected-run exit codes/no cloud. All earlier analyze/image/video tests are retained.

Real-world validation is separate from these synthetic tests. No reference camera pose is needed by production processing. No new package installation or dependency conflict occurred. A restricted Matplotlib cache warning in the first preview was resolved by setting a writable output cache for subsequent runs. Pytest uses normal temporary-directory access in this environment.

## Limitations and next stage

Quality gates are conservative heuristics, not proof against every false match, rolling shutter, moving object, lens-model error or weakly constrained scene. Homography dominance deliberately rejects some valid planar pairs. Estimated intrinsics can produce internally consistent yet biased structure. Standard undistortion has finite numerical precision/iteration behavior. Inlier masks and tracks allow diagnosis, but there is no subpixel feature refinement or nonlinear pose/structure refinement yet. Only two views and a limited feature set produce a sparse partial scene.

EXIF raw-calibration conversion, fisheye models, automatic checkerboard calibration, absolute scale, multi-view track management and minimum-machine benchmarking remain unsupported. The old video's nominal timestamps/decoder limitations remain unchanged. Open3D's pre-existing Dash/Flask transitive dependencies remain installed; they are unused by a server.

Recommended next stage: validate static unrectified photographs with measured intrinsics, create persistent observation tracks and select well-conditioned initial pairs, then implement incremental SfM with PnP pose registration and joint bundle adjustment in its own milestone. Retain this two-view quality and reference-evaluation baseline before extending to dense reconstruction or meshing.
