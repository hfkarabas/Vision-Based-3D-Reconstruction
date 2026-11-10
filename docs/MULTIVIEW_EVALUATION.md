# Multi-view evaluation

Runs used local datasets on Windows with an Intel Core Ultra 5 245KF and the pinned Python 3.12, OpenCV 4.11 and Open3D 0.19 environment. The dense worker uses one OpenCV thread and requires no GPU or external MVS executable. Memory is the child process lifetime peak working set. The measurements below are individual runs. No dense ground truth was evaluated.

## Sparse regression

| Scene | Registered | Sparse landmarks | Median / RMS px | Engine seconds before → after |
|---|---:|---:|---|---:|
| Calibrated Castle | 6/6 | 3915 | 0.285393 / 0.586640 | 17.75 → 18.36 |
| Calibrated TempleRing 30 | 18/30 | 2664 | 0.260683 / 0.653783 | 10.16 → 10.95 |

Sparse PLY, poses and landmark JSON were byte-identical in both reruns. Temple reference pose errors were 6.613° median aligned rotation and 2.845° median relative translation direction. Experimental graph recovery registered 23/30 and took 12.52 s, with 0.333 px median and 0.725 px RMS. On all registered recovery cameras, reference errors were 12.736° and 5.048°. On the **same 18 cameras**, they were 7.453° and 3.665°, still worse than baseline. Recovery therefore remains opt-in. No camera optimization thresholds changed.

## Castle profile comparison with mesh disabled

| Profile | Reference maps | Checked / raw depth samples | Dense points after fusion | Depth / fusion / total s | Peak MiB |
|---|---:|---|---:|---|---:|
| Fast | 6 | 703847 / 896223 | 554068 | 0.90 / 0.96 / 2.81 | 392.8 |
| Balanced | 5 | 1195687 / 1560115 | 982658 | 1.15 / 1.47 / 3.69 | 545.0 |
| High | 5 | 2001519 / 2643352 | 1670790 | 1.99 / 2.90 / 6.18 | 817.8 |

Higher resolution produced more samples but exceeded the disparity search bound for Castle reference image 0. Support counts describe agreement under estimated cameras. They do not measure surface accuracy.

## End-to-end scenarios

| Scenario | Registered | Dense points | Mesh faces | Dense stage seconds | Export verification |
|---|---:|---:|---:|---:|---|
| Calibrated Motorcycle stereo | 2/2 | 268359 | 520582 | 1.10 | Colored PLY cloud and mesh reloaded |
| Calibrated Castle, Fast | 6/6 | 554068 | 1194432 | 7.85 | PLY and GLB reloaded |
| Calibrated Temple 30, Fast | 18/30 | 513500 | 1055314 | 6.96 | PLY and GLB reloaded |
| Castle EXIF, Fast | 6/6 | 552271 | 1188157 | 6.53 | PLY reloaded |

EXIF Castle used the existing photographs with no supplied JSON calibration. It had 3911 sparse landmarks. This checks the automatic workflow on a real sequence, with coverage limited to one EXIF camera and scene. The Tk harness also ran calibrated Castle from start through dense, mesh and GLB, switched between cloud results and opened the actual mesh with camera overlays. The approximate output has no independent geometry reference.

An initial Temple attempt generated zero maps because candidate pairs rectified vertically. A transpose of images, rectified projection and Q coordinates enabled vertical disparity. The known-depth synthetic test verifies reconstruction in the original world frame. A trial Open3D-written GLB failed to reload because of a buffer-view range error. The replacement writer emits explicit glTF accessors and validates exported triangle positions, normals and colors through Open3D. An early quadric-decimation trial requested 300000 faces and returned 436302 in 39.68 s total, so simplification remains best-effort and opt-in. The Castle mesh shows a recognizable façade with holes and some sky artifacts. It covers visible surfaces only.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts/benchmark_sfm.py --output outputs/mv-sparse-check --case castle6 --case temple30
.venv\Scripts\python.exe scripts/benchmark_multiview.py --scene castle=outputs/mv-sparse-check/castle6 --output outputs/mv-profile-check
.venv\Scripts\python.exe scripts/benchmark_multiview.py --scene temple=outputs/mv-sparse-check/temple30 --output outputs/mv-temple-check --profile fast --mesh --glb
.venv\Scripts\python.exe scripts/benchmark_sfm.py --suite configs/recovery_experiments.json --case temple30_graph --output outputs/mv-recovery-check
```

The harness records fresh-process commands, stage metrics, profile settings, environment, sparse results, reference pose metrics where present, logs and export status. Original local records are under ignored `outputs/mv-baseline`, `outputs/mv-final-sparse`, `outputs/mv-profile-final`, `outputs/mv-castle-validated`, `outputs/mv-temple-bench-final`, `outputs/mv-castle-exif` and `outputs/mv-desktop`.
