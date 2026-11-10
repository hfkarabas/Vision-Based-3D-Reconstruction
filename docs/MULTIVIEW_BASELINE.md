# Multi-view reconstruction baseline

Checkout `af031709751fcc93939c141d16a5a1da9cc356a7`. Existing presentation and depth-capture changes are outside this work. The Windows Python 3.12 environment uses pinned OpenCV 4.11, Open3D 0.19, NumPy and SciPy. The baseline ran on an Intel Core Ultra 5 245KF. No GPU processing was required.

The console entry point is `vision3d.cli:main`. The desktop entry point is `python -m vision3d.demo` or `run_demo.bat`. Tk runs reconstruction and Open3D viewing in child processes. Sequence output contains source paths and hashes, original and resized per-image intrinsics, world-to-camera matrices, colored landmarks with observation maps, and sparse PLY. Pair output has a separate metadata schema. The experimental stereo pipeline uses calibrated rectification, bidirectional SGBM, and a disparity-grid surface. It has no third-view consistency and cannot serve as generalized fusion.

Fresh local runs used existing Castle, TempleRing and Motorcycle images. The full suite passed 132 tests with 14 dependency warnings.

| Case | Cameras | Sparse points | Median / RMS px | Engine s | Peak MiB |
|---|---:|---:|---|---:|---:|
| Castle 6 | 6/6 | 3915 | 0.285393 / 0.586640 | 17.75 | 566.6 |
| Temple 30 | 18/30 | 2664 | 0.260683 / 0.653783 | 10.16 | 198.2 |
| Motorcycle | 2/2 | 859 | 0.086653 / 0.180480 | 1.09 | 138.6 |

Motorcycle stereo produced 268359 points and 520582 triangles in 1.10 s. Temple reference errors remained 6.613361° median aligned rotation and 2.844933° median translation direction. Reproduce with `scripts/benchmark_sfm.py --output outputs/mv-baseline --case castle6 --case temple30`, then `scripts/motorcycle_experiment.py --output outputs/mv-baseline-motorcycle` and `vision3d dense-pair --pair-dir outputs/mv-baseline-motorcycle --output outputs/mv-baseline-stereo --mesh`. Use fresh output directories.

Priorities are independent multi-view consistency, bounded fusion, validated open surfaces, and desktop integration. Intrinsics refinement, generalized texture baking and improved reference-based registration accuracy were absent at baseline.
