# Reconstruction baseline

Baseline checkout `7d98d1d`. Existing edits and deletions in the presentation deliverables and untracked depth captures are outside this work.

The Python 3.12 environment uses the pinned OpenCV, NumPy, SciPy, Pillow and Open3D stack. `vision3d.exe` invokes the console CLI. `run_demo.bat` launches `python -m vision3d.demo`, whose Tk event loop delegates reconstruction to a child CLI process and viewing to a separate Open3D process. Image loading applies EXIF orientation. SIFT/ORB feed reciprocal descriptor matching, verified tracks, two-view initialization, PnP, triangulation and optional transactional local optimization. Colored PLY is written and reloaded for validation. The pair engine exports a distinct camera/metrics schema from sequences.

Fresh runs from the existing local datasets, with unchanged thresholds and no downloads:

| Case | Registered | Points | Median/RMS px | Engine seconds | Peak working set |
|---|---:|---:|---|---:|---:|
| Castle 6, local BA | 6/6 | 3915 | 0.285393 / 0.586640 | 16.03 | 563.6 MiB |
| Temple 30, local BA | 18/30 | 2664 | 0.260683 / 0.653783 | 9.18 | 199.5 MiB |
| Motorcycle pair | 2/2 | 859 | 0.086653 / — | 1.06 | 138.0 MiB |

Temple median aligned rotation and translation-direction errors were 6.613361° and 2.844933°. Motorcycle reference rotation/direction errors were 0.249754° and 0.476587°. Reports are under ignored `outputs/completion-before*`. The full existing suite passed (122 tests, 14 dependency deprecation warnings).

Priorities are to initialize each image's intrinsics independently, route desktop pairs through the existing pair engine, preserve calibrated behavior, and validate outputs and failure paths before adding new geometry stages. EXIF is currently used for orientation but not focal initialization. The desktop requires JSON and routes pairs through the sequence interface. Temple coverage and drift remain open, and graph recovery/joint BA are experimental because historical reference errors worsened. Dense stereo, surface reconstruction and texturing have no established pipeline. SIFT matching is the largest measured Castle stage, while Temple spends more time in bounded BA. Optimization needs controlled measurements.
