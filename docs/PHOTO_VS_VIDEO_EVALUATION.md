# Photo versus video evaluation

## Darth Vader comparison status

No new Darth Vader MP4 was supplied or found in the checked Vader, Desktop and project input locations. The 30 original photographs remain the real-world reference. The Castle MP4 below is an encoded photo montage used to exercise the software. Its results do not measure the benefit of continuous video acquisition.

| Vader metric | Enhanced photo run | Continuous video run |
|---|---:|---|
| Inputs / registered | 30 / 5 (16.7%) | Pending capture |
| Verified edges / components | 38 / 5 | Pending |
| Tracks spanning at least three views | 311 | Pending |
| Sparse landmarks | 645 | Pending |
| Median / RMS reprojection px | 0.384 / 0.686 | Pending |
| Median triangulation angle | 8.49° | Pending |
| Sparse engine time | 11.20 s with warm match cache | Pending |
| Fast dense points / faces | 25006 / 46851 | Pending |
| Visual result | Fragmented visible surfaces with background and platform | Pending |

Use the same processing resolution, calibration policy, enhanced verifier, descriptor ratio, seed probing and Fast dense profile. Cap the video at 30 keyframes for the exhaustive comparison. Report actual keyframe count and register percentage, graph components, multi-view track count, sparse reprojection fit, triangulation angle, cache state and separate extraction/SfM/dense times. For qualitative completeness, inspect front, rear and sides for visible helmet and stand coverage, holes, background leakage and isolated fragments. Compare scene-wide outputs before evaluating masks. Capture a fixed-focal, well-lit orbit with overlap across elevation changes. Reprojection and point counts alone do not establish absolute pose or surface accuracy. [Vader photo experiments](DARTH_VADER_EXPERIMENTS.md)

## Development verification

The fixture contains six local Castle stills, each held for 12 frames. The encoded 1600×1202 MP4 reports 72 frames at 12 fps. The selector scored 24 frames, rejected 18 near duplicates and retained six views at indices 0, 12, 24, 36, 48 and 60. The Tk run took 2.23 s to extract them. Calibrated and approximate runs used the same keyframes.

| Configuration | Registered | Verified edges / components | Multi-view tracks | Sparse points | Median / RMS px | Ray angle | Sparse s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated, ordinary photo schedule on the keyframes | 6/6 | 11 / 1 | 1521 | 3504 | 0.367 / 0.564 | 8.95° | 18.21 |
| Calibrated, temporal schedule with discovery | 6/6 | 15 / 1 | 1631 | 3626 | 0.367 / 0.577 | 9.19° | 21.48 |
| Estimated video focal prior, temporal schedule | 5/6 | 15 / 1 | 1631 | 3042 | 0.336 / 0.572 | 12.71° | 20.08 |

The temporal configuration evaluated the remaining pairs on this short fixture and improved track support, while RMS increased on a different set of accepted observations. This does not establish that nonlocal discovery alone improved pose quality.

The complete calibrated Tk run produced 553531 dense points and 1185207 faces in 8.64 s of dense processing. A programmatic Tk harness selected the MP4 through the file-selection handler, started the worker, and opened the saved mesh in Open3D. The estimated-intrinsics run produced 386556 points and 881880 faces in 5.51 s. Both PLY and GLB outputs were reloaded by the existing exporter. Castle has no reference pose or surface for assessing absolute geometry.

The full suite passed 145 tests with 14 pre-existing dependency warnings. Synthetic MP4 tests cover decoding, duplicate and blur rejection, bounded temporal coverage, calibration and scaling, corrupt input, cancellation, temporal matching options and desktop dispatch. Castle and Temple calibrated photograph regressions retained identical pose matrices, sparse PLY and landmark JSON. Motorcycle reproduced 859 sparse points, 268359 dense points and 520582 triangles with validated PLY reload.

```powershell
.venv\Scripts\python.exe scripts/castle_video_fixture.py --output outputs/castle-video-fixture
.venv\Scripts\python.exe -m vision3d.cli keyframes --video outputs/castle-video-fixture/castle-photo-montage.mp4 --camera outputs/castle-video-fixture/castle-video-calibration.json --output outputs/castle-video-keyframes
```

Continue with the sparse and dense commands in the [video workflow](VIDEO_RECONSTRUCTION.md), substituting the fixture keyframe directory. Local evidence lives under `outputs/video-investigation/`, including the calibrated GUI run, approximate results and calibrated benchmark reruns.
