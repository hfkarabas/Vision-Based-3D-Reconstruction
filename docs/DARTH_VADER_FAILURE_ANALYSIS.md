# Darth Vader reconstruction failure analysis

## Reproduced input and baseline

All 30 original JPEGs at `C:/Users/furka/OneDrive/Desktop/vader` decoded successfully. The filename order and all 30 SHA-256 values matched the initial desktop run's manifest and the replay. No source photograph or EXIF record was edited. The original run was `outputs/demo/run_20261110T065247868089Z_2d11a62b/reconstruction`. The replay is `outputs/vader-investigation/baseline`.

The replay reproduced seed images 21 and 22, 2/30 registered cameras, 73 landmarks, 14 verified edges, 16 components, 541 tracks and 15 tracks spanning three views. Median/RMS reprojection error was 0.1404/0.2743 px. Runtime was 32.02 s compared with 27.34 s in the original run. The two-view result was labeled partial.

## Failure stages

- **Features.** The 1,191–3,591 keypoints per view were distributed across the helmet, stand, patterned orange platform and room. On six representative photos (0, 4, 9, 13, 21, 25), manually drawn loose object-and-stand rectangles contained about 67%, 80%, 37%, 62%, 62% and 54% of keypoints. These are approximate bounding-region counts, not segmentation labels. The remainder includes platform and room details. Repeated studs, glossy highlights and changed viewpoint can reduce descriptor distinctiveness. The overlays show that a total feature shortage is an inadequate explanation.
- **Scheduling.** The filename window tries offsets 1 and 2, plus offset 4. This produces 29 + 28 + 26 = 83 of 435 pairs. It misses revisits across capture height and orientation. The visually similar 1/26 pair, for example, was never tested by the baseline. The desktop default had a discovery budget of zero to preserve the stable calibrated workflow. Appearance-ranked bridging was tested independently and did not unlock a third camera.
- **Matching and geometry.** Of 83 evaluated pairs, 10 had fewer than eight mutual ratio survivors and 45 had fewer than 20 even before geometry. Fundamental RANSAC worked on undistorted shared virtual pixel coordinates. Only 14 edges survived the 20-inlier threshold. Of the 14 possible seed edges, five were accepted. Six had insufficient correspondences for two-view initialization, two were homography-dominant and one was degenerate. The 1 px virtual-pixel verifier threshold, depth, angle and cheirality checks were retained. The replay preserves the original geometric checks.
- **Tracks.** 526 tracks ended at two observations, and there were zero rejected merge conflicts in the baseline. The failure mechanism is scarce verified overlap and repeatable keypoint identity across views, with little evidence that the conflict policy discarded usable tracks. A verified pair edge can contain features unrelated to those that seeded the current map.
- **Registration.** The loop visited all 28 pending cameras. None reached PnP because each had fewer than 20 map correspondences. Images 20 and 23 had one and four, respectively. The other 26 had zero. The low reprojection error describes only the two-view component. After exhaustive matching, another seed attracted many more correspondences, but PnP proposals for cameras 0 and 25 sometimes concentrated in under 1% of the image. Those were still rejected.

## Intrinsics

All photos identify as Samsung Galaxy S24+ and record EXIF orientation 6. Oriented images are 3000×4000 and processed images at the baseline limit are 1200×1600. All 30 used the same EXIF 35 mm-equivalent initialization, 2657.94 px original focal length, with no fallback. The processed focal is about 1063.18 px. Intrinsics assume a centered principal point and zero distortion. The metadata establishes a consistent prior, with no independent calibration or lens-distortion estimate. A controlled 3000 px focal variant did not recover a third camera. This does not establish true focal length or rule out calibration error. Joint lens refinement would be weakly constrained by the available views and was left out.

## Diagnostics

`outputs/vader-investigation/baseline-diagnostics/` contains original thumbnails, keypoint overlays, feature coverage, a verified match heatmap and a registration row per image. The thumbnails preserve the source orientation but use square display tiles. The enhanced replay has separate diagnostics in `outputs/vader-investigation/final-diagnostics/`, including actual verified correspondence lines for 1/26 and 12/23. See the [experiment matrix](DARTH_VADER_EXPERIMENTS.md) and [implementation report](REAL_WORLD_SFM_IMPROVEMENTS.md).
