# Video keyframe selection

Selection scans the entire MP4 in decode order. The default sample stride approximates four candidate frames per second and grows when necessary to cap scoring at 1200 candidates. OpenCV grabs intervening frames without retrieving them for scoring. Inter-frame codecs may still decode intermediate frames internally. Candidate images are scored at a maximum longest side of 640 px. Selected frames are retrieved again in order in a second pass and stored as PNGs at decoded resolution.

1. Laplacian variance rejects blurred candidates below 18 by default. CLI `--min-sharpness` adjusts the gate. Scores depend on texture and scene lighting.
2. Up to 700 SIFT features are requested. The candidate needs at least 45 detections by default and at least five occupied cells of a 4×4 grid. CLI `--min-features` adjusts the count threshold.
3. Reciprocal descriptor matches against the last retained candidate measure overlap. At least 12 mutual matches with median displacement under 0.6% of the image diagonal signal a near duplicate. Stationary views can be rejected even when a nondegenerate fundamental matrix cannot be estimated.
4. Otherwise, the fundamental verifier requires at least 12 inliers and a 25% inlier fraction at a 1.5 px tolerance in the reduced image. Weak overlap is skipped. A sustained loss can start a separate segment after at least three weak candidates and one second, if a nearby candidate supports the new view. Segments are recorded and do not imply a single connected camera graph.
5. When the provisional sequence exceeds the cap, the default `time-uniform` strategy evenly thins its ordered indices. The opt-in `overlap-aware` strategy instead scores a bounded set of forward candidate pairs with the same verifier and finds a path of the same length using dynamic programming. Both keep the first and last viable candidates. There is no target minimum. A clip with fewer than two viable keyframes produces a clear rejection and selection diagnostics.

The overlap-aware graph tests the next eight candidate offsets, up to five wider offsets scaled to the budget, and any missing links in the uniform baseline. This bounds the search to at most 14 pair evaluations per candidate. The objective first minimizes weak or near-duplicate links. It then favors links with geometric support and useful image displacement while penalizing large departures from the mean temporal gap. Rewards saturate to limit clustering around the strongest feature matches. If the budget or a genuine scene cut makes an unbroken path infeasible, the result can still contain weak transitions.

The pixel displacement is a motion proxy. It cannot separate rotation, translation, independently moving subjects or a changing crop. SfM keeps its existing parallax, pose and reprojection gates. Image quality checks use reduced images, while camera models and reconstruction resizing use native decoded image coordinates and the existing pixel-center transform.

`keyframes.json` records video metadata, nominal timestamps, options, per-sample feature counts, grid coverage, sharpness, overlap statistics and decision counts. The `selection` section independently measures the final neighbors for either strategy and records verification counts, weak transitions, interval frequencies and the first-to-last span divided by the decoded video span. The keyframe verifier uses reduced-resolution images and differs from SfM verification. The child process reports scoring and frame-writing stages to Tk. The existing cancel and shutdown paths terminate and reap the child. The API also accepts a cancellation callback for use outside the desktop.

## Controlled comparison

From the checkout, set `$video` to the original Vader MP4 (`20261110_101208.mp4`) and `$camera` to the previous run's `keyframes/camera.json`. The camera file contains the original approximate focal prior. Use a fresh output root and run both strategies with the same camera and SfM options.

```powershell
foreach ($strategy in @('time-uniform', 'overlap-aware')) {
    $out = "outputs/video-overlap-repeat/$strategy"
    .venv\Scripts\python.exe -m vision3d.cli keyframes --video $video --camera $camera --output "$out/keyframes" --sample-hz 4 --max-keyframes 40 --strategy $strategy
    .venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input "$out/keyframes/frames" --camera $camera --output "$out/reconstruction" --max-dimension 1600 --max-images 50 --window 4 --long-gap 8 --recovery-pairs 30 --verifier magsac --probe-seeds --ratio 0.80 --local-ba --refine-points --min-registered 5
}
```

Compare `keyframes/keyframes.json` and the sparse run's `metrics.json`, `pairs.json`, and `poses.json`. SfM can export a partial reconstruction with exit code 3 when it registers fewer than five views.

The runs under `outputs/video-overlap-ab/` used the same 1,236-frame source, camera model, SfM settings and 40-keyframe budget. Both covered frames 0–1218. The overlap-aware run increased registrations from 3 to 5, verified SfM pairs from 36 to 59 and tracks spanning at least three views from 227 to 675. Connected components fell from 11 to 6. Extraction took 35.0 s for time-uniform selection and 56.9 s for overlap-aware selection. The video has no ground-truth poses, and the two runs reconstructed different subsets, so their reprojection residuals describe their respective maps.
