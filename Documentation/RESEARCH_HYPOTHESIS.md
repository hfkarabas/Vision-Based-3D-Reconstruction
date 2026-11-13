# Research hypothesis and stop conditions

The project objective remains to turn UAV photos/video into meaningful, recognizable textured 3D environment assets. The broad proposal, “quality-guided regional adaptive reconstruction under computational constraints,” is **mostly established at the conceptual level**. A narrower implementation may be potentially differentiated, but its originality and utility are not yet demonstrated. See [direct competitors](PRIOR_WORK_COMPARISON.md).

Selecting more views for weak regions, using a partial model, prioritizing uncertain surface, and saving computation each have direct predecessors. Restricting candidates to already captured photos is also established. Combining their names or adding a texture score does not establish a new contribution.

## Candidate that merits further investigation

After visual baseline acceptance, investigate a **cost-calibrated regional action scheduler with explicit evidence dependence and an abstention action**. Inputs are fixed captured images, a frozen established SfM model, a low-cost baseline surface/depth state, and a wall-time/RAM budget. Actions invoke existing backends: add a reference/source-view set, increase image resolution, rerun a bounded supported region if the backend supports it, or abstain. A backend that only supports whole-reference maps must be charged for the whole map; do not pretend it executed a crop.

Proposed technical question: does accounting for correlated camera support and observed failure/benefit after each action improve **valid** completeness per second compared with static quality ranking? Merely counting additional points can reward duplicated or incorrect sheets. The representation must include unsupported/unobserved space so a bad seed mesh cannot permanently suppress missing regions.

The precise original claim, if prior work permits it, would be an empirically calibrated marginal **valid-area gain** model and decision rule that accounts for support dependence, asymmetric false-surface cost, measured end-to-end action cost, and feedback from preceding actions. A generic weighted sum or greedy ranking is insufficient. No submodularity guarantee should be asserted when feedback or action interactions violate its assumptions.

Newer search checks also found Dong and Li's 2026 grouped conformal confidence calibration and Cilic et al.'s 2026 cost-aware edge/cloud reconstruction scheduling (L41–L42). Thus neither dependence-aware confidence nor cost-aware scheduling can stand alone as an originality claim. Their full methods and publication chronology remain to be resolved before assessing the narrower combination.

## Falsifiable hypothesis, not a result

At predeclared budgets of 25%, 50%, and 75% of a frozen full established-backend run, the scheduler will increase supported completeness by at least 5 percentage points over the strongest reproducible static selector at the 50% budget, without lowering geometric precision by more than 1 percentage point. This is a proposed minimum useful effect, not a measured achievement. Include all diagnostic, scheduling, fusion, remeshing, texturing and export time in the practical budget; additionally report MVS-only cost for mechanism analysis.

Primary endpoints require independently measured reference geometry. Use P/R/F-score at metric distances chosen from the reference resolution and uncertainty before testing. For AirZoo, absent certified 3D reference, report DSM-only/common-footprint residuals and spatial coverage separately; they cannot prove 3D precision/completeness. Reference DSM may evaluate an alignment but must not choose actions. Never use the same reference to train a gain predictor and claim held-out accuracy on it.

## Experiments that could disprove it

| Control | Purpose |
|---|---|
| Full COLMAP; OpenMVS where available | Establish usable upper-cost reference and backend dependence |
| Uniform/random reference selection with fixed seeds | Test whether sophisticated ranking is necessary |
| Sparse-overlap/parallax/GSD greedy coverage | Strong simple selector |
| Mostegel-style confidence/quality fulfillment ranking | Closest static depth-prioritization competitor; label reproduction fidelity |
| Hornung-style proxy weak-region addition | Closest partial-model selection competitor |
| NBS-style coverage/uncertainty pair selection | Test whether the apparent contribution already follows from NBV/NBS |
| Existing native uav/uav-recovery/uav-coverage | Historical and engineering comparison, not sole competitors |
| No dependence penalty; no feedback; no cost calibration; no abstention; geometry-only vs texture-aware | Attribute any effect to the proposed mechanism |

Freeze one development scene/region and reserve spatially disjoint scenes for test. AirZoo Rural and Urban alone are too few for a broad generalization claim. Include additional geographically distinct UAV captures and a benchmark with verified 3D reference; the local Caterpillar video is not established as UAV. Repeated seeds quantify selection variance; aggregate by scene or spatial block, not millions of correlated points treated as independent samples. Report paired confidence intervals, quality-budget curves, peak resources and all failures. A full-run “oracle” is diagnostic only and cannot inform the tested scheduler.

Concrete external evaluation candidates, **not downloaded locally**: [UseGeo](https://usegeo.fbk.eu/home) provides co-acquired UAV imagery and LiDAR reference, making it more suitable for geometric evaluation than an unverified DSM alone. [JB3D](https://github.com/JB3D-Dataset/Multi-View-Stereo) provides aerial videos and laser reference, but its current release notes a camera-translation export correction; resolve that correction and data terms before use. [WHU-OMVS](https://gpcv.whu.edu.cn/data/WHU_OMVS_dataset/WHU_dataset.htm) is rendered synthetic aerial data, suitable for controlled tests but not independent measured real-world evidence. UseGeo plus existing AirZoo scenes and an ETH3D measured-reference control is a practical candidate design, subject to access, subset size, calibration and license checks.

The hypothesis fails if the strongest simple selector matches it within uncertainty, gains come from removing difficult footprint, gains disappear when overhead is counted, texture improves while geometry deteriorates, or performance depends on a single scene/seed/backend. Do not retune thresholds on the held-out results.

## Up to three alternatives if overlap remains decisive

1. Narrow empirical contribution: a reproducible UAV reconstruction benchmark and failure taxonomy with equal-input crossed backends, spatial support diagnostics and visual acceptance. Valuable undergraduate research; describe it as experimental methodology, not a novel MVS algorithm.
2. Dependence-aware evidence calibration: estimate whether nominal multi-view support is actually redundant across clustered UAV stations, then test calibration and action-ranking value. Requires direct uncertainty/fusion literature audit first.
3. Backend-neutral stopping/abstention rule: predict when extra processing cannot recover unsupported regions and preserve precision under a budget. Requires comparison against existing confidence and next-best-view stopping rules.

No alternative changes the asset-output goal. None is an approved novelty claim. Do not implement any candidate until the independent visual baseline passes and the closest literature is methodologically resolved.
