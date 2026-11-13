# Proposed research architecture

Recommendation is provisional until explicit user visual review. Full independent SfM/MVS and textured Delaunay/Poisson exports are complete; Poisson is the preferred visual candidate. Rural is coherent; Urban has substantial gaps. Raw Delaunay peripheral closures are unsuitable. Use established geometry software, a small experimental decision layer only if justified, and the existing desktop shell. No large refactor is part of Mission 0. See [measured results](research/BASELINE_RESULTS.md).

```text
Original photos / decoded video frames + immutable metadata
  -> dataset manifest and calibration policy
  -> independent COLMAP SfM -> versioned camera/track interface
  -> established MVS adapter (COLMAP; OpenMVS comparison)
  -> depth / support / fused cloud artifacts
  -> established mesher + photographic texturer
  -> OBJ/MTL/textures and GLB -> independent Blender verification
                          |
                  quality evaluation
                          |
          optional future budgeted decision policy
          (schedules existing operations; no new MVS engine)
```

The desktop submits jobs and opens exported assets. It must not own backend geometry state. Preserve the native implementation as an explicitly named comparison backend. Use existing CLI workers, manifests, logging, cancellation and replay where practical.

## Artifact contracts

Each stage records executable/version, full argument array, input hashes, calibration model and pixel convention, image transformation history, camera coordinate convention, units/gauge, start/end time, return code and sampled resource scope. Camera data uses a documented world-to-camera convention, explicit distortion and image dimensions; adapters handle half-pixel differences instead of silently sharing raw matrices. Similarity/geospatial alignment is a separate artifact. Retain all connected models and rejected images.

Depth artifacts include image identity, validity, units, normals, source-view list and backend parameters. Mesh artifacts retain raw and derived versions, normals, support/visibility evidence, texture atlas and UV convention. Metric reports identify footprint, alignment, sampling policy, thresholds and reference lineage. The scheduler, if later justified, receives evaluation-independent state and emits auditable action requests with predicted cost/benefit and reasons to abstain. Its logs must support replay without a GUI.

COLMAP currently supplies independent SfM, CUDA PatchMatch, fusion, Delaunay/Poisson and the installed `mesh_texturer`. OpenMVS offers an alternative densify/mesh/refine/texture route after `InterfaceCOLMAP`; select it from results rather than reputation. The [upstream interface documentation](https://github.com/cdcseacave/openMVS/blob/develop/docs/wiki/Interface.md) specifies the undistorted COLMAP interface. The upstream repository currently identifies AGPL-3.0 and includes a memory-aware subscene workflow; redistribution/integration decisions need a version-specific license inventory. Record actual licenses and bundled third-party notices before packaging; process separation alone does not settle licensing obligations. See the [upstream repository](https://github.com/cdcseacave/openMVS).

The [COLMAP license](https://github.com/colmap/colmap/blob/main/LICENSE) identifies a BSD license for COLMAP itself and explicitly distinguishes its dependencies. No license/notice files were found in the inspected portable root/bin, so do not redistribute that binary bundle from this checkpoint. The [AirZoo-Real README license section](https://huggingface.co/datasets/RingoWRW97/AirZoo-Real/blob/main/README.md) leaves dataset terms pending; redistribution rights remain unresolved. All mission assets remain local. Dataset distribution permission is a separate release requirement.

## Migration milestones and success criteria

| Order | Work | Gate / technical risk |
|---|---|---|
| 1 | Finish independent full-scene SfM diagnostics | Both complete scene attempts disclosed, graph and calibration plausible; rotational station clusters may make self-calibration ill-conditioned |
| 2 | Complete dense, two surfaces, established photographic texture and exports | All-stage provenance, whole-scene top/oblique/side views; reject floating layers, unsupported closures and broad fragmentation; 16 GB RAM requires bounded resolution/caches |
| 3 | User visual review of full montage and source-image correspondence | Record pending/approved/rejected explicitly; import alone is insufficient |
| 4 | Matched-input native/backend comparisons and reference audit | Common-frame coverage and accuracy; separate image count/resolution/backend effects; resolve CRS/datum before absolute metrics |
| 5 | Finish direct-competitor and citation-chain review | Written precise distinction or revise thesis to empirical contribution; broad idea already overlaps strongly |
| 6 | Minimal adapter into desktop and reproducible CLI | Same saved established asset opens/exports through application; verify cancellation and replay; avoid rewriting viewer |
| 7 | Only then implement one bounded research decision method | Predeclare metric, budget, comparator and held-out split; no new SfM/MVS engine |
| 8 | Ablations, negative results, thesis and demo package | Claim only effects supported by scene-level uncertainty and visual inspection |

Stages 1–5 are scientific gates rather than estimated completion dates. The shortest responsible graduation path is a convincing established reconstruction plus an honest, small evaluation contribution. If a novel scheduler is not differentiated, finish a rigorous systems/experimental thesis with the supervisor instead of inventing originality.
