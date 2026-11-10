# Multi-view reconstruction completion report

The desktop now runs a sparse-to-dense workflow for three or more registered views. It can produce a colored fused cloud, an open vertex-colored mesh and a validated GLB. Sparse-only remains the desktop default. The existing two-photo pipeline and calibrated sequence behavior remain available.

## Design

The reusable `reconstruct_multiview` stage reads sparse poses, original intrinsics, landmarks and hashed sources. Covisibility and median triangulation angle rank possible neighbors. OpenCV rectification and bidirectional SGBM estimate depth. This reuses the existing CPU stereo backend, including a bounded disparity search and rejection of inconsistent left/right matches. Vertical rectification transposes both the pixel geometry and reprojection matrix. Plane sweep and PatchMatch would require a new estimator or a separately deployed backend. OpenMVS remains a candidate for a future quality comparison. [OpenCV calibration and stereo](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) · [OpenMVS architecture](https://github.com/cdcseacave/openMVS/blob/develop/docs/architecture.md)

Each dense sample must project consistently into a different reference depth map. The second stereo edge must include a third camera, so reversing an edge cannot corroborate itself. The check combines positive in-bounds depth, relative depth agreement and a color difference bound. World voxel averages reduce duplicates while retaining image colors. Fast, Balanced and High use 640, 900 and 1200 px longest sides, cap references at 8, 12 and 16, and use relative depth tolerances of 2.5%, 2% and 1.5%. The total depth canvas is capped at 8 million pixels.

Meshing connects adjacent depth samples only when their depth variation is within the limit, then maps faces to fused voxels. It removes degenerate and duplicate faces, removes small connected components, computes normals and optionally applies quadric simplification. A bounded open surface is preferable here to Poisson hole filling because the scenes include unseen sides and discontinuities. Mesh vertices are projected into depth-consistent photographs for color blending. Vertices without valid projections retain their fused image colors. [Open3D mesh methods](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html)

PLY keeps the sparse world axes and arbitrary baseline units. GLB preserves vertex positions, topology, normals and linear vertex colors with a proper right/up/back basis change. Both formats are independently reloaded by Open3D before being reported as successful. Pose and intrinsic JSON accompany each result. No dependencies were added. [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)

## Desktop and verification

```powershell
& 'C:\projects\Vision-Based-3D-Reconstruction\run_demo.bat'
```

Choose **Load Castle Demo**, select **fast** under **Multi-view**, enable **Colored mesh** and optionally **GLB export**, then start. Switch between the sparse and dense results after the worker finishes. **Open Mesh** loads the surface. **Open Output Folder** exposes the files. Cancellation and dense failures keep a successfully written sparse result available. The two-photo dense stereo command remains a separate CLI workflow.

**137 tests passed** with 14 existing dependency warnings. Tests cover relative-pose direction, vertical rectification and depth scale, voxel merging, a synthetic three-view dense run, PLY/GLB reload, changed image hashes, insufficient views, invalid output reuse and desktop failure isolation. The real calibrated Castle desktop run and mesh camera overlay were validated programmatically. See [evaluation](MULTIVIEW_EVALUATION.md) for the before/after figures and commands. A genuine Open3D capture is stored locally at `outputs/mv-castle-validated/mesh-preview.png`.

## Remaining work

- UV unwrapping and texture baking, interoperable OBJ export and optional smoothing are unfinished. GLB currently carries vertex colors.
- The mesh can contain open boundaries, overlapping patches and artifacts. Higher point counts do not establish better geometry. Dense surface accuracy, broad ordinary-photo robustness and production-scale memory use remain unverified.
- Temple registration remains 18/30 by default. Graph recovery was reproduced and left opt-in due to worse reference agreement. Intrinsics refinement needs controlled reference-based validation.
- Next, validate on additional calibrated and uncalibrated scenes, measure against reference surfaces, improve pose recovery on weak overlaps and evaluate a dedicated PatchMatch backend for difficult scenes.
