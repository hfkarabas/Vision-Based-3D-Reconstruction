# UAV surface assets

`surface-asset` reuses a successful `dense-sequence --mesh` run. It checks that the recorded poses and intrinsics agree with the sparse artifacts before processing the PLY. A relocated sparse directory can be supplied with `--sparse-dir`.

```powershell
.venv/Scripts/python.exe -m vision3d.cli surface-asset --dense-dir outputs/example-dense --output outputs/example-surface --target-faces 450000
```

The target is a requested upper face count for quadric simplification. Shared-edge winding is repaired by solving face-orientation constraints. In contradictory cycles, the smaller incident triangle is cut. The camera-facing direction of each resulting component is estimated from visible, non-grazing samples. An unresolved component retains its consistent local winding. Output acceptance checks edge incidence, winding, degenerate faces and normals, and samples nearest distances in both directions between the input and output meshes. The default P99 bound is two dense voxels. The sampled maximum must fit within five times that bound. `--max-error` expresses a different P99 bound in SfM units.

The fixed AirZoo Rural and Urban meshes passed at requested targets of 450,000 and 900,000 faces, respectively. The more aggressive 450,000-face Urban run exceeded the sampled maximum distance and was rejected. Both accepted exports removed shared-edge winding conflicts. They also increased component counts, since contradictory strips were cut and simplification can disconnect thin regions. The distance checks measure deviation from the input surface. Independent reference geometry is unavailable.

## Photographic UV texture

```powershell
.venv/Scripts/python.exe -m vision3d.cli texture-asset --surface-dir outputs/example-surface --output outputs/example-texture --atlas-size 4096 --obj
```

The program checks the original image hashes and projects each triangle into visible registered photographs. It checks occlusion at the center and near the three corners, then ranks eligible views by incidence angle, projected pixel area and local image contrast. A near-tie view can replace an isolated label to reduce seams. Charts share padded photo tiles in the atlas. GLB uses the atlas as its base-color texture and uses vertex colors for faces that lack a valid source. Optional OBJ uses the same atlas through MTL and a neutral fallback material. Keep `textured.obj`, `textured.mtl` and `atlas.jpg` together.

GLB and OBJ positions use the proper rotation `[x, -y, -z]` from the input SfM frame. Export scale defaults to one output unit per SfM unit. `--units-per-world` applies a user-supplied factor. No metric calibration is inferred. Open3D reload validates material groups, triangle positions, normals, UVs and the atlas dimensions.

Set `--holdout-stride 7` to reserve every seventh registered photo from texture selection. The report compares atlas colors with visible samples in those photographs. These views still contribute to the sparse poses and may contribute to the dense mesh. `scripts/audit_surface_depth.py` separately measures mesh first-hit depth against reciprocal stereo from cameras absent from successful dense pair estimates.

In the desktop, choose a dense profile and select **Colored mesh** and **Photo texture**. Geometry and UV export run as cancellable stages after the dense result. **Open UV GLB** opens the generated asset or lets you pick a previously exported GLB. The quality report and both asset directories remain available for inspection.
