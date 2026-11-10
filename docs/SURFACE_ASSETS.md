# UAV surface assets

`surface-asset` reuses a successful `dense-sequence --mesh` run. It checks that the recorded poses and intrinsics agree with the sparse artifacts before processing the PLY. A relocated sparse directory can be supplied with `--sparse-dir`.

```powershell
.venv/Scripts/python.exe -m vision3d.cli surface-asset --dense-dir outputs/example-dense --output outputs/example-surface --target-faces 450000
```

The target is a requested upper face count for quadric simplification. Shared-edge winding is repaired by solving face-orientation constraints. In contradictory cycles, the smaller incident triangle is cut. The camera-facing direction of each resulting component is estimated from visible, non-grazing samples. An unresolved component retains its consistent local winding. Output acceptance checks edge incidence, winding, degenerate faces and normals, and samples nearest distances in both directions between the input and output meshes. The default P99 bound is two dense voxels. The sampled maximum must fit within five times that bound. `--max-error` expresses a different P99 bound in SfM units.

The fixed AirZoo Rural and Urban meshes passed at requested targets of 450,000 and 900,000 faces, respectively. The more aggressive 450,000-face Urban run exceeded the sampled maximum distance and was rejected. Both accepted exports removed shared-edge winding conflicts. They also increased component counts, since contradictory strips were cut and simplification can disconnect thin regions. The distance checks measure deviation from the input surface. Independent reference geometry is unavailable.
