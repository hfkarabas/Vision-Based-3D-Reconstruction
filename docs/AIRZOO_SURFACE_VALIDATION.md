# AirZoo surface validation

The fixed 40-image Rural and Urban dense meshes from the previous mission are the reference inputs. The separate 48-image Rural run regenerated dense depth from the saved 31-camera sparse map. For command options and desktop use, see [Surface assets](SURFACE_ASSETS.md).

| Metric | Rural | Urban |
|---|---:|---:|
| Input → optimized triangles | 1,420,823 → 434,645 | 3,107,260 → 848,146 |
| Input → optimized winding conflicts | 17,401 → 0 | 50,615 → 0 |
| Input → optimized connected components | 496 → 3,699 | 9 → 6,124 |
| P99 input-to-output surface displacement in SfM units | 0.000717 | 0.000921 |
| Photographic UV coverage by area | 32.62% | 21.73% |
| Textured GLB size | 22.95 MB | 37.06 MB |

Both GLB and OBJ/MTL reloaded through Open3D. Rural's benchmark replay reproduced the textured GLB byte-for-byte. The new texture controls and the desktop subprocess stages ran against a real saved Rural reconstruction. The 48-image run yielded an independently reloaded GLB with 248,023 faces and 34.15% photographic coverage. The full test suite passed 179 tests before the shared-witness depth experiment.

The topology and bounded distance checks support use as optimized open meshes. All eight sampled neighborhoods in each optimized mesh still contained intersections. The camera-facing winding repair and simplification increase fragmentation. The sub-voxel weld and the conservative intersection experiment failed to establish defensible fixes on the real scenes. Rural's reserved stereo views produced no viable pair. Urban's four reserved pairs measured a small decrease in depth agreement. These are internal and held-out depth checks in an arbitrary SfM frame. Unity 6000.6.1f1 could not import in batch mode because the editor has no active license.

Run the benchmark driver from the repository root with fresh output directories. The previously saved sparse and dense runs must be locally available.

```powershell
$p='outputs/airzoo-benchmark/major-surface-improvement'
.venv/Scripts/python.exe -B scripts/benchmark_surface_assets.py --scene rural --output "$p/replay-rural-new"
.venv/Scripts/python.exe -B scripts/benchmark_surface_assets.py --scene urban --output "$p/replay-urban-new"
```

Detailed reports, models, textures and renders are generated locally under `outputs/airzoo-benchmark/major-surface-improvement/`. They are excluded from version control.
