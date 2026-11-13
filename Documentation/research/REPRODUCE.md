# Replay and artifact guide

Run from `C:\projects\Vision-Based-3D-Reconstruction`. Existing source images, portable COLMAP, local PyCOLMAP/pyproj and Blender are required at the paths recorded in environment.json and scripts. No installer is needed for this checkpoint. Configuration JSON is descriptive; the exact executed argument arrays in stage JSON are authoritative.

Use a fresh output root. Do not rerun the converters over an existing asset folder. Do not execute competing orchestration processes on the same output root. Both scenes fit the tested 900-pixel pilot on this host; run sequentially for interpretable performance. This replay deliberately avoids the original shared-GPU timing confound.

```powershell
$baselineRoot = 'outputs/research-baseline-replay'
.venv/Scripts/python.exe -B scripts/research_baseline.py audit --output-root $baselineRoot
.venv/Scripts/python.exe -B scripts/research_baseline_diagnostics.py acquisition --output-root $baselineRoot
.venv/Scripts/python.exe -B scripts/research_baseline_diagnostics.py references --output-root $baselineRoot
foreach ($baselineScene in @('rural','urban')) {
    .venv/Scripts/python.exe -B scripts/research_baseline.py sfm --scene $baselineScene --output-root $baselineRoot
    if ($LASTEXITCODE -ne 0) { throw 'SfM failed; inspect logs' }
    .venv/Scripts/python.exe -B scripts/research_baseline_diagnostics.py sfm --scene $baselineScene --output-root $baselineRoot
    .venv/Scripts/python.exe -B scripts/research_baseline.py dense --scene $baselineScene --output-root $baselineRoot
    if ($LASTEXITCODE -ne 0) { throw 'Dense stage failed; inspect logs' }
    .venv/Scripts/python.exe -B scripts/evaluate_research_baseline.py --scene $baselineScene --output-root $baselineRoot
    .venv/Scripts/python.exe -B scripts/export_research_texture.py --scene $baselineScene --output-root $baselineRoot
    .venv/Scripts/python.exe -B scripts/export_research_texture.py --scene $baselineScene --method poisson --output-root $baselineRoot
    foreach ($baselineAsset in @('asset','asset-poisson')) {
        & 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' --background --factory-startup --python-exit-code 2 --python scripts/blender_research_baseline.py -- --asset-dir "$baselineRoot/$baselineScene/dense-900/$baselineAsset"
        if ($LASTEXITCODE -ne 0) { throw 'Blender failed; inspect native exit code/log' }
        & 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' --background --factory-startup --python-exit-code 2 --python scripts/blender_research_baseline.py -- --asset-dir "$baselineRoot/$baselineScene/dense-900/$baselineAsset" --validate-only
    }
    .venv/Scripts/python.exe -B scripts/evaluate_research_baseline.py --scene $baselineScene --review --output-root $baselineRoot
}
```

This is a replay recipe, not a claim that the fresh replay directory was executed. The actual completed root is `outputs/research-baseline`. Stage JSON in that root records return codes, sampled RSS, commands and runtimes. `blender-returncodes.json` records four independent validation invocations via Python subprocess (avoiding PowerShell warning-to-error wrapping); all are zero. Render reports exist for all four assets. Initial failed Rural export is retained in `asset-unreferenced-export`, with its log; final conversion omits two unused Delaunay vertices while preserving every face and UV.

## Actual result locations

- `{scene}/sfm-2000/models/0`: independent COLMAP binary camera/track/point model; database and feature/matching/mapper logs adjacent.
- `{scene}/dense-900/workspace`: undistorted images, independent cameras, geometric depth/normal/consistency maps and `fused.ply` plus visibility lists.
- `{scene}/dense-900/{delaunay,poisson}.ply`: raw surfaces, unchanged.
- `{scene}/dense-900/textured-{delaunay,poisson}`: COLMAP photographic texture PLY/atlas outputs.
- `{scene}/dense-900/asset`: raw Delaunay OBJ/MTL/PNG/GLB; visual failure evidence, not recommended final asset.
- `{scene}/dense-900/asset-poisson`: preferred review candidate OBJ/MTL/PNG/GLB, conversion provenance, Blender render/validation reports.
- `reports/{scene}-visual-review.jpg`: complete 30-panel review; user acceptance pending.
- `reports/FINAL_BENCHMARK_SUMMARY.md` and `final-benchmark-summary.json`: measured summary and limitations.

COLMAP 4.2.1 performs geometry and photographic texturing; format adapters perform serialization; Blender performs independent import, GLB export/reimport and rendering. RTK positions only enter acquisition diagnostics and post-estimation similarity fitting, never independent SfM estimation. They cannot also be described as independent surface truth. DSM metrics fit a vertical offset and are not absolute or full-3D accuracy.

No OpenMVS invocation was completed. No new research algorithm was implemented. No system-wide changes were made. No large assets/datasets are intended for Git. The local commit was blocked by read-only `.git`; no push occurred.
