"""Export an existing projected DTM with Qgis2threejs' upstream GridGeometry."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import rasterio

from omega_uav_pipeline import measured, sha256


def export(dtm, plugin, output, node):
    output.mkdir(parents=True, exist_ok=False)
    with rasterio.open(dtm) as src:
        if (src.crs is None or not src.crs.is_projected or
                src.crs.linear_units_factor[1] != 1 or src.transform.b or
                src.transform.d or src.transform.a <= 0 or src.transform.e >= 0):
            raise ValueError("Qgis2threejs adapter requires a north-up, projected metric DEM")
        if min(src.width, src.height) < 2:
            raise ValueError("A DEM surface requires at least two rows and columns")
        if src.width * src.height > 500_000:
            raise ValueError("DEM grid exceeds the current bounded export profile")
        values = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        valid = np.isfinite(values)
        if not valid.any() or np.any(values[valid] == -9999):
            raise ValueError("DEM has no valid values or collides with the exchange NoData value")
        manifest = dict(
            columns=src.width, rows=src.height, nodata=-9999, data="elevation.f32",
            width=(src.width - 1) * src.transform.a,
            height=(src.height - 1) * -src.transform.e,
            bounds=list(src.bounds), crs=src.crs.to_string(),
            origin_enu=[(src.bounds.left + src.bounds.right) / 2,
                        (src.bounds.bottom + src.bounds.top) / 2, 0],
        )
    source = plugin / "web/src/layer/demlayer.ts"
    metadata = (plugin / "metadata.txt").read_text(encoding="utf-8")
    version = next(line.split("=", 1)[1].strip() for line in metadata.splitlines()
                   if line.startswith("version="))
    manifest.update(dtm=str(dtm), dtm_sha256=sha256(dtm), qgis2threejs_version=version,
                    upstream_module_sha256=sha256(source),
                    node_version=subprocess.check_output([node, "--version"], text=True).strip())
    np.where(valid, values, -9999).astype("<f4").tofile(output / manifest["data"])
    exchange = output / "grid.json"
    exchange.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle = output / "upstream.cjs"
    measured(output, "bundle", [
        node, plugin / "node_modules/esbuild/bin/esbuild", source,
        "--bundle", "--platform=node", "--format=cjs", f"--outfile={bundle}",
    ])
    measured(output, "export", [
        node, Path(__file__).with_suffix(".cjs"), bundle, exchange, output / "terrain.ply",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dtm", "plugin", "output"):
        parser.add_argument(f"--{name}", required=True, type=lambda path: Path(path).resolve())
    parser.add_argument("--node", default="node")
    export(**vars(parser.parse_args()))
