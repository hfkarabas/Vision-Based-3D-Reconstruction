"""Run with ODM's python: export an OpenMVS scene containing only the shots that see a tile.

usage: tile_export.py <opensfm dir> <out dir> <cx> <cy> <half> [min_points]
Tile = axis-aligned square in the reconstruction's topocentric XY frame. A shot is kept when it
observes at least `min_points` sparse points inside the tile.
"""
import json
import os
import sys

odm_home = os.environ["ODM_HOME"]
for p in ("SuperBuild/install/bin/opensfm", "SuperBuild/install/lib/python3.12/dist-packages"):
    sys.path.insert(0, os.path.join(odm_home, p))

from opensfm.actions import export_openmvs  # noqa: E402
from opensfm.dataset import DataSet  # noqa: E402

osfm, out, cx, cy, half = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:6])
min_points = int(sys.argv[6]) if len(sys.argv) > 6 else 20


class OutputRedirect:
    """Undistorted dataset whose data_path (used only for the output file) points elsewhere."""

    def __init__(self, udata, path):
        self._u, self.data_path = udata, path

    def __getattr__(self, name):
        return getattr(self._u, name)


udata = DataSet(osfm).undistorted_dataset()
rec = udata.load_undistorted_reconstruction()[0]
tm = udata.load_undistorted_tracks_manager()
inside = [pid for pid, pt in rec.points.items()
          if abs(pt.coordinates[0] - cx) <= half and abs(pt.coordinates[1] - cy) <= half]
count = {}
for pid in inside:
    for shot in tm.get_track_observations(pid):
        if shot in rec.shots:
            count[shot] = count.get(shot, 0) + 1
keep = {s: True for s, n in count.items() if n >= min_points}
os.makedirs(out, exist_ok=True)
export_openmvs.export(rec, tm, OutputRedirect(udata, out), keep)
z = sorted(rec.points[p].coordinates[2] for p in inside)
print(json.dumps({"shots_total": len(rec.shots), "shots_kept": len(keep), "points_in_tile": len(inside),
                  "z_median": z[len(z) // 2] if z else None}))
