"""Per-tile image subsets from an ODM OpenSfM reconstruction. Runs with ODM's own Python.

usage: python opensfm_tiles.py <request.json>

request = {"opensfm": dir, "odm_home": dir, "k": int, "tiles": [[id, x0, y0, x1, y1], ...],
           "export": [{"name": str, "tiles": [id, ...], "out": dir}, ...]}   # export optional
Writes <request>.result.json: per tile the sparse points inside and the shots observing >= k of them;
per export group the scene path and its shots. Coordinates are the undistorted reconstruction's frame
(ODM: UTM minus the offset in odm_georeferencing/coords.txt).

The OpenMVS scene of a group is written with OpenSfM's own exporter (``export_openmvs.export``) restricted
to the group's shots; only the output folder is redirected so the project's scene.mvs is not touched.
"""
import json
import os
import sys

request_path = sys.argv[1]
req = json.load(open(request_path, encoding="utf-8"))
for p in ("SuperBuild/install/bin/opensfm", "SuperBuild/install/lib/python3.12/dist-packages"):
    sys.path.insert(0, os.path.join(req["odm_home"], p))

import numpy as np  # noqa: E402
from opensfm.actions import export_openmvs  # noqa: E402
from opensfm.dataset import DataSet  # noqa: E402


class OutputRedirect:
    """Undistorted dataset whose data_path (used by the exporter only for the output file) is elsewhere."""

    def __init__(self, udata, path):
        self._u, self.data_path = udata, path

    def __getattr__(self, name):
        return getattr(self._u, name)


udata = DataSet(req["opensfm"]).undistorted_dataset()
rec = udata.load_undistorted_reconstruction()[0]
tm = udata.load_undistorted_tracks_manager()
k = int(req["k"])

ids = list(rec.points)
xy = np.array([rec.points[i].coordinates[:2] for i in ids]) if ids else np.zeros((0, 2))
tiles = {int(t[0]): t[1:] for t in req["tiles"]}
members = {}  # tile id -> point ids inside
for tid, (x0, y0, x1, y1) in tiles.items():
    inside = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
    members[tid] = [ids[i] for i in np.flatnonzero(inside)]


def shots_for(point_ids):
    count = {}
    for pid in point_ids:
        for shot in tm.get_track_observations(pid):
            if shot in rec.shots:
                count[shot] = count.get(shot, 0) + 1
    return sorted(s for s, n in count.items() if n >= k)


result = {"shots_total": len(rec.shots), "points_total": len(ids), "tiles": {}, "groups": []}
for tid, pts in members.items():
    result["tiles"][str(tid)] = {"points": len(pts), "shots": shots_for(pts)}
for group in req.get("export", []):
    pts = [p for tid in group["tiles"] for p in members[int(tid)]]
    shots = shots_for(pts)
    os.makedirs(group["out"], exist_ok=True)
    export_openmvs.export(rec, tm, OutputRedirect(udata, group["out"]), {s: True for s in shots})
    result["groups"].append({"name": group["name"], "tiles": group["tiles"], "shots": shots,
                             "scene": os.path.join(group["out"], "openmvs", "scene.mvs")})
json.dump(result, open(request_path.replace(".json", ".result.json"), "w", encoding="utf-8"))
