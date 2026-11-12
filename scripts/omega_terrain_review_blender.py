"""Review a registered terrain asset in Blender without changing its geometry."""

import argparse
import hashlib
from pathlib import Path
import shutil
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import omega_building_blender as shared
import omega_terrain_blender as views


def geometry_digest(obj):
    positions = shared.world_vertices(obj)
    indices = np.empty(len(obj.data.loops), dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", indices)
    digest = hashlib.sha256(positions.tobytes())
    digest.update(indices.tobytes())
    return digest.hexdigest()


def coat_fringe(obj, policy):
    """Mask vertical photo drape on an image-reviewed fringe, keeping the roof plane."""
    corners = shared.world_vertices(obj)
    triangles = np.array([list(polygon.vertices) for polygon in obj.data.polygons])
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("The terrain material review requires an existing triangle mesh.")
    xyz = corners[triangles]
    east, north = (xyz[:, :, :2].mean(axis=1) + policy["origin_enu"][:2]).T
    a, b, c, d = policy["bounds_projected"]
    e, f, g, h = policy["protect_interior_projected"]
    inside = (a <= east) & (east <= c) & (b <= north) & (north <= d)
    protected = (e <= east) & (east <= g) & (f <= north) & (north <= h)
    normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    length = np.linalg.norm(normal, axis=1)
    valid = length > 1e-8
    cosine = np.divide(np.abs(normal[:, 2]), length, out=np.ones_like(length), where=valid)
    steep = cosine < np.cos(np.deg2rad(policy["minimum_slope_degrees"]))
    selected = np.flatnonzero(inside & ~protected & steep & valid)
    if not selected.size:
        raise ValueError("The declared fringe contains no supported steep terrain faces.")
    material = shared.material("Unresolved terrain fringe - neutral", (0.35, 0.35, 0.35, 1))
    obj.data.materials.append(material)
    for index in selected:
        obj.data.polygons[int(index)].material_index = len(obj.data.materials) - 1
    return dict(faces=int(selected.size), area_m2=float(length[selected].sum() * 0.5),
                total_faces=len(triangles), interior_faces_edited=0)


def save_native(source, report, target):
    bpy.ops.wm.open_mainfile(filepath=str(source))
    scene = bpy.context.scene
    scene.timeline_markers.clear()
    for frame, (name, props) in enumerate(report["cameras"].items(), 1):
        camera = bpy.data.cameras.new(name)
        camera.type = "ORTHO"
        camera.ortho_scale = props["ortho_scale"]
        camera.clip_end = 5000
        obj = bpy.data.objects.new(name, camera)
        scene.collection.objects.link(obj)
        obj.location = props["position"]
        obj.rotation_euler = props["euler"]
        marker = scene.timeline_markers.new(name, frame=frame)
        marker.camera = obj
        if name == "oblique":
            scene.frame_set(frame)
            scene.camera = obj
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == "VIEW_3D":
                        area.spaces.active.region_3d.view_perspective = "CAMERA"
    bpy.ops.wm.save_as_mainfile(filepath=str(target))


def sample_mesh_section(baseline, candidate, origin, endpoints, output):
    start, end = np.asarray(endpoints, dtype=float).reshape(2, 2)
    positions = np.linspace(start, end, 101)
    distance = np.linalg.norm(positions - start, axis=1)
    values = []
    for model in (baseline, candidate):
        terrain = [obj for obj in views.import_mesh(model) if not obj.get("building_id")]
        query = shared.terrain_query(terrain)
        samples = [query(*point) for point in positions - np.asarray(origin[:2])]
        values.append(np.asarray(samples, dtype=float) + origin[2])
    np.savetxt(output, np.column_stack([distance, positions, *values]), delimiter=",",
               header="distance_m,easting_m,northing_m,baseline_mesh_m,candidate_mesh_m", comments="")


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    native = out / (args.name + ".blend")
    portable = out / (args.name + ".glb")
    if native.exists() or portable.exists():
        raise FileExistsError("Use a fresh output directory for candidate assets.")
    bpy.ops.wm.open_mainfile(filepath=str(args.scene.resolve()))
    terrain = [obj for obj in shared.meshes() if not obj.get("building_id")]
    if len(terrain) != 1:
        raise ValueError("Expected one terrain mesh in the native baseline.")
    terrain = terrain[0]
    policy = shared.read(args.policy)["urban_material_review"]
    scene = bpy.context.scene
    if scene.get("projected_crs") != policy["projection"] or not np.allclose(
        scene.get("origin_east_north_up_m", []), policy["origin_enu"], atol=0.001, rtol=0
    ):
        raise ValueError("The material policy must use the scene's documented CRS and origin.")
    before = {obj.name: geometry_digest(obj) for obj in shared.meshes()}
    stats = coat_fringe(terrain, policy)
    after = {obj.name: geometry_digest(obj) for obj in shared.meshes()}
    if after != before:
        raise ValueError("The material operation unexpectedly changed mesh geometry.")
    staged = out / "material_trial.blend"
    if staged.exists():
        raise FileExistsError(staged)
    bpy.ops.wm.save_as_mainfile(filepath=str(staged))
    bpy.ops.object.select_all(action="DESELECT")
    expected = {}
    for obj in shared.meshes():
        obj.select_set(True)
        if identity := obj.get("building_id"):
            z = shared.world_vertices(obj)[:, 2]
            expected[identity] = [float(z.min()), float(z.max())]
    bpy.ops.export_scene.gltf(
        filepath=str(portable), export_format="GLB", use_selection=True, export_extras=True
    )
    preview = out / "preview"
    views.comparison(argparse.Namespace(before=args.baseline, after=portable, output=preview,
                                        views=args.views, local_only=False))
    comparison = shared.read(preview / "validation.json")
    for name in ("top", "oblique", "low_angle"):
        shutil.copyfile(preview / f"{name}_after.png", preview / f"{name}.png")
    save_native(staged, comparison, native)
    reports = out / "reports"
    reports.mkdir(exist_ok=True)
    shared.write(reports / "material_trial.json", dict(
        policy=policy, baseline_native=str(args.scene.resolve()), baseline_glb=str(args.baseline.resolve()),
        geometry_identical=True, geometry_sha256=before, **stats
    ))
    shared.write(reports / "building_validation.json",
                 shared.inspect_portable(portable, expected, stats["total_faces"]))
    if args.section:
        sample_mesh_section(args.baseline, portable, policy["origin_enu"], args.section,
                            reports / "court_mesh_section.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "baseline", "policy", "views", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--name", default="urban_terrain_white_buildings")
    parser.add_argument("--section", nargs=4, type=float, metavar=("E0", "N0", "E1", "N1"))
    run(parser.parse_args(sys.argv[sys.argv.index("--") + 1:]))


if __name__ == "__main__":
    main()
