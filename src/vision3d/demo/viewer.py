"""Isolated Open3D viewer for a genuine exported PLY; no Tk embedding."""
import argparse
import json
from pathlib import Path
from time import perf_counter


class OverlayToggle:
    """One visibility state shared by keyboard and desktop controls."""
    def __init__(self, geometry, active, reset, changed=lambda: None):
        self.geometry, self.active, self.reset, self.changed = geometry, active, reset, changed
        self.shown = False

    def set_visible(self, vis, visible):
        if not isinstance(visible, bool):
            raise ValueError("Overlay visibility must be boolean")
        if visible == self.shown:
            return False
        operation = vis.add_geometry if visible else vis.remove_geometry
        if operation(self.geometry, reset_bounding_box=True) is False:
            raise RuntimeError("Open3D could not update camera/axis visibility")
        self.shown = visible
        self.active[id(self.geometry)] = visible
        self.reset(vis)
        self.changed()
        return False

    def toggle(self, vis):
        return self.set_visible(vis, not self.shown)


class CameraControl:
    """Read only changed desktop commands, at most ten times per second."""
    def __init__(self, path, overlay):
        self.path, self.overlay = Path(path), overlay
        self.last_text, self.next_poll = None, 0.0

    def poll(self, vis):
        now = perf_counter()
        if now < self.next_poll:
            return False
        self.next_poll = now + .1
        try:
            text = self.path.read_text(encoding='utf-8')
            if text == self.last_text:
                return False
            value = json.loads(text)['show_cameras']
            if not isinstance(value, bool):
                return False
            self.overlay.set_visible(vis, value)
            self.last_text = text
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Missing or in-progress command files do not disrupt rendering.
        return False


def fit_zoom(bounds, view, aspect, field_of_view):
    """Fit the eight scene bounds corners inside 90% of the perspective viewport."""
    from itertools import product
    import numpy as np

    low, high = bounds
    extent = max(float(np.max(high - low)), 1e-12)
    corners = np.array(list(product(*zip(low, high, strict=True)))) - view["lookat"]
    right = np.cross(view["up"], view["front"])
    half_fov = np.tan(np.deg2rad(field_of_view / 2))
    projected = np.maximum(np.abs(corners @ right) / aspect, np.abs(corners @ view["up"]))
    distance = np.max(corners @ view["front"] + projected / (.9 * half_fov))
    distance += .03 * extent
    return max(.6, float(distance * half_fov / extent))


def reset_view(vis, view, bounds):
    """Reset framing, including any change in the window aspect ratio."""
    vis.reset_view_point(True)
    control = vis.get_view_control()
    intrinsics = control.convert_to_pinhole_camera_parameters().intrinsic
    aspect = intrinsics.width / max(intrinsics.height, 1)
    control.set_lookat(view["lookat"])
    control.set_front(view["front"])
    control.set_up(view["up"])
    control.set_zoom(fit_zoom(bounds, view, max(aspect, .1), control.get_field_of_view()))
    return False


def standalone_view(points, textured):
    import numpy as np

    center = points.min(axis=0) + np.ptp(points, axis=0) / 2
    # GLB export flips down/forward axes. Keep its baked geometry and orient
    # only the preview. PLY uses the reconstruction's OpenCV world axes.
    front = np.array([.35, .55, 1.])
    up = np.array([0., 1., 0.])
    if not textured:
        front[1:] *= -1
        up[1] = -1
    front /= np.linalg.norm(front)
    up -= up.dot(front) * front
    up /= np.linalg.norm(up)
    return {"lookat": center, "front": front, "up": up}


def load_cloud(path):
    import numpy as np
    import open3d as o3d
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".ply":
        raise ValueError(f"PLY file not found: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty() or not np.isfinite(np.asarray(cloud.points)).all():
        raise ValueError("PLY contains no usable finite 3D points.")
    if not cloud.has_colors() or not np.isfinite(np.asarray(cloud.colors)).all():
        raise ValueError("PLY has no valid point colors; expected a colored reconstruction.")
    return cloud


def load_mesh(path):
    import numpy as np
    import open3d as o3d

    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".ply":
        raise ValueError(f"Mesh PLY file not found: {path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if (not mesh.has_triangles() or not mesh.has_vertex_colors() or
            not np.isfinite(np.asarray(mesh.vertices)).all() or not np.isfinite(np.asarray(mesh.vertex_colors)).all()):
        raise ValueError("PLY has no usable colored triangle mesh.")
    return mesh


def load_textured_meshes(path):
    import numpy as np
    import open3d as o3d

    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".glb":
        raise ValueError(f"Textured GLB file not found: {path}")
    model = o3d.io.read_triangle_model(str(path))
    meshes, photo_groups = [], 0
    for group in model.meshes:
        mesh = group.mesh
        if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertices)).all():
            raise ValueError("GLB contains an invalid mesh group")
        image = model.materials[group.material_idx].albedo_img
        if mesh.has_triangle_uvs() and np.asarray(image).ndim == 3:
            mesh.textures = [image]
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh.triangles), dtype=np.int32))
            photo_groups += 1
        elif mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors).clip(0., 1.)
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.where(
                colors <= .0031308, 12.92 * colors, 1.055 * colors ** (1 / 2.4) - .055))
        meshes.append(mesh)
    if photo_groups == 0:
        raise ValueError("GLB contains no bound photographic texture")
    return meshes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactive viewer for an exported point cloud or mesh")
    parser.add_argument("--ply", "--model", dest="ply", required=True)
    parser.add_argument("--status-file", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mesh", action="store_true", help="Load the PLY as a triangle mesh")
    mode.add_argument("--textured", action="store_true", help="Load a photographic GLB")
    parser.add_argument("--control-file", help="Optional desktop camera-visibility commands")
    args = parser.parse_args(argv)
    status_file = Path(args.status_file)
    def status(value, **extra):
        status_file.write_text(json.dumps({"status": value, **extra}), encoding="utf-8")
    visualizer = None
    try:
        import open3d as o3d
        import numpy as np
        if args.textured:
            geometries = load_textured_meshes(args.ply)
            points = np.concatenate([np.asarray(mesh.vertices) for mesh in geometries])
            label = "photographic mesh"
        else:
            geometry = load_mesh(args.ply) if args.mesh else load_cloud(args.ply)
            geometries = [geometry]
            points = geometry.vertices if args.mesh else geometry.points
            label = "colored mesh" if args.mesh else "point cloud"
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        if not visualizer.create_window(f"Vision-Based 3D Reconstruction — {label}", width=1100, height=760):
            raise RuntimeError("Open3D could not create a rendering window. Check the desktop/OpenGL driver.")
        if not all(visualizer.add_geometry(geometry) for geometry in geometries):
            raise RuntimeError("Open3D could not add the geometry to the viewer.")
        options = visualizer.get_render_option()
        options.background_color = [1.0, 1.0, 1.0]
        options.point_size = 4.0
        options.mesh_show_back_face = args.mesh or args.textured
        if args.textured:
            options.mesh_color_option = o3d.visualization.MeshColorOption.Color
            options.light_on = False
        # Camera-aligned framing changes only the view, not the actual exported points.
        from .view_geometry import camera_view, camera_frustum, load_context, load_intrinsics
        points = np.asarray(points)
        view = standalone_view(points, args.textured)
        context_status = "standalone_fitted_view"
        cameras = axes_toggle = None
        active_overlays = {}
        model_bounds = (points.min(axis=0), points.max(axis=0))
        context_bounds = model_bounds
        context_center = None
        def reset(vis):
            if any(active_overlays.values()):
                return reset_view(vis, {**view, "lookat": context_center}, context_bounds)
            return reset_view(vis, view, model_bounds)

        def pan(vis, dx, dy):
            vis.get_view_control().translate(dx, dy)
            return False

        def open_status():
            status("open", points=len(points), ply=str(Path(args.ply).resolve()), viewpoint=context_status,
                   cameras_available=cameras is not None, cameras_visible=bool(cameras and cameras.shown),
                   axes_visible=bool(axes_toggle and axes_toggle.shown),
                   model=label,
                   controls="Left drag orbit · Ctrl + left drag or I/J/K/L pan · scroll zoom · R or F fit/reset · C cameras · A axes")
        try:
            poses = [] if args.textured else load_context(args.ply)
            view = camera_view(points, [p['world_to_camera'] for p in poses])
            context_status = "camera_aligned_all_exported_points"
            # Frustums/axes are optional context overlays, never scene landmarks.
            span = float(np.linalg.norm(np.ptp(points, axis=0)))
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(span * .08, 1e-6))
            intrinsics = load_intrinsics(args.ply)
            vertices, edges = [], []
            for p in poses:
                k = intrinsics[p['image_id']]
                v, e = camera_frustum(p['world_to_camera'], k['K'], k['width'], k['height'], max(span * .04, 1e-6))
                edges.extend((e + len(vertices)).tolist()); vertices.extend(v.tolist())
            frustums = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(edges))
            frustums.paint_uniform_color([.25, .25, .25])
            bounds = np.vstack([points, vertices, [[0, 0, 0]]])
            context_bounds = (bounds.min(axis=0), bounds.max(axis=0))
            context_center = (context_bounds[0] + context_bounds[1]) / 2
            cameras = OverlayToggle(frustums, active_overlays, reset, open_status)
            axes_toggle = OverlayToggle(axes, active_overlays, reset, open_status)
            visualizer.register_key_callback(ord('C'), cameras.toggle)
            visualizer.register_key_callback(ord('A'), axes_toggle.toggle)
            if args.control_file:
                control = CameraControl(args.control_file, cameras)
                visualizer.register_animation_callback(control.poll)
        except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError):
            pass
        reset(visualizer)
        for key in ("R", "F"):
            visualizer.register_key_callback(ord(key), reset)
        for key, dx, dy in (("J", -45, 0), ("L", 45, 0), ("K", 0, 45), ("I", 0, -45)):
            visualizer.register_key_callback(ord(key), lambda vis, x=dx, y=dy: pan(vis, x, y))
        visualizer.poll_events()
        visualizer.update_renderer()
        open_status()
        visualizer.run()
        status("closed", points=len(points))
    except Exception as exc:
        status("error", message=f"{type(exc).__name__}: {exc}")
        return 2
    finally:
        if visualizer is not None:
            visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
