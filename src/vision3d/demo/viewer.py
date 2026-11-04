"""Isolated Open3D viewer for a genuine exported PLY; no Tk embedding."""
import argparse
import json
from pathlib import Path


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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactive viewer for an exported sparse point cloud")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--status-file", required=True)
    args = parser.parse_args(argv)
    status_file = Path(args.status_file)
    def status(value, **extra):
        status_file.write_text(json.dumps({"status": value, **extra}), encoding="utf-8")
    visualizer = None
    try:
        import open3d as o3d
        cloud = load_cloud(args.ply)
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        if not visualizer.create_window("Vision-Based 3D Reconstruction — sparse point cloud", width=1100, height=760):
            raise RuntimeError("Open3D could not create a rendering window. Check the desktop/OpenGL driver.")
        if not visualizer.add_geometry(cloud):
            raise RuntimeError("Open3D could not add the point cloud to the viewer.")
        options = visualizer.get_render_option()
        options.background_color = [1.0, 1.0, 1.0]
        options.point_size = 4.0
        visualizer.reset_view_point(True)
        # Camera-aligned framing changes only the view, not the actual exported points.
        from .view_geometry import camera_view, camera_frustum, load_context
        import numpy as np
        context_status = "default_view_no_camera_metadata"
        try:
            poses = load_context(args.ply)
            view = camera_view(np.asarray(cloud.points), [p['world_to_camera'] for p in poses])
            active_overlays = {}
            context_center = view['lookat']
            def reset(vis):
                control = vis.get_view_control()
                control.set_lookat(context_center if any(active_overlays.values()) else view['lookat'])
                control.set_front(view['front']); control.set_up(view['up']); control.set_zoom(0.6 if any(active_overlays.values()) else 0.72)
                return False
            reset(visualizer)
            visualizer.register_key_callback(ord('R'), reset)
            context_status = "camera_aligned_all_exported_points"
            # Frustums/axes are optional context overlays, never scene landmarks.
            span = float(np.linalg.norm(np.ptp(np.asarray(cloud.points), axis=0)))
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(span * .08, 1e-6))
            intrinsics = {p['image_id']: p['processed'] for p in json.loads((Path(args.ply).parent/'intrinsics.json').read_text())}
            vertices, edges = [], []
            for p in poses:
                k = intrinsics[p['image_id']]
                v, e = camera_frustum(p['world_to_camera'], k['K'], k['width'], k['height'], max(span * .04, 1e-6))
                edges.extend((e + len(vertices)).tolist()); vertices.extend(v.tolist())
            frustums = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(edges))
            frustums.paint_uniform_color([.25, .25, .25])
            bounds = np.vstack([np.asarray(cloud.points), vertices, [[0, 0, 0]]])
            context_center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
            def toggle(geometry):
                shown = False
                def callback(vis):
                    nonlocal shown
                    if shown: vis.remove_geometry(geometry, reset_bounding_box=True)
                    else: vis.add_geometry(geometry, reset_bounding_box=True)
                    shown = not shown
                    active_overlays[id(geometry)] = shown
                    reset(vis)
                    return False
                return callback
            visualizer.register_key_callback(ord('C'), toggle(frustums))
            visualizer.register_key_callback(ord('A'), toggle(axes))
        except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError):
            pass  # Standalone colored PLY files remain viewable without camera context.
        visualizer.poll_events()
        visualizer.update_renderer()
        status("open", points=len(cloud.points), ply=str(Path(args.ply).resolve()), viewpoint=context_status,
               display_filter="none; every exported point is displayed", controls="R reset; C camera outlines; A coordinate axes (when metadata available)")
        visualizer.run()
        status("closed", points=len(cloud.points))
    except Exception as exc:
        status("error", message=f"{type(exc).__name__}: {exc}")
        return 2
    finally:
        if visualizer is not None:
            visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
