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
        visualizer = o3d.visualization.Visualizer()
        if not visualizer.create_window("Vision-Based 3D Reconstruction — sparse point cloud", width=1100, height=760):
            raise RuntimeError("Open3D could not create a rendering window. Check the desktop/OpenGL driver.")
        if not visualizer.add_geometry(cloud):
            raise RuntimeError("Open3D could not add the point cloud to the viewer.")
        options = visualizer.get_render_option()
        options.background_color = [1.0, 1.0, 1.0]
        options.point_size = 3.0
        visualizer.reset_view_point(True)
        visualizer.poll_events()
        visualizer.update_renderer()
        status("open", points=len(cloud.points), ply=str(Path(args.ply).resolve()))
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
