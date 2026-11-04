"""Post-hoc Sim(3) camera alignment and scale-independent pair pose errors."""
import numpy as np


def angle_degrees(rotation):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def similarity_alignment(estimated, reference):
    x, y = np.asarray(estimated, np.float64), np.asarray(reference, np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3 or not np.isfinite([x, y]).all():
        raise ValueError("Alignment requires >=3 finite paired Nx3 camera centers")
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    if np.linalg.matrix_rank(xc) < 2 or np.linalg.matrix_rank(yc) < 2:
        raise ValueError("Camera-center alignment is rotationally ambiguous for a collinear trajectory")
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    signs = np.array([1, 1, np.linalg.det(u @ vt)])
    rotation = u @ np.diag(signs) @ vt
    scale = float((singular * signs).sum() / np.mean(np.sum(xc ** 2, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid similarity scale")
    offset = y.mean(axis=0) - scale * rotation @ x.mean(axis=0)
    return scale, rotation, offset


def evaluate_poses(estimated_records, reference):
    records = [r for r in estimated_records if r.get("world_to_camera") is not None and r["name"] in reference]
    records.sort(key=lambda r: r["image_id"])
    if len(records) < 3:
        return {"status": "unavailable", "reason": "fewer_than_three_reference_registered_cameras"}
    estimated = [np.asarray(r["world_to_camera"], float) for r in records]
    truth = [np.asarray(reference[r["name"]], float) for r in records]
    if any(p.shape != (4, 4) or not np.isfinite(p).all() or
           not np.allclose(p[3], [0, 0, 0, 1]) or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-6)
           or not np.isclose(np.linalg.det(p[:3, :3]), 1, atol=1e-6) for p in estimated + truth):
        raise ValueError("Reference evaluation requires finite proper world-to-camera poses")
    centers = lambda poses: np.array([-p[:3, :3].T @ p[:3, 3] for p in poses])
    x, y = centers(estimated), centers(truth)
    try:
        scale, rotation, offset = similarity_alignment(x, y)
    except ValueError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    aligned = scale * x @ rotation.T + offset
    errors = np.linalg.norm(aligned - y, axis=1)
    rotation_errors = [angle_degrees(e[:3, :3] @ rotation.T @ t[:3, :3].T) for e, t in zip(estimated, truth)]
    relative_rotations, directions = [], []
    for k in range(1, len(estimated)):
        a, b, ta, tb = estimated[k - 1], estimated[k], truth[k - 1], truth[k]
        re, rt = b[:3, :3] @ a[:3, :3].T, tb[:3, :3] @ ta[:3, :3].T
        relative_rotations.append(angle_degrees(re @ rt.T))
        te, tt = b[:3, 3] - re @ a[:3, 3], tb[:3, 3] - rt @ ta[:3, 3]
        if min(np.linalg.norm(te), np.linalg.norm(tt)) > 1e-9:
            directions.append(float(np.rad2deg(np.arccos(np.clip(te @ tt / (np.linalg.norm(te) * np.linalg.norm(tt)), -1, 1)))))
    rms = float(np.sqrt(np.mean(errors ** 2)))
    extent = float(np.linalg.norm(np.ptp(y, axis=0)))
    return {"status": "evaluated", "registered_reference_cameras": len(records),
            "alignment": {"scale": scale, "rotation": rotation.tolist(), "offset": offset.tolist()},
            "center_rmse_reference_units": rms, "center_rmse_fraction_trajectory_extent": rms / extent,
            "center_errors_reference_units": errors.tolist(), "aligned_rotation_errors_degrees": rotation_errors,
            "median_aligned_rotation_degrees": float(np.median(rotation_errors)),
            "median_relative_rotation_degrees": float(np.median(relative_rotations)),
            "relative_translation_direction_errors_degrees": directions,
            "median_translation_direction_degrees": float(np.median(directions)) if directions else None,
            "geometric_accuracy_warning": bool(rms / extent > 0.1 or np.median(rotation_errors) > 5),
            "warning_basis": "reporting flag only: >10% trajectory extent center RMSE or >5 degrees median aligned rotation"}


def plot_pose_comparison(estimated_records, reference, evaluation, path):
    """Static diagnostic of matched reference cameras only, after reconstruction."""
    if evaluation.get("status") != "evaluated":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = sorted((r for r in estimated_records if r.get("world_to_camera") is not None and r["name"] in reference), key=lambda r: r["image_id"])
    poses = [np.array(r["world_to_camera"]) for r in records]
    truth = [np.array(reference[r["name"]]) for r in records]
    centers = lambda matrices: np.array([-p[:3, :3].T @ p[:3, 3] for p in matrices])
    transform = evaluation["alignment"]
    aligned = transform["scale"] * centers(poses) @ np.array(transform["rotation"]).T + transform["offset"]
    target = centers(truth)
    fig = plt.figure(figsize=(11, 5))
    axis = fig.add_subplot(121, projection="3d")
    axis.plot(*target.T, "o-", label="Reference")
    axis.plot(*aligned.T, "x--", label="Estimated, Sim(3) aligned")
    axis.legend(fontsize=8)
    axis.set(title=f"{len(records)} matched camera centers", xlabel="X", ylabel="Y", zlabel="Z")
    chart = fig.add_subplot(122)
    chart.plot([r["image_id"] for r in records], evaluation["aligned_rotation_errors_degrees"], "o-")
    chart.set(xlabel="Original image ID", ylabel="Rotation error (degrees)", title="Aligned camera rotation accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
