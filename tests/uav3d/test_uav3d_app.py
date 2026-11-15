from datetime import datetime, timezone
from pathlib import Path

from uav3d import app


def test_build_command_maps_every_option():
    cmd = app.build_command("C:/img", "C:/out", threads=10, buildings=False, export=False, refine_budget=0.5,
                            ram_gb=8.0, keep_intermediates=True, reuse_odm=True, python="py")
    assert cmd[:6] == ["py", "-m", "uav3d", "run", "C:/img", "--out"]
    for flag in ("--skip-buildings", "--skip-export", "--keep-odm-intermediates", "--reuse-odm"):
        assert flag in cmd
    assert cmd[cmd.index("--threads") + 1] == "10"
    assert cmd[cmd.index("--refine-budget") + 1] == "0.5"
    assert cmd[cmd.index("--budget-ram-gb") + 1] == "8"
    plain = app.build_command("a", "b", python="py")
    assert plain == ["py", "-m", "uav3d", "run", "a", "--out", "b"]


def _run(status, names, **extra):
    stages = [{"name": n, "started": f"2026-11-15T10:{i:02d}:00+00:00", "peak_rss_gb": 1.0 + i}
              for i, n in enumerate(names)]
    return {"started": "2026-11-15T09:59:00+00:00", "status": status, "ingest": {"count": 5}, "stages": stages, **extra}


def test_phase_rows_running():
    now = datetime(2026, 11, 15, 10, 3, tzinfo=timezone.utc)
    rows = {r["key"]: r for r in app.phase_rows(_run("running", ["odm:sfm", "odm:dense"]), now=now)}
    assert rows["images"]["status"] == "done"
    assert rows["sfm"]["status"] == "done" and rows["sfm"]["seconds"] == 60
    assert rows["dense"]["status"] == "running" and rows["dense"]["seconds"] == 120
    assert rows["terrain"]["status"] == "waiting"
    assert "adaptive" not in rows  # hidden unless requested or present


def test_phase_rows_failed_and_finished():
    failed = _run("failed", ["odm:sfm", "odm:dense", "detect", "terrain"], error="terrain: RuntimeError: x",
                  finished="2026-11-15T10:05:00+00:00")
    rows = {r["key"]: r for r in app.phase_rows(failed)}
    assert rows["terrain"]["status"] == "failed" and rows["buildings"]["status"] == "not_reached"
    ok = _run("ok", ["odm:sfm", "odm:dense", "detect", "terrain", "buildings:footprints", "buildings:roofer"],
              finished="2026-11-15T10:07:00+00:00")
    rows = {r["key"]: r for r in app.phase_rows(ok)}
    assert rows["buildings"]["status"] == "done" and rows["buildings"]["peak_gb"] == 6.0
    assert rows["export"]["status"] == "skipped"
    early = {"status": "failed", "error": "ingest: ValueError: no images", "stages": []}
    assert app.phase_rows(early)[0]["status"] == "failed"


def test_summary_reads_results(tmp_path: Path):
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene" / "scene.glb").write_bytes(b"x")
    run = _run("ok", ["odm:sfm", "export"], finished="2026-11-15T10:59:00+00:00",
               buildings={"status": "ok", "buildings": 30}, capture={"warning": "Oblique capture"})
    run["stages"][0]["gsd"] = {"gsd_m": 0.0267, "metric": True}
    run["stages"][1].update(buildings_in_scene=28, flagged_buildings=["a", "b"])
    s = app.summary(run, tmp_path)
    assert s["gsd_cm"] == 2.67 and s["buildings"] == 30 and s["buildings_in_scene"] == 28 and s["flagged"] == 2
    assert s["minutes"] == 60.0 and s["glb"] and s["blend"] is None and s["warnings"] == ["Oblique capture"]


def test_quiet_threads_leaves_cores_free():
    assert app.quiet_threads(14) == 10 and app.quiet_threads(4) == 3 and app.quiet_threads(1) == 1
