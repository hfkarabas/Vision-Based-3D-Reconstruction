import sys

import psutil
import pytest

from uav3d.record import run_measured


@pytest.mark.skipif(len(psutil.Process().cpu_affinity()) < 2, reason="needs >= 2 CPUs")
def test_run_measured_pins_child_and_grandchild(tmp_path):
    # child spawns a grandchild; both report their affinity
    code = ("import psutil, subprocess, sys;"
            "print('child', len(psutil.Process().cpu_affinity()), flush=True);"
            "subprocess.run([sys.executable, '-c', 'import psutil;"
            "print(\"grand\", len(psutil.Process().cpu_affinity()))'])")
    log = tmp_path / "log.txt"
    r = run_measured([sys.executable, "-c", code], log, cpus=1)
    assert r["returncode"] == 0
    text = log.read_text()
    assert "child 1" in text and "grand 1" in text


def test_run_measured_without_cpus_leaves_affinity(tmp_path):
    n = len(psutil.Process().cpu_affinity())
    log = tmp_path / "log.txt"
    run_measured([sys.executable, "-c", "import psutil; print(len(psutil.Process().cpu_affinity()))"], log)
    assert log.read_text().strip() == str(n)
