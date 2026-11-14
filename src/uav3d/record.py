"""Run record (run.json) and measured subprocess execution (wall time + peak RAM of the tree)."""
from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunRecord:
    """Append-only description of one run, rewritten to disk after every change."""

    def __init__(self, path: Path, **header):
        self.path = path
        self.data = {
            "started": _now(),
            "host": {"platform": platform.platform(), "cpus": psutil.cpu_count(),
                     "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
            **header,
            "stages": [],
            "status": "running",
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()

    def stage(self, name: str, **info) -> dict:
        entry = {"name": name, "started": _now(), **info}
        self.data["stages"].append(entry)
        self.save()
        return entry

    def finish(self, status: str, error: str | None = None) -> None:
        self.data.update(status=status, finished=_now())
        if error:
            self.data["error"] = error
        self.save()


def run_measured(cmd: list[str], log: Path, *, cwd: Path | None = None, env: dict | None = None,
                 poll_s: float = 0.5, cpus: int | None = None) -> dict:
    """Run ``cmd``, streaming stdout+stderr to ``log``; sample RSS of the whole process tree.

    ``cpus`` pins the whole process tree to the first N logical CPUs (children inherit the
    affinity; late children are pinned by the sampler). Needed because some ODM sub-steps
    (OpenMVS point-cloud filter) ignore ODM's --max-concurrency.
    Returns {"returncode", "wall_s", "peak_rss_gb"}. Peak RSS is a sampled lower bound.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    peak = 0
    t0 = time.perf_counter()
    mask = None
    if cpus:
        available = psutil.Process().cpu_affinity()
        if cpus < len(available):
            mask = available[:cpus]
    with open(log, "ab") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        root = psutil.Process(proc.pid)
        pinned: set[int] = set()

        def pin(p: psutil.Process) -> None:
            if mask is None or p.pid in pinned:
                return
            try:
                p.cpu_affinity(mask)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            pinned.add(p.pid)

        pin(root)
        stop = threading.Event()

        def sample() -> None:
            nonlocal peak
            while not stop.is_set():
                try:
                    procs = [root, *root.children(recursive=True)]
                    rss = 0
                    for p in procs:
                        pin(p)
                        try:
                            rss += p.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    return
                stop.wait(poll_s)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            code = proc.wait()
        finally:
            stop.set()
            sampler.join(timeout=5)
    return {"returncode": code, "wall_s": round(time.perf_counter() - t0, 1),
            "peak_rss_gb": round(peak / 2**30, 2)}
