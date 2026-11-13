"""Thin orchestration over proven open-source UAV photogrammetry tools (ODM, Roofer).

See Documentation/ARCHITECTURE.md. This package must not import the legacy ``vision3d``
pipeline, and only ``uav3d.evaluate`` may read reference data.
"""

__version__ = "0.1.0"
