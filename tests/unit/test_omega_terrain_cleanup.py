"""Checks for the optional GIS terrain integration environment."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

if importlib.util.find_spec("rasterio") is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from omega_terrain_cleanup import court_plane, interpolate


@unittest.skipUnless(importlib.util.find_spec("rasterio"), "optional GIS runtime required")
class TerrainCleanupTests(unittest.TestCase):
    def test_sloped_supported_court(self):
        rows, cols = np.indices((20, 20))
        ground = (27 + 0.02 * rows + 0.01 * cols).astype("float32")
        source = ground.copy()
        source[9, 9] += 3
        core = (rows > 1) & (cols > 1) & (rows < 18) & (cols < 18)
        plane, evidence = court_plane(source, core, 1)
        np.testing.assert_allclose(plane[core], ground[core], atol=1e-4)
        self.assertAlmostEqual(evidence["fitted_slope_percent"], np.sqrt(5), places=3)
        self.assertTrue(np.isnan(plane[~core]).all())

    def test_unsupported_court_abstains(self):
        rows, cols = np.indices((20, 20))
        source = (np.sin(rows) * np.cos(cols) * 8).astype("float32")
        with self.assertRaisesRegex(ValueError, "stable plane"):
            court_plane(source, np.ones_like(source, dtype=bool), 1)

    def test_fill_distance_is_bounded(self):
        source = np.full((15, 15), 20.0, dtype="float32")
        anchors = np.zeros_like(source, dtype=bool)
        anchors[7, 7] = True
        values, distances = interpolate(source, anchors, max_distance=3, cell=1)
        self.assertEqual(float(values[7, 7]), 20)
        self.assertTrue(np.isnan(values[distances > 3]).all())
