"""Evaluation against independent reference data (labels, LiDAR, reference DSM).

This is the only package allowed to read reference data. Nothing in the production path
(plan, footprints, buildings, the scheduler) may import it; a test enforces this.
"""
