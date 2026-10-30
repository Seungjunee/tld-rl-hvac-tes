"""Normalization utilities shared across transfer methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sinergym.utils.wrappers import NormalizeObservation
from studies.case_study.core.types import ModelBundle
from studies.case_study.core.wrappers import PhysicalNormalizeObservation


def _uses_physical_normalization(env) -> bool:
    """Check if the env wrapper chain uses PhysicalNormalizeObservation."""
    current = env
    while hasattr(current, "env"):
        if isinstance(current, PhysicalNormalizeObservation):
            return True
        current = current.env
    return False


def seed_normalization_from_source(env, source_bundle: ModelBundle) -> bool:
    """Seed NormalizeObservation stats from source bundle's export.

    Walks the wrapper chain, finds NormalizeObservation, and loads
    mean/var/count from the source bundle's norm_paths. Stats update
    (automatic_update) is left unchanged so the target can adapt.

    With PhysicalNormalizeObservation, seeding is unnecessary (static
    bounds derived from physics), so this function returns True immediately.

    Returns True if normalization was seeded (or not needed), False if no stats found.
    """
    # Physical normalization uses static bounds — no stats to transfer
    if _uses_physical_normalization(env):
        return True

    norm_paths = source_bundle.norm_paths
    if norm_paths is None:
        return False

    mean_path = norm_paths.mean
    var_path = norm_paths.var
    count_path = norm_paths.count

    if mean_path is None or not Path(mean_path).exists():
        return False

    norm_wrapper = _find_normalize_wrapper(env)
    if norm_wrapper is None:
        return False

    # Load source stats
    source_mean = np.loadtxt(str(mean_path), dtype=np.float64)
    if np.ndim(source_mean) == 0:
        source_mean = np.array([float(source_mean)], dtype=np.float64)

    target_dim = norm_wrapper.obs_rms.mean.shape[0]
    source_dim = len(source_mean)

    if source_dim == target_dim:
        # Exact match — direct copy
        norm_wrapper.obs_rms.mean = source_mean
        if var_path and Path(var_path).exists():
            source_var = np.loadtxt(str(var_path), dtype=np.float64)
            if np.ndim(source_var) == 0:
                source_var = np.array([float(source_var)], dtype=np.float64)
            norm_wrapper.obs_rms.var = source_var
        if count_path and Path(count_path).exists():
            source_count = np.loadtxt(str(count_path), dtype=np.float64)
            norm_wrapper.obs_rms.count = float(np.asarray(source_count).item())
        return True

    # Dimension mismatch — skip seeding (e2e/duppad have different obs dims)
    return False


def _find_normalize_wrapper(env) -> NormalizeObservation | None:
    """Walk wrapper chain to find NormalizeObservation."""
    current = env
    while hasattr(current, "env"):
        if isinstance(current, NormalizeObservation):
            return current
        current = current.env
    return None
