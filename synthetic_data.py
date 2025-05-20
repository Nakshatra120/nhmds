"""
synthetic_data.py
=================
Utility functions for generating synthetic hyperbolic data and distance matrices
with user‑controlled, *correlated* uncertainty.

Designed to sit next to ``core_hmds.py`` and ``plot_hmds.py`` so that you can do::

    from synthetic_data import (
        sample_hyperbolic_points,
        build_distance_matrix,
        add_correlated_noise,
    )

and plug the resulting (noisy) matrix straight into the HMDS / nonmetricHMDS
classes.

All distances use the **Poincaré ball model** (curvature ‑1) because it’s easy to
sample and has a closed‑form distance expression.
"""

from __future__ import annotations

from typing import Callable, Literal, Union

import numpy as np

__all__ = [
    "sample_hyperbolic_points",
    "poincare_distance",
    "build_distance_matrix",
    "add_correlated_noise",
]


# -----------------------------------------------------------------------------
#   Sampling helpers
# -----------------------------------------------------------------------------

def sample_hyperbolic_points(
    n: int,
    dim: int = 2,
    *,
    max_norm: float = 0.85,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw *n* random points inside the ``dim``‑dimensional Poincaré ball.

    Parameters
    ----------
    n
        Number of points.
    dim
        Embedding dimension *inside* the ball (Euclidean dimension before warping).
    max_norm
        Upper bound for the radial coordinate.  Keeping this <1 avoids points too
        close to the boundary (huge distances + numerical overflow).
    rng
        Optional NumPy random generator for reproducibility.
    """
    rng = np.random.default_rng(rng)

    # Random directions (isotropic)
    X = rng.normal(size=(n, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # Random radii biased towards the centre so we get nicely spread data
    u = rng.random(n)
    r = max_norm * u ** (1 / dim)  # inverse‑CDF of volume element in dim‑ball

    return X * r[:, None]


# -----------------------------------------------------------------------------
#   Distances in the Poincaré ball
# -----------------------------------------------------------------------------

def poincare_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Hyperbolic distance between **two** points *u*, *v* in the Poincaré ball.

    The closed‑form expression (curvature ‑1)::

        d(u,v) = arcosh\left(1 + \frac{2‖u-v‖^2}{(1-‖u‖^2)(1-‖v‖^2)}\right)
    """
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)
    diff = np.linalg.norm(u - v)
    arg = 1 + 2 * diff**2 / ((1 - du**2) * (1 - dv**2))
    # Due to numerical round‑off the argument can creep slightly <1 → clamp
    return np.arccosh(np.clip(arg, 1.0, None))


# -----------------------------------------------------------------------------
#   Full distance matrix
# -----------------------------------------------------------------------------

def build_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Vectorised pairwise distance matrix for *points* on the Poincaré ball."""
    n = len(points)
    dmat = np.zeros((n, n))
    for i in range(n):
        # Broadcast against all later indices to avoid redundant work
        dst = np.array([poincare_distance(points[i], points[j]) for j in range(i + 1, n)])
        dmat[i, i + 1 :] = dst
        dmat[i + 1 :, i] = dst
    return dmat


# -----------------------------------------------------------------------------
#   Correlated noise injection
# -----------------------------------------------------------------------------
_NoiseRule = Union[
    Literal["linear", "quadratic"],
    Callable[[float], float],
]

def add_correlated_noise(
    dmat: np.ndarray,
    *,
    rule: _NoiseRule = "linear",
    scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return *a copy* of *dmat* with distance‑dependent Gaussian noise.

    Parameters
    ----------
    dmat
        Ground‑truth distance matrix.
    rule
        • ``"linear"``    σ_ij = *scale* · d_ij
        • ``"quadratic"`` σ_ij = *scale* · d_ij²
        • *callable*     A function mapping true distance → σ (e.g. ``lambda x: 0.05+x**1.5``)
    scale
        Proportionality factor when ``rule`` is a string.  Ignored for a callable rule.
    rng
        Optional NumPy random generator for reproducibility.

    Notes
    -----
    The matrix is symmetrised after adding noise to ensure d_ij = d_ji.
    The diagonal is kept at 0.
    """
    rng = np.random.default_rng(rng)
    noisy = dmat.copy()

    # Precompute σ_ij for every pair.
    if callable(rule):
        sigmas = rule(dmat)
    elif rule == "linear":
        sigmas = scale * dmat
    elif rule == "quadratic":
        sigmas = scale * dmat**2
    else:
        raise ValueError("Unknown noise rule: {rule!r}")

    # Add symmetric Gaussian noise
    n = len(dmat)
    for i in range(n):
        eps = rng.normal(0.0, sigmas[i, i + 1 :])
        noisy[i, i + 1 :] += eps
        noisy[i + 1 :, i] = noisy[i, i + 1 :]

    return noisy
