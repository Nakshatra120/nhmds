
"""
plot_hmds.py

Visualization helpers for HMDS models.  Designed to be used alongside
core_hmds.py – pass in a trained model instance and call the desired
plotting routine.

Example
-------
>>> from core_hmds import HMDS, nonmetricHMDS
>>> from plot_hmds import plot_loss, plot_sigmas
>>> model = HMDS(dmat, D=3)
>>> model.train(n=10_000)
>>> plot_loss(model)
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------- Metric model visualizations --------------------------------

def plot_loss(model):
    """Plot optimisation loss history (expects ``model.loss_curve``)."""
    plt.figure()
    plt.plot(model.loss_curve)
    plt.xlabel('Step')
    plt.ylabel('-Log Likelihood')
    plt.title('HMDS optimisation')
    plt.show()


def plot_lambda(model):
    """Plot inferred λ (curvature scaling) over optimisation."""
    if not hasattr(model, 'lambda_curve'):
        raise AttributeError('Model has no attribute ``lambda_curve``.')
    plt.figure()
    plt.plot(model.lambda_curve)
    plt.xlabel('Step')
    plt.ylabel('λ')
    plt.title('λ trajectory')
    plt.show()


def plot_sigmas(model):
    """Histogram of per‑node uncertainties σ (exp(logσ))."""
    sigmas = np.exp(model.log_sig.numpy())
    plt.figure()
    plt.hist(sigmas, bins='auto')
    plt.xlabel('σ')
    plt.ylabel('Count')
    plt.title('Posterior σ distribution')
    plt.show()


# ---------------- Non‑metric‑specific visualisations -------------------------

def plot_transformation(nm_model):
    """Plot learned monotonic transformation for Non‑metric HMDS."""
    plt.figure()
    max_dist = np.max(nm_model.pairwise_dist_lorentz().numpy())
    x = np.linspace(-5, max_dist + 10, 200, dtype=np.float32)
    plt.plot(x, nm_model.monotonic(x))
    plt.vlines(max_dist, 0, nm_model.monotonic(max_dist),
               linestyles='dashed', color='red', alpha=0.6)
    plt.xlabel('dᵢⱼ (Lorentz)')
    plt.ylabel('f(dᵢⱼ)')
    plt.title('Non‑metric monotonic mapping')
    plt.show()
