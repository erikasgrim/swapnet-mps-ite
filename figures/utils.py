"""Shared utilities for figure generation notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FONT_CONFIG = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 12,
}

DEFAULT_COLOR_CYCLE = ["#004791", "#4DA060", "#E86A58", "#FFBE00"]


def configure_plot_style(
    style: str = "default",
    font_config: dict | None = None,
    color_cycle: list[str] | None = None,
) -> None:
    """Apply consistent matplotlib configuration for paper-ready plots."""
    plt.style.use(style)
    cycle = color_cycle or DEFAULT_COLOR_CYCLE
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cycle)
    plt.rcParams.update(font_config or DEFAULT_FONT_CONFIG)


def set_size(
    width: float | str = 232,
    fraction: float = 1,
    squeeze_height: float = 1,
    subplots: tuple[int, int] = (1, 1),
) -> tuple[float, float]:
    """Return figure dimensions that avoid scaling inside LaTeX documents."""
    if width == "full":
        width_pt = 484
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = (
        fig_width_in
        * golden_ratio
        * (subplots[0] / subplots[1])
        * squeeze_height
    )

    return fig_width_in, fig_height_in


def tnq_to_index(n_timesteps: int, n_assets: int, n_qubits: int, t: int, n: int, q: int) -> int:
    """Map (time, asset, qubit) coordinates to flattened binary variable index."""
    return (t * n_assets + n) * n_qubits + q


def index_to_tnq(
    n_timesteps: int, n_assets: int, n_qubits: int, idx: int
) -> tuple[int, int, int]:
    """Inverse of ``tnq_to_index``."""
    q = idx % n_qubits
    n = (idx // n_qubits) % n_assets
    t = (idx // (n_qubits * n_assets)) % n_timesteps
    return t, n, q


def solution_to_tnq(
    solution: np.ndarray, n_timesteps: int, n_assets: int, n_qubits: int
) -> np.ndarray:
    """Reshape flattened binary solution into (time, asset, qubit) tensor."""
    tnq_solution = np.zeros((n_timesteps, n_assets, n_qubits), dtype=int)
    for idx, value in enumerate(solution):
        t, n, q = index_to_tnq(n_timesteps, n_assets, n_qubits, idx)
        tnq_solution[t, n, q] = value
    return tnq_solution


def decode_weights(x: np.ndarray, budget: float) -> np.ndarray:
    """Decode binary representation into real-valued portfolio weights."""
    powers = 2 ** np.arange(x.shape[-1])
    return (x * powers).sum(axis=-1) / budget


def total_profit(
    x: np.ndarray,
    returns: np.ndarray,
    budget: float,
    lam: float,
) -> np.ndarray:
    """Compute total profit per optimization step, including transaction costs."""
    weights = decode_weights(x, budget)

    # Portfolio profit from returns
    profits_t = (weights * returns[None, :, :]).sum(axis=-1)
    gross_profit = profits_t.sum(axis=-1)

    # Transaction cost term λ * Σ_t (Δw_t)^2
    dw = np.diff(weights, axis=1)
    cost = lam * (dw**2).sum(axis=(1, 2))

    return gross_profit - cost


def sharpe_ratio(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    budget: float,
    risk_free: float = 0.0,
) -> np.ndarray:
    """Compute theoretical Sharpe ratio per optimization step."""
    weights = decode_weights(x, budget)

    # Expected return per step: sum_t w_t · mu_t
    expected_return = np.einsum("stn,tn->s", weights, mu)

    # Portfolio variance: sum_t w_t^T Σ_t w_t
    variance = np.einsum("stn,tnm,stm->s", weights, sigma, weights)

    return np.where(variance > 0, (expected_return - risk_free) / np.sqrt(variance), np.nan)
