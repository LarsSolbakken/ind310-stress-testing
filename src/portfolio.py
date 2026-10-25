"""Portfolio weights and returns."""
import numpy as np
import pandas as pd


def make_weights(**w):
    """make_weights(SP500=0.5, BONDS=0.5) -> Series; checks that weights sum to 1."""
    weights = pd.Series(w, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1, got {weights.sum()}")
    return weights


def portfolio_returns(asset_returns, weights):
    """Daily-rebalanced portfolio simple return: weighted sum of asset simple returns."""
    return asset_returns[list(weights.index)] @ weights
