"""Small plotting helpers shared by the molten-salt Gmix workflows."""

from __future__ import annotations


TARGET_T_K = 1200.0


def ebar(ax, x, y, yerr, **kwargs):
    """Draw compact error bars using only Matplotlib's public API."""
    kwargs.setdefault("capsize", 1.5)
    kwargs.setdefault("elinewidth", 0.8)
    kwargs.setdefault("capthick", 0.8)
    return ax.errorbar(x, y, yerr=yerr, **kwargs)
