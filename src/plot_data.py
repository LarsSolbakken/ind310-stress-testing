"""Plot prices and daily simple returns per asset; saves results/data_overview.png.

Run from the project root:  python -m src.plot_data
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.data import SPLIT_DATE, compute_returns, download_prices

COLORS = ["#2a78d6", "#eb6834"]  # categorical slots 1-2, one per asset
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"


def main():
    prices = download_prices()
    returns = compute_returns(prices)
    split = pd.Timestamp(SPLIT_DATE)

    fig, axes = plt.subplots(2, len(prices.columns), figsize=(11, 6), sharex=True, facecolor=SURFACE)
    for j, (name, color) in enumerate(zip(prices.columns, COLORS)):
        for i, (series, label) in enumerate([(prices[name], "Price (adj. close)"), (returns[name], "Daily simple return")]):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.plot(series.index, series.values, color=color, lw=0.8 if i else 1.5)
            ax.axvline(split, color=MUTED, lw=1, ls="--")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=MUTED, length=0)
            ax.set_title(f"{name} - {label}" if i == 0 else label, loc="left", color=INK, fontsize=10)
        axes[1, j].axhline(0, color=MUTED, lw=0.8)
    axes[0, 0].text(split, axes[0, 0].get_ylim()[1], " test ->", color=MUTED, fontsize=8, va="top")
    axes[0, 0].text(split, axes[0, 0].get_ylim()[1], "<- train ", color=MUTED, fontsize=8, va="top", ha="right")

    fig.tight_layout()
    out = "results/data_overview.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
