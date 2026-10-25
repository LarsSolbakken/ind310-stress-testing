"""Train the conditional VAE on the training period only and save checkpoints/vae.pt.

  python -m src.train_vae

Step 1: fit on 2004-2015 and validate on 2016-2018 to find the best number of epochs (early stopping).
Step 2: retrain on all of 2004-2018 for that many epochs and save the model.
The test period (2019-2024) is never used here.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_data
from src.vae import (CHECKPOINT, WINDOW, ScenarioGenerator, checksum, iwae_nll, kl_per_dim, make_samples,
                     train_cvae)

VAL_START = "2016-01-01"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"


def plot_training(log, best_epoch, path):
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(log["epoch"], log["train_loss"], color="#2a78d6", lw=1.5, label="Fit data 2004-2015")
    ax.plot(log["epoch"], log["val_loss"], color="#eb6834", lw=1.5, label="Validation 2016-2018")
    ax.axvline(best_epoch, color=MUTED, lw=1, ls="--")
    losses = log.loc[log["beta"] == 1.0, ["train_loss", "val_loss"]]
    lo, hi = losses.min().min(), losses.quantile(0.95).max()
    ax.set_ylim(lo - 0.05 * (hi - lo), hi)
    ax.text(best_epoch, hi, f" best epoch {best_epoch}", color=MUTED, fontsize=9, va="top")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel("epoch", color=MUTED)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("VAE training: loss per day (negative ELBO, lower is better)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    train, _ = load_data()
    t0 = time.time()

    # Step 1: standardize with 2004-2015 statistics; samples whose target day is in 2016-2018 validate.
    fit = train[train.index < VAL_START]
    c, x = make_samples((train.to_numpy() - fit.mean().to_numpy()) / fit.std().to_numpy())
    is_val = train.index[WINDOW:] >= VAL_START
    model1, log1 = train_cvae(x[~is_val], c[~is_val], x[is_val], c[is_val])
    full_kl = log1[log1["beta"] == 1.0]
    best_epoch = int(full_kl.loc[full_kl["val_loss"].idxmin(), "epoch"])
    print(f"Step 1: {(~is_val).sum()} fit / {is_val.sum()} validation samples; best epoch {best_epoch} "
          f"(validation loss {full_kl['val_loss'].min():.3f}, validation IWAE NLL "
          f"{iwae_nll(model1, x[is_val], c[is_val]):.3f}), stopped after {len(log1)} epochs")

    # Step 2: retrain on all of 2004-2018 for best_epoch epochs.
    mean, std = train.mean().to_numpy(), train.std().to_numpy()
    c, x = make_samples((train.to_numpy() - mean) / std)
    model, log2 = train_cvae(x, c, epochs=best_epoch)
    ScenarioGenerator(model, train.columns, mean, std, c.min(axis=0), c.max(axis=0)).save(CHECKPOINT)

    kl = kl_per_dim(model, x, c)
    print(f"Step 2: {len(x)} samples, final training loss {log2['train_loss'].iloc[-1]:.3f}, "
          f"Student-t degrees of freedom nu = {model.nu.item():.2f}")
    print(f"KL per latent dimension: {np.round(kl, 3)} -> {(kl > 0.01).sum()} of {len(kl)} dimensions used")
    print(f"Weights checksum {checksum(model)}, {time.time() - t0:.0f} s, saved {CHECKPOINT}")
    plot_training(log1, best_epoch, "results/vae_training.png")


if __name__ == "__main__":
    main()
