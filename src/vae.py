"""Conditional VAE that generates scenarios for tomorrow's asset returns, given the last WINDOW days.

Workflow:
  python -m src.train_vae                         # train on 2004-2018 only, save checkpoints/vae.pt
  python -m src.vae_realism --experiment <name>   # check that generated scenarios look like real returns
  python -m src.run_backtests --experiment <name> # VAE variants run through the same backtest as the baselines

The three VAE variants backtested on 2019-2024 (the only ones, see README):
  VAEVaR                        "VAE-frozen"   trained once on 2004-2018
  RollingVAEVaR(refit_every=20)  "VAE-monthly"  retrained every 20 trading days on all data up to the day before
  RollingVAEVaR(refit_every=252) "VAE-yearly"   the same, every 252 trading days

Inside the model all returns are standardized: (r - training mean) / training std, per asset.
  condition c : summary of the last WINDOW days (see condition_features)
  encoder     : q(z | x, c), only needed during training
  decoder     : p(x | z, c) = multivariate Student-t; a network gives its center and scale
                (Cholesky factor), the tail thickness nu is learned
  prior       : z ~ N(0, I)
A scenario is made by drawing z from the prior, then x from the decoder's distribution.

Design choices, made on training data only (fit 2004-2015 / validate 2016-2018 and the realism
check on 2004-2018), never on the test period:
  - summary features instead of the raw 20-day window: the raw window overfitted within ~20 epochs
  - Student-t instead of normal decoder: with a normal decoder the scenarios had almost no fat tails
    (kurtosis ~0 on stormy days) and too many 99% violations on validation
  - "free bits" (each latent dimension gets FREE_BITS nats of KL for free during training): without
    it the model ignored z altogether (posterior collapse)
  - 60 instead of 20 days of memory: with 20 days, simulated crises died out too fast and the
    stock-bond correlation was too negative; same validation likelihood
  - condition features clipped to their training range when generating: without it, simulated
    paths that became more volatile than anything in training spiralled into 30-80% daily moves
"""
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import chi2
from torch import nn
from torch.distributions import Normal

from src.portfolio import portfolio_returns
from src.var_models import VaRModel, var_es_from_scenarios

SEED = 42
VOL_HORIZONS = (5, 20, 60)  # days over which realized volatility is measured
CORR_HORIZONS = (20, 60)  # days over which realized correlation is measured
WINDOW = max(VOL_HORIZONS)  # days of history the model sees
FREE_BITS = 0.1
EPOCHS = 77  # epochs of the frozen model (early stopping in train_vae); every retraining uses exactly this
CHECKPOINT =Path(__file__).resolve().parents[1] / "checkpoints" / "vae.pt"


def make_deterministic(seed=SEED):
    """Fix random seeds and use deterministic, single-threaded torch, so reruns match exactly."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def condition_features(windows):
    """windows: (n, WINDOW, assets) standardized returns -> (n, features) float32 array.

    Per asset: log realized volatility over the last 5, 20 and 60 days, the last day's return and
    the total return of the last 20 days (scaled by 1/sqrt(20)). Plus the realized correlation of
    each pair of assets over the last 20 and 60 days. For 2 assets: 5 * 2 + 2 = 12 features.
    """
    d = windows.shape[2]
    upper = np.triu_indices(d, k=1)
    features = [np.log(np.sqrt((windows[:, -h:] ** 2).mean(axis=1)) + 1e-6) for h in VOL_HORIZONS]
    features += [windows[:, -1], windows[:, -20:].sum(axis=1) / np.sqrt(20)]
    for h in CORR_HORIZONS:
        recent = windows[:, -h:]
        vol = np.sqrt((recent**2).mean(axis=1))
        corr = np.einsum("nwi,nwj->nij", recent, recent) / h / (vol[:, :, None] * vol[:, None, :] + 1e-12)
        features.append(corr[:, upper[0], upper[1]])
    return np.concatenate(features, axis=1).astype(np.float32)


def make_samples(z_returns, window=WINDOW):
    """z_returns: (T, assets) standardized returns -> (conditions, targets).
    Sample i uses days i .. i+window-1 as condition and day i+window as target."""
    idx = np.arange(len(z_returns) - window)[:, None] + np.arange(window)[None, :]
    return condition_features(z_returns[idx]), z_returns[window:].astype(np.float32)


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


def student_t_log_prob(x, center, L, nu):
    """Log density of a multivariate Student-t with scale matrix L L^T and nu degrees of freedom."""
    d = x.shape[-1]
    u = torch.linalg.solve_triangular(L, (x - center).unsqueeze(-1), upper=False).squeeze(-1)
    log_det = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return (torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2) - d / 2 * torch.log(nu * math.pi)
            - log_det - (nu + d) / 2 * torch.log1p((u**2).sum(-1) / nu))


class CVAE(nn.Module):
    def __init__(self, n_assets, n_cond, latent_dim=4, hidden=64):
        super().__init__()
        self.config = dict(n_assets=n_assets, n_cond=n_cond, latent_dim=latent_dim, hidden=hidden)
        self.d, self.latent_dim = n_assets, latent_dim
        self.encoder = mlp(n_assets + n_cond, hidden, 2 * latent_dim)
        self.decoder = mlp(latent_dim + n_cond, hidden, n_assets + n_assets * (n_assets + 1) // 2)
        self.rows, self.cols = torch.tril_indices(n_assets, n_assets)
        self.raw_nu = nn.Parameter(torch.tensor(2.0))  # nu = 2 + softplus(raw_nu), about 4 at the start

    @property
    def nu(self):
        """Degrees of freedom of the decoder's Student-t (lower = fatter tails)."""
        return 2.0 + F.softplus(self.raw_nu)

    def encode(self, x, c):
        """Mean and log-variance of q(z | x, c)."""
        return self.encoder(torch.cat([x, c], dim=1)).chunk(2, dim=1)

    def decode(self, z, c):
        """Center (n, assets) and lower-triangular scale factor (n, assets, assets) of p(x | z, c)."""
        out = self.decoder(torch.cat([z, c], dim=1))
        L = torch.zeros(len(out), self.d, self.d)
        L[:, self.rows, self.cols] = out[:, self.d:]
        diag = F.softplus(torch.diagonal(L, dim1=1, dim2=2)) + 1e-3  # positive scales
        return out[:, :self.d], torch.tril(L, -1) + torch.diag_embed(diag)

    def log_px(self, x, z, c):
        center, L = self.decode(z, c)
        return student_t_log_prob(x, center, L, self.nu)

    def loss(self, x, c, beta=1.0, free_bits=0.0, generator=None):
        """Reconstruction NLL + beta * KL, averaged over the batch. With free_bits = 0 and beta = 1
        this is the negative ELBO; during training each latent dimension's KL is floored at free_bits."""
        mu, logvar = self.encode(x, c)
        z = mu + torch.exp(0.5 * logvar) * torch.randn(mu.shape, generator=generator)
        nll = -self.log_px(x, z, c).mean()
        kl_per_dim = (0.5 * (mu**2 + logvar.exp() - 1 - logvar)).mean(dim=0)
        return nll + beta * torch.clamp(kl_per_dim, min=free_bits).sum(), nll, kl_per_dim.sum()

    @torch.no_grad()
    def sample(self, c, generator=None):
        """One standardized draw of x per row of c: z from the prior, then x from the decoder.
        A Student-t draw is center + L @ eps * sqrt(nu / chi2), with eps standard normal."""
        z = torch.randn(len(c), self.latent_dim, generator=generator)
        center, L = self.decode(z, c)
        eps = torch.randn(len(c), self.d, 1, generator=generator)
        u = torch.rand(len(c), generator=generator).double().clamp(1e-12, 1 - 1e-12).numpy()
        nu = self.nu.item()
        w = torch.as_tensor(np.sqrt(nu / chi2.ppf(u, nu)), dtype=torch.float32)
        return center + (L @ eps).squeeze(-1) * w[:, None]


@torch.no_grad()
def evaluate(model, x, c, seed=SEED):
    """Negative ELBO (beta = 1, no free bits) with fixed noise, so values are comparable across epochs."""
    loss, _, _ = model.loss(torch.as_tensor(x), torch.as_tensor(c), generator=torch.Generator().manual_seed(seed))
    return loss.item()


@torch.no_grad()
def iwae_nll(model, x, c, k=200, seed=SEED):
    """Negative log-likelihood per day, estimated with k importance samples (tighter than the ELBO,
    and fair when comparing models that use z with models that don't)."""
    x, c = torch.as_tensor(x), torch.as_tensor(c)
    gen = torch.Generator().manual_seed(seed)
    mu, logvar = model.encode(x, c)
    sd = torch.exp(0.5 * logvar)
    log_w = []
    for _ in range(k):
        z = mu + sd * torch.randn(mu.shape, generator=gen)
        log_w.append(model.log_px(x, z, c) + Normal(0, 1).log_prob(z).sum(1) - Normal(mu, sd).log_prob(z).sum(1))
    return -(torch.logsumexp(torch.stack(log_w), dim=0) - math.log(k)).mean().item()


def train_cvae(x, c, x_val=None, c_val=None, epochs=1000, patience=50, latent_dim=4, hidden=64,
               lr=1e-3, batch_size=256, weight_decay=1e-4, warmup=20, free_bits=FREE_BITS, seed=SEED):
    """Train a CVAE on targets x and conditions c (numpy arrays). Returns (model, per-epoch log).

    The KL weight beta rises from 0 to 1 over the first `warmup` epochs. With validation data,
    training stops once the validation loss (negative ELBO) has not improved for `patience` epochs
    (counting only epochs with beta = 1), and the best model is returned.
    """
    make_deterministic(seed)
    x, c = torch.as_tensor(x), torch.as_tensor(c)
    model = CVAE(x.shape[1], c.shape[1], latent_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, log = np.inf, None, 0, []
    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / warmup)
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), batch_size):
            batch = perm[i:i + batch_size]
            loss, _, _ = model.loss(x[batch], c[batch], beta, free_bits, gen)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # outliers such as AGG in 2008
            opt.step()

        row = {"epoch": epoch, "beta": beta, "train_loss": evaluate(model, x, c, seed)}
        if x_val is not None:
            row["val_loss"] = evaluate(model, x_val, c_val, seed)
        log.append(row)
        if x_val is not None and beta == 1.0:
            if row["val_loss"] < best_loss:
                best_loss, best_state, best_epoch = row["val_loss"], copy.deepcopy(model.state_dict()), epoch
            elif epoch - best_epoch >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(log)


@torch.no_grad()
def kl_per_dim(model, x, c):
    """Average KL per latent dimension; a dimension near 0 is not used by the model."""
    mu, logvar = model.encode(torch.as_tensor(x), torch.as_tensor(c))
    return (0.5 * (mu**2 + logvar.exp() - 1 - logvar)).mean(dim=0).numpy()


def checksum(model):
    """Short fingerprint of all weights, to confirm that retraining gives the identical model."""
    flat = torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()
    return hashlib.sha256(flat.tobytes()).hexdigest()[:12]


class ScenarioGenerator:
    """A trained CVAE plus the training standardization: turns recent history into scenarios.

    Condition features are clipped to [c_min, c_max], the range seen in training, so the network
    never has to extrapolate: a state more extreme than any training day is treated as the most
    extreme training state.
    """

    def __init__(self, model, assets, mean, std, c_min, c_max, window=WINDOW):
        self.model, self.assets, self.window = model.eval(), list(assets), window
        self.mean, self.std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        self.c_min, self.c_max = np.asarray(c_min, dtype=np.float32), np.asarray(c_max, dtype=np.float32)

    def standardize(self, returns):
        return (returns[self.assets].to_numpy() - self.mean) / self.std

    def conditions(self, z_windows):
        """(n, window, assets) standardized returns -> clipped condition features as a tensor."""
        return torch.as_tensor(np.clip(condition_features(z_windows), self.c_min, self.c_max))

    def is_clipped(self, history):
        """True if the market state at the end of `history` is outside the range seen in training,
        so that the condition features get clipped."""
        c = condition_features(self.standardize(history.iloc[-self.window:])[None])
        return bool(((c < self.c_min) | (c > self.c_max)).any())

    def sample(self, history, n, seed):
        """n scenarios of next-day simple returns (DataFrame), given the last `window` rows of history."""
        c = self.conditions(self.standardize(history.iloc[-self.window:])[None]).expand(n, -1)
        draws = self.model.sample(c, torch.Generator().manual_seed(seed)).numpy()
        return pd.DataFrame(draws * self.std + self.mean, columns=self.assets)

    def simulate_paths(self, init_window, n_steps, n_paths, seed):
        """Free-running simulation: each new day is drawn given the model's own previous `window`
        days, starting from the real days in init_window. Returns (n_paths, n_steps, assets)."""
        gen = torch.Generator().manual_seed(seed)
        z = np.repeat(self.standardize(init_window.iloc[-self.window:])[None], n_paths, axis=0)
        out = np.empty((n_paths, n_steps, len(self.assets)))
        for t in range(n_steps):
            out[:, t] = self.model.sample(self.conditions(z), gen).numpy()
            z = np.concatenate([z[:, 1:], out[:, t, None]], axis=1)
        return out * self.std + self.mean

    def save(self, path=CHECKPOINT):
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.model.config,
                    "assets": self.assets, "mean": self.mean.tolist(), "std": self.std.tolist(),
                    "c_min": self.c_min.tolist(), "c_max": self.c_max.tolist(), "window": self.window}, path)

    @classmethod
    def load(cls, path=CHECKPOINT):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Train the VAE first: python -m src.train_vae")
        ckpt = torch.load(path)
        model = CVAE(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["assets"], ckpt["mean"], ckpt["std"], ckpt["c_min"], ckpt["c_max"], ckpt["window"])


def fit_generator(returns, epochs=EPOCHS, seed=SEED):
    """Train a CVAE on all of `returns` for a fixed number of epochs (no validation) and return it
    as a ScenarioGenerator. Standardization and the clipping range come from the same data.
    Used for the frozen model (train_vae) and for every rolling retraining."""
    mean, std = returns.mean().to_numpy(), returns.std().to_numpy()
    c, x = make_samples((returns.to_numpy() - mean) / std)
    model, _ = train_cvae(x, c, epochs=epochs, seed=seed)
    return ScenarioGenerator(model, returns.columns, mean, std, c.min(axis=0), c.max(axis=0))


class VAEVaR(VaRModel):
    """Backtest plug-in "VAE-frozen": VaR/ES from n_scenarios VAE scenarios for the next day.

    The model is frozen (trained on 2004-2018 only); it only reads the last WINDOW days. Each day
    gets its own seed from the date, so reruns and different portfolios use the same scenarios.
    clipped_days lists the last data date of every forecast whose inputs had to be clipped.
    """

    def __init__(self, path=CHECKPOINT, n_scenarios=10_000, seed=SEED, name="VAE-frozen"):
        make_deterministic(seed)
        self.generator = ScenarioGenerator.load(path) if path else None
        self.n_scenarios, self.seed, self.name = n_scenarios, seed, name
        self.clipped_days = []

    def forecast(self, history, weights, alphas):
        if self.generator.is_clipped(history):
            self.clipped_days.append(history.index[-1])
        day_seed = self.seed + int(history.index[-1].strftime("%Y%m%d"))
        scenarios = self.generator.sample(history, self.n_scenarios, day_seed)
        port = portfolio_returns(scenarios, weights)
        return {a: var_es_from_scenarios(port, a) for a in alphas}


_trained = {}  # (last training date, seed) -> ScenarioGenerator; training is deterministic, so it is shared


class RollingVAEVaR(VAEVaR):
    """VAE retrained from scratch every `refit_every` forecasts on all data up to the day before,
    with exactly the frozen model's settings (architecture, features, EPOCHS epochs, seed).

    refit_every=20 is "VAE-monthly" (the same rhythm as GARCH), 252 is "VAE-yearly". `fits` records
    every retraining. Trained models are cached by their last training date, so the portfolios
    (and the two variants, where dates coincide) share the trainings.
    """

    def __init__(self, refit_every, name, n_scenarios=10_000, seed=SEED):
        super().__init__(path=None, n_scenarios=n_scenarios, seed=seed, name=name)
        self.refit_every, self._calls, self.fits = refit_every, 0, []

    def forecast(self, history, weights, alphas):
        if self._calls % self.refit_every == 0:
            key = (history.index[-1], self.seed)
            if key not in _trained:
                _trained[key] = fit_generator(history, seed=self.seed)
            self.generator = _trained[key]
            self.fits.append({"train_end": history.index[-1].date(), "train_days": len(history),
                              "checksum": checksum(self.generator.model)})
        self._calls += 1
        return super().forecast(history, weights, alphas)
