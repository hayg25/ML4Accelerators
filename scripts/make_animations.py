"""Frame-by-frame figures for the animated slides (reveal.js r-stack + fragments).

Each animation is a series of images with identical size and axes, shown one after the
other on a click.  Usage:  python scripts/make_animations.py  → assets/anim/*.svg
"""
from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import optics
from make_figures import (NAVY, BLUE, ORANGE, RED, TEAL, GREY, K1D, sx_1d, data_1d)

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "assets" / "anim"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, transparent=False, facecolor="white")
    plt.close(fig)


def label(ax, text):
    ax.text(0.02, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=13,
            color=NAVY, bbox=dict(fc="white", ec="none", alpha=0.85))


# ------------------------------------------------------------------ loss: data → line → curve
def anim_loss():
    x, y = data_1d()
    c1 = np.polyfit(x, y, 1)
    nn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, tol=1e-7,
                                                      n_iter_no_change=200, random_state=0)).fit(x[:, None], y / 100)
    models = [None, ("model A: straight line", lambda u: np.polyval(c1, u), RED),
              ("model B: flexible curve", lambda u: nn.predict(np.asarray(u)[:, None]) * 100, TEAL)]
    for k, m in enumerate(models):
        fig, ax = plt.subplots(figsize=(9, 3.9))
        ax.plot(x, y, "o", color=NAVY, ms=7, zorder=3, label="25 measurements")
        if m is None:
            label(ax, "the data: what is the relation between k1 and σx?")
        else:
            name, f, col = m
            pred = f(x)
            for xi, yi, pi in zip(x, y, pred):
                ax.plot([xi, xi], [yi, pi], color=ORANGE, lw=1.8)
            ax.plot(K1D, f(K1D), color=col, lw=3)
            mse = np.mean((pred - y) ** 2)
            label(ax, f"{name}  —  mean squared error = {mse:,.0f} µm²")
        ax.set_xlim(-16, 26); ax.set_ylim(0, 800)
        ax.set_xlabel(r"$k_1$ of Q1 [1/m$^2$]   (input $x$)"); ax.set_ylabel(r"$\sigma_x$ [µm]   (target $y$)")
        save(fig, f"loss_{k}.svg")


# ------------------------------------------------------------------ gradient descent in 1-D, step by step
def anim_gd1d():
    L = lambda th: (th - 2.0) ** 2 + 0.5
    dL = lambda th: 2 * (th - 2.0)
    th = np.linspace(-3, 7, 300)
    p = [-2.0]
    for _ in range(8):
        p.append(p[-1] - 0.3 * dL(p[-1]))
    p = np.array(p)
    for k, n in enumerate([0, 1, 2, 3, 5, 8]):
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(th, L(th), color=GREY, lw=2.5)
        ax.plot(p[: n + 1], L(p[: n + 1]), "-o", color=ORANGE, ms=8, lw=1.8)
        ax.plot(p[n], L(p[n]), "o", color=NAVY, ms=13, zorder=4)
        # tangent = local slope
        s = dL(p[n]); xs = np.linspace(p[n] - 1.3, p[n] + 1.3, 10)
        ax.plot(xs, L(p[n]) + s * (xs - p[n]), "--", color=BLUE, lw=2)
        label(ax, f"step {n}:  slope = {s:+.1f}  →  move θ by −η·slope = {-0.3 * s:+.2f}" if n < 8
              else "after 8 steps: at the bottom, slope ≈ 0")
        ax.set_xlim(-3, 7); ax.set_ylim(0, 30)
        ax.set_xlabel(r"parameter $\theta$"); ax.set_ylabel(r"loss $L(\theta)$")
        save(fig, f"gd1d_{k}.svg")


# ------------------------------------------------------------------ gradient descent in 2-D on the beamline
def anim_gd2d():
    q1 = 4.9
    k2 = np.linspace(-30, 30, 121); k3 = np.linspace(-30, 30, 121)
    Z = np.array([[np.log10(optics.sigma_screen([q1, a, b]).sum() * 1e6) for a in k2] for b in k3])
    f = lambda k: np.log10(optics.sigma_screen([q1, *k]).sum() * 1e6)
    grad = lambda k, h=1e-4: np.array([(f(k + h * e) - f(k - h * e)) / (2 * h) for e in np.eye(2)])
    k = np.array([-20.0, -22.0]); path = [k.copy()]; m = v = np.zeros(2)
    for t in range(1, 151):
        g = grad(k); m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        k = k - 1.0 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8); path.append(k.copy())
    path = np.array(path)
    for i, n in enumerate([0, 5, 20, 150]):
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        cs = ax.contourf(k2, k3, Z, levels=30, cmap="viridis_r")
        ax.contour(k2, k3, Z, levels=12, colors="white", linewidths=0.4, alpha=0.6)
        ax.plot(path[: n + 1, 0], path[: n + 1, 1], "-", color=ORANGE, lw=2.5)
        ax.plot(*path[0], "o", color="white", mec=NAVY, ms=9)
        ax.plot(*path[n], "*" if n == 150 else "o", color=ORANGE, mec=NAVY, ms=18 if n == 150 else 10)
        ax.set_title(f"{n} steps" + ("  —  minimum reached" if n == 150 else ""), color=NAVY, fontsize=12)
        ax.set_xlabel(r"$k_1$ of Q2 [1/m$^2$]"); ax.set_ylabel(r"$k_1$ of Q3 [1/m$^2$]")
        cb = fig.colorbar(cs, ax=ax, shrink=0.85); cb.set_label(r"log$_{10}$(σx+σy / µm)")
        save(fig, f"gd2d_{i}.svg")


# ------------------------------------------------------------------ building a curve neuron by neuron
def anim_relu():
    x, y = data_1d(n=60, noise=0.0)
    xs, ys = x / 20.0, y / 500.0
    best = None
    for seed in range(20):
        m = MLPRegressor(hidden_layer_sizes=(6,), activation="relu", solver="lbfgs", max_iter=20000,
                         tol=1e-10, random_state=seed).fit(xs[:, None], ys)
        err = np.mean((m.predict(xs[:, None]) - ys) ** 2)
        if best is None or err < best[0]:
            best = (err, m)
    m = best[1]
    W1, b1, W2, b2 = m.coefs_[0][0], m.intercepts_[0], m.coefs_[1][:, 0], m.intercepts_[1][0]
    t = K1D / 20.0
    parts = [W2[j] * np.maximum(0, W1[j] * t + b1[j]) * 500 for j in range(6)]
    # add the most important neurons first (largest contribution), for a clear build-up
    order = np.argsort([-np.abs(p).max() for p in parts])
    cols = plt.cm.tab10(np.arange(6))
    truth = np.array([sx_1d(k) for k in K1D])
    for k in range(1, 7):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
        for i, j in enumerate(order[:k]):
            axes[0].plot(K1D, parts[j], color=cols[i], lw=2.2, label=f"neuron {i+1}")
        axes[0].axhline(0, color="#ccc", lw=1)
        axes[0].set_xlim(-15, 25); axes[0].set_ylim(-900, 900)
        axes[0].set_title(f"{k} hidden neuron{'s' if k > 1 else ''}: each adds one 'hinge'", color=NAVY, fontsize=13)
        axes[0].set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
        total = np.sum([parts[j] for j in order[:k]], axis=0) + b2 * 500
        axes[1].plot(K1D, truth, "--", color=GREY, lw=2, label="true σx")
        axes[1].plot(K1D, total, color=ORANGE, lw=2.8, label=f"sum of {k} hinge{'s' if k > 1 else ''} + bias")
        axes[1].set_xlim(-15, 25); axes[1].set_ylim(-100, 900)
        axes[1].set_title("network output = weighted sum", color=NAVY, fontsize=13)
        axes[1].set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); axes[1].set_ylabel("σx [µm]")
        axes[1].legend(fontsize=9, frameon=False, loc="upper center")
        save(fig, f"relu_{k}.svg")


# ------------------------------------------------------------------ training, epoch by epoch
def anim_training():
    x, y = data_1d(n=40, seed=7)
    xs, ys = ((x - 5) / 12)[:, None], (y - y.mean()) / y.std()
    m = MLPRegressor(hidden_layer_sizes=(32, 32), activation="relu", learning_rate_init=0.003,
                     max_iter=1, warm_start=True, random_state=1)
    t = ((K1D - 5) / 12)[:, None]
    snaps, losses = {}, []
    keep = (1, 5, 15, 60, 3000)
    for ep in range(1, 3001):
        m.fit(xs, ys)
        losses.append(np.mean((m.predict(xs) * y.std() + y.mean() - y) ** 2))
        if ep in keep:
            snaps[ep] = m.predict(t) * y.std() + y.mean()
    losses = np.array(losses)
    for i, ep in enumerate(keep):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1]))
        axes[0].plot(x, y, "o", color=NAVY, ms=5, label="training data")
        axes[0].plot(K1D, snaps[ep], color=ORANGE, lw=2.8, label="network prediction")
        axes[0].set_ylim(-100, 900); axes[0].set_xlim(-15, 25)
        axes[0].set_title(f"after {ep} epoch{'s' if ep > 1 else ''}", color=NAVY, fontsize=13)
        axes[0].set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); axes[0].set_ylabel("σx [µm]"); axes[0].legend(frameon=False)
        csum = np.cumsum(np.insert(losses, 0, 0.0))                  # causal running mean (25 epochs)
        idx = np.arange(1, len(losses) + 1)
        sm = (csum[idx] - csum[np.maximum(idx - 25, 0)]) / np.minimum(idx, 25)
        axes[1].loglog(np.arange(1, ep + 1), sm[:ep], color=ORANGE, lw=2)
        axes[1].plot(ep, sm[ep - 1], "o", color=NAVY, ms=8)
        axes[1].set_xlim(1, 3000); axes[1].set_ylim(losses.min() * 0.5, losses.max() * 2)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss [µm²]")
        axes[1].set_title("the loss goes down", color=NAVY, fontsize=13)
        save(fig, f"train_{i}.svg")


# ------------------------------------------------------------------ under / good / over-fitting
def anim_overfit():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 1, 15)); f = lambda t: np.sin(2 * np.pi * t)
    y = f(x) + 0.25 * rng.standard_normal(len(x)); t = np.linspace(0, 1, 400)
    for i, (deg, title, col) in enumerate([(1, "underfitting: too rigid", RED), (3, "good fit", TEAL),
                                           (14, "overfitting: fits the noise", RED)]):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(t, f(t), "--", color=GREY, lw=1.5, label="true law")
        ax.plot(t, np.polyval(np.polyfit(x, y, deg), t), color=col, lw=3, label=f"polynomial, degree {deg}")
        ax.plot(x, y, "o", color=NAVY, ms=7, label="noisy data")
        ax.set_ylim(-1.8, 1.8); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, color=NAVY, fontsize=14); ax.legend(frameon=False, loc="upper right", fontsize=10)
        save(fig, f"overfit_{i}.svg")


# ------------------------------------------------------------------ Bayesian optimisation, step by step
def anim_bo():
    rng = np.random.default_rng(0)
    truth = lambda k: np.log10(optics.sigma_screen([k, -13.5, 12.0]).sum() * 1e6)
    meas = lambda k: np.log10(10 ** truth(k) * (1 + 0.03 * rng.standard_normal()))
    grid = np.linspace(-30, 30, 400); tgrid = np.array([truth(k) for k in grid])
    X = rng.uniform(-30, 30, 3); y = np.array([meas(k) for k in X])
    for it in range(8):
        kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(0.5, (1e-2, 1e1), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
        gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3, random_state=0)
        gp.fit(X[:, None] / 30, y)
        mu, sd = gp.predict(grid[:, None] / 30, return_std=True)
        sdc = np.maximum(sd, 1e-9); z = (y.min() - mu - 0.01) / sdc
        ei = (y.min() - mu - 0.01) * norm.cdf(z) + sdc * norm.pdf(z)
        xn = grid[np.argmax(ei)]
        fig, ax = plt.subplots(figsize=(10, 3.9))
        ax.plot(grid, tgrid, "--", color=GREY, lw=1.3, label="true (unknown)")
        ax.plot(grid, mu, color=BLUE, lw=2.4, label="GP prediction")
        ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.15, label="uncertainty (±2σ)")
        ax.plot(X, y, "o", color=NAVY, ms=8, label="measurements", zorder=5)
        ax.plot(X[np.argmin(y)], y.min(), "*", color=ORANGE, mec=NAVY, ms=18, zorder=6, label="best so far")
        ax2 = ax.twinx(); ax2.fill_between(grid, 0, ei, color=ORANGE, alpha=0.35); ax2.set_axis_off()
        ax2.set_ylim(0, max(ei.max(), 1e-9) * 3)
        ax.axvline(xn, color=ORANGE, lw=2.2)
        ax.text(xn, 3.52, " next", color=ORANGE, fontsize=12, va="top", weight="bold")
        ax.set_title(f"{len(y)} measurements — best: {10 ** y.min():.0f} µm", color=NAVY, fontsize=13)
        ax.set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); ax.set_ylabel("log$_{10}$ beam size")
        ax.set_ylim(1.4, 3.6); ax.set_xlim(-30, 30)
        ax.legend(loc="upper center", fontsize=9, frameon=False, ncol=5, bbox_to_anchor=(0.5, -0.17))
        save(fig, f"bo_{it}.svg")
        X = np.append(X, xn); y = np.append(y, meas(xn))


# ------------------------------------------------------------------ Gaussian process: prior → posterior
def anim_gp():
    xg = np.linspace(-30, 30, 300)[:, None]
    truth = lambda k: np.log10(optics.sigma_screen([k, -13.5, 12.0]).sum() * 1e6)
    Xall = np.array([-22.0, 10.0, -8.0, 24.0])
    kern = ConstantKernel(0.25, "fixed") * Matern(length_scale=8.0, length_scale_bounds="fixed", nu=2.5)
    ymean = np.mean([truth(k) for k in Xall])
    for n in range(0, 5):
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if n == 0:
            samples = GaussianProcessRegressor(kern, optimizer=None).sample_y(xg, 6, random_state=1) + ymean
            mu, sd = np.full(len(xg), ymean), np.full(len(xg), 0.5)
        else:
            Xd = Xall[:n, None]; yd = np.array([truth(k) for k in Xall[:n]])
            gp = GaussianProcessRegressor(kern, optimizer=None, alpha=1e-4).fit(Xd, yd - ymean)
            mu, sd = gp.predict(xg, return_std=True); mu += ymean
            samples = gp.sample_y(xg, 6, random_state=1) + ymean
            ax.plot(Xd[:, 0], yd, "o", color=NAVY, ms=10, zorder=5, label="measurements")
        ax.plot(xg[:, 0], samples, lw=1.1, alpha=0.7)
        ax.fill_between(xg[:, 0], mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.13)
        ax.plot(xg[:, 0], mu, color=BLUE, lw=3, label="mean prediction")
        ax.set_ylim(0.8, 3.8); ax.set_xlim(-30, 30)
        ax.set_title("before any measurement: many smooth curves are possible" if n == 0 else
                     f"after {n} measurement{'s' if n > 1 else ''}: only curves through the points remain",
                     color=NAVY, fontsize=13)
        ax.set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); ax.set_ylabel("log$_{10}$ beam size")
        ax.legend(frameon=False, loc="lower left", fontsize=9)
        save(fig, f"gp_{n}.svg")


# ------------------------------------------------------------------ PCA on two BPMs
def anim_pca():
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 300)
    X = np.c_[a * 120, a * 85] + rng.normal(0, 8, (300, 2))
    out = np.array([90.0, -70.0])
    pca = PCA(1).fit(X); d = pca.components_[0]; m = pca.mean_
    ln = np.array([m - 400 * d, m + 400 * d]); proj = m + d * np.dot(out - m, d)
    for k in range(3):
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        ax.plot(X[:, 0], X[:, 1], "o", color=GREY, ms=4, alpha=0.6, label="normal shots")
        if k >= 1:
            ax.plot(ln[:, 0], ln[:, 1], color=BLUE, lw=2.5, label="1st principal component")
        if k >= 2:
            ax.plot(*out, "o", color=RED, ms=11, label="faulty shot")
            ax.plot([out[0], proj[0]], [out[1], proj[1]], color=RED, lw=2, ls="--")
            ax.text((out[0] + proj[0]) / 2 + 12, (out[1] + proj[1]) / 2 - 10, "reconstruction\nerror", color=RED, fontsize=11)
        ax.set_xlabel("BPM 3 reading [µm]"); ax.set_ylabel("BPM 4 reading [µm]")
        ax.set_xlim(-350, 350); ax.set_ylim(-260, 260); ax.legend(fontsize=9, frameon=False, loc="upper left")
        save(fig, f"pca_{k}.svg")


# ------------------------------------------------------------------ beamline: off → tuned
def anim_beamline():
    from make_figures import envelope
    for i, (k, title) in enumerate([((0, 0, 0), "quadrupoles off"),
                                    ((4.9, -13.5, 11.9), "tuned quadrupoles")]):
        fig, ax = plt.subplots(figsize=(10, 3.6))
        env = envelope(k)
        ax.plot(env[:, 0], env[:, 1] * 1e6, color=BLUE, lw=2.5, label=r"$\sigma_x$")
        ax.plot(env[:, 0], env[:, 2] * 1e6, color=ORANGE, lw=2.5, label=r"$\sigma_y$")
        for j, q0 in enumerate((0.5, 1.02, 1.54)):
            ax.add_patch(Rectangle((q0, 0), 0.12, 400, color=GREY, alpha=0.3, lw=0))
            ax.text(q0 + 0.06, 385, f"Q{j+1}", ha="center", va="top", fontsize=11, color=NAVY)
        ax.axvline(env[-1, 0], color=NAVY, lw=3)
        ax.text(env[-1, 0] - 0.04, 385, "screen", ha="right", va="top", fontsize=11, color=NAVY)
        ax.set_ylim(0, 400); ax.set_xlabel("position along the line  s  [m]")
        ax.set_ylabel("RMS beam size [µm]")
        tot = (env[-1, 1] + env[-1, 2]) * 1e6
        ax.set_title(f"{title}:  σx+σy = {tot:.0f} µm at the screen", fontsize=13, color=NAVY)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        save(fig, f"beamline_{i}.svg")


if __name__ == "__main__":
    anim_beamline()
    for f in (anim_loss, anim_gd1d, anim_gd2d, anim_relu, anim_training, anim_overfit, anim_bo, anim_gp, anim_pca):
        f(); print("done", f.__name__)
