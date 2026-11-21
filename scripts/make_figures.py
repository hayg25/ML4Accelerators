"""Generate the static figures used in the slides (numpy / scikit-learn / matplotlib only).

Usage:  python scripts/make_figures.py      → writes assets/figs/*.svg
"""
from pathlib import Path
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

import optics

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "assets" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

NAVY, BLUE, ORANGE, RED, TEAL, GREY = "#14213d", "#2a6fdb", "#f08a24", "#d62839", "#1b998b", "#8d99ae"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#444",
    "axes.labelcolor": NAVY, "xtick.color": "#444", "ytick.color": "#444",
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
    "figure.dpi": 100, "savefig.bbox": "tight", "savefig.transparent": True,
})


def save(fig, name):
    fig.savefig(OUT / name)
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / name.replace(".svg", ".png"), dpi=70)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------- beamline
def envelope(k, n=200):
    L = optics.LAYOUT
    segs = [("d", L["d0"]), ("q", 0), ("d", L["d1"]), ("q", 1), ("d", L["d2"]), ("q", 2), ("d", L["d3"])]
    b, a, e = optics.BEAM["beta"], optics.BEAM["alpha"], optics.BEAM["emit"]
    S = [e * np.array([[b, -a], [-a, (1 + a * a) / b]]) for _ in range(2)]
    s, out = 0.0, [(0.0, np.sqrt(S[0][0, 0]), np.sqrt(S[1][0, 0]))]
    for kind, val in segs:
        length = val if kind == "d" else L["lq"]
        nstep = max(2, int(n * length / 3.0))
        for _ in range(nstep):
            dl = length / nstep
            for p in (0, 1):
                M = optics.drift(dl) if kind == "d" else optics.quad(dl, k[val])[p]
                S[p] = M @ S[p] @ M.T
            s += dl
            out.append((s, np.sqrt(S[0][0, 0]), np.sqrt(S[1][0, 0])))
    return np.array(out)


def fig_beamline():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, k, title in [(axes[0], (0, 0, 0), "quadrupoles off"),
                         (axes[1], (4.9, -13.5, 11.9), "tuned quadrupoles")]:
        env = envelope(k)
        ax.plot(env[:, 0], env[:, 1] * 1e6, color=BLUE, lw=2.5, label=r"$\sigma_x$")
        ax.plot(env[:, 0], env[:, 2] * 1e6, color=ORANGE, lw=2.5, label=r"$\sigma_y$")
        for i, q0 in enumerate((0.5, 1.02, 1.54)):
            ax.add_patch(Rectangle((q0, 0), 0.12, 400, color=GREY, alpha=0.3, lw=0))
            ax.text(q0 + 0.06, 385, f"Q{i+1}", ha="center", va="top", fontsize=11, color=NAVY)
        ax.axvline(env[-1, 0], color=NAVY, lw=3)
        ax.text(env[-1, 0] - 0.04, 385, "screen", ha="right", va="top", fontsize=11, color=NAVY)
        ax.set_ylim(0, 400); ax.set_xlabel("position along the line  s  [m]")
        tot = (env[-1, 1] + env[-1, 2]) * 1e6
        ax.set_title(f"{title}:  σx+σy = {tot:.0f} µm at screen", fontsize=13, color=NAVY)
    axes[0].set_ylabel("RMS beam size [µm]")
    axes[0].legend(loc="lower left", frameon=False)
    save(fig, "beamline.svg")


# ---------------------------------------------------------------------------- overfitting
def fig_overfit():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 1, 15))
    f = lambda t: np.sin(2 * np.pi * t)
    y = f(x) + 0.25 * rng.standard_normal(len(x))
    t = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, deg, title in zip(axes, (1, 3, 14), ("underfitting", "good fit", "overfitting")):
        c = np.polyfit(x, y, deg)
        ax.plot(t, f(t), "--", color=GREY, lw=1.5)
        ax.plot(t, np.polyval(c, t), color={1: RED, 3: TEAL, 14: RED}[deg], lw=2.5)
        ax.plot(x, y, "o", color=NAVY, ms=6)
        ax.set_ylim(-1.8, 1.8); ax.set_title(f"{title}  (degree {deg})", color=NAVY)
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "overfit.svg")


# ---------------------------------------------------------------------------- gradient descent
def fig_gd():
    q1 = 4.9
    k2 = np.linspace(-30, 30, 161); k3 = np.linspace(-30, 30, 161)
    Z = np.array([[np.log10(optics.sigma_screen([q1, a, b]).sum() * 1e6) for a in k2] for b in k3])
    f = lambda k: np.log10(optics.sigma_screen([q1, *k]).sum() * 1e6)
    def grad(k, h=1e-4):
        return np.array([(f(k + h * e) - f(k - h * e)) / (2 * h) for e in np.eye(2)])
    k = np.array([-20.0, -22.0]); path = [k.copy()]
    m = v = np.zeros(2)
    for t in range(1, 151):
        g = grad(k); m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        k = k - 1.0 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8); path.append(k.copy())
    path = np.array(path)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cs = ax.contourf(k2, k3, Z, levels=30, cmap="viridis_r")
    ax.contour(k2, k3, Z, levels=12, colors="white", linewidths=0.4, alpha=0.6)
    ax.plot(path[:, 0], path[:, 1], "-", color=ORANGE, lw=2.5)
    ax.plot(*path[0], "o", color="white", mec=NAVY, ms=9)
    ax.plot(*path[-1], "*", color=ORANGE, mec=NAVY, ms=18)
    ax.set_xlabel(r"$k_1$ of Q2 [1/m$^2$]"); ax.set_ylabel(r"$k_1$ of Q3 [1/m$^2$]")
    cb = fig.colorbar(cs, ax=ax, shrink=0.85); cb.set_label(r"loss = log$_{10}$(σx+σy / µm)")
    save(fig, "gradient_descent.svg")


# ---------------------------------------------------------------------------- neural network
def fig_nn():
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); ax.set_axis_off()
    layers = [3, 6, 6, 2]
    xs = [0, 2.2, 4.4, 6.6]
    pos = [[(x, (n - 1) / 2 - i) for i in range(n)] for x, n in zip(xs, layers)]
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=GREY, lw=0.6, alpha=0.6, zorder=1)
    colors = [BLUE, TEAL, TEAL, ORANGE]
    for layer, c in zip(pos, colors):
        for p in layer:
            ax.add_patch(Circle(p, 0.28, color=c, zorder=2))
    for p, lab in zip(pos[0], [r"$k_{Q1}$", r"$k_{Q2}$", r"$k_{Q3}$"]):
        ax.text(p[0] - 0.45, p[1], lab, ha="right", va="center", fontsize=15, color=NAVY)
    for p, lab in zip(pos[-1], [r"$\sigma_x$", r"$\sigma_y$"]):
        ax.text(p[0] + 0.45, p[1], lab, ha="left", va="center", fontsize=15, color=NAVY)
    for x, lab in zip(xs, ["inputs\n(magnet settings)", "hidden layer", "hidden layer", "outputs\n(beam)"]):
        ax.text(x, -3.3, lab, ha="center", va="top", fontsize=11, color=NAVY)
    ax.set_xlim(-1.6, 8.2); ax.set_ylim(-4.2, 3.0); ax.set_aspect("equal")
    save(fig, "neural_network.svg")


# ---------------------------------------------------------------------------- surrogate
def fig_surrogate():
    rng = np.random.default_rng(0)
    X = rng.uniform(-30, 30, (3000, 3))
    Y = np.log10(np.array([optics.sigma_screen(k) for k in X]) * 1e6)
    Xtr, Ytr, Xte, Yte = X[:2400], Y[:2400], X[2400:], Y[2400:]
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64, 64), max_iter=2000, tol=1e-6,
                                                       n_iter_no_change=50, random_state=0)).fit(Xtr, Ytr)
    P = mlp.predict(Xte)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.loglog(10 ** Yte[:, 0], 10 ** P[:, 0], "o", ms=3, color=BLUE, alpha=0.5)
    lims = [15, 1e4]; ax.plot(lims, lims, "--", color=NAVY, lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("σx from simulation [µm]"); ax.set_ylabel("σx from neural network [µm]")
    ax.text(20, 4e3, f"test set, R² = {r2_score(Yte, P):.3f}", fontsize=12, color=NAVY)
    save(fig, "surrogate_parity.svg")


# ---------------------------------------------------------------------------- Bayesian optimisation
def fig_bo():
    rng = np.random.default_rng(0)
    truth = lambda k: np.log10(optics.sigma_screen([k, -13.5, 12.0]).sum() * 1e6)
    meas = lambda k: np.log10(10 ** truth(k) * (1 + 0.03 * rng.standard_normal()))
    grid = np.linspace(-30, 30, 400); tgrid = np.array([truth(k) for k in grid])
    X = rng.uniform(-30, 30, 3); y = np.array([meas(k) for k in X])
    snaps = {}
    for it in range(8):
        kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(0.5, (1e-2, 1e1), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
        gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3, random_state=0)
        gp.fit(X[:, None] / 30, y)
        mu, sd = gp.predict(grid[:, None] / 30, return_std=True)
        sdc = np.maximum(sd, 1e-9); z = (y.min() - mu - 0.01) / sdc
        ei = (y.min() - mu - 0.01) * norm.cdf(z) + sdc * norm.pdf(z)
        xn = grid[np.argmax(ei)]
        if it in (0, 3, 7):
            snaps[it] = (X.copy(), y.copy(), mu, sd, ei, xn)
        X = np.append(X, xn); y = np.append(y, meas(xn))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (it, (Xs, ys, mu, sd, ei, xn)) in zip(axes, snaps.items()):
        ax.plot(grid, tgrid, "--", color=GREY, lw=1.3, label="true (unknown)")
        ax.plot(grid, mu, color=BLUE, lw=2, label="GP prediction")
        ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.15, label="uncertainty")
        ax.plot(Xs, ys, "o", color=NAVY, ms=6, label="measurements")
        ax2 = ax.twinx(); ax2.fill_between(grid, 0, ei, color=ORANGE, alpha=0.35); ax2.set_axis_off()
        ax2.set_ylim(0, ei.max() * 3)
        ax.axvline(xn, color=ORANGE, lw=2)
        ax.set_title(f"{len(ys)} measurements", color=NAVY)
        ax.set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); ax.set_ylim(1.4, 3.6)
    axes[0].set_ylabel("log$_{10}$ beam size")
    axes[0].legend(loc="upper center", fontsize=9, frameon=False, ncol=2)
    save(fig, "bayesian_optimisation.svg")


# ---------------------------------------------------------------------------- anomaly detection
def fig_anomaly():
    rng = np.random.default_rng(1)
    # orbit response of a FODO line to two kicks (numpy version of notebook 03)
    def orbit(a1, a2):
        x = np.zeros(2); out = []
        x[1] += a1; x = optics.drift(0.52) @ x; x[1] += a2; x = optics.drift(0.52) @ x
        for _ in range(5):
            for k in (10, -10):
                x = optics.quad(0.12, k)[0] @ x; out.append(x[0] * 1e6); x = optics.drift(1.0) @ x
        return np.array(out)
    K = rng.normal(0, 30e-6, (1500, 2))
    X = np.array([orbit(*k) for k in K]) + rng.normal(0, 5, (1500, 10))
    bad = rng.choice(1500, 60, replace=False); X[bad, rng.integers(0, 10, 60)] += rng.choice([-60, 60], 60)
    lab = np.zeros(1500, bool); lab[bad] = True
    pca = PCA(2).fit(X); R = X - pca.inverse_transform(pca.transform(X)); score = np.linalg.norm(R, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    for i in np.where(~lab)[0][:40]:
        axes[0].plot(X[i], color=GREY, alpha=0.4, lw=1)
    i = bad[0]; j = np.argmax(np.abs(R[i]))
    axes[0].plot(X[i], color=RED, lw=2.5); axes[0].plot(j, X[i, j], "x", color=RED, ms=14, mew=3)
    axes[0].set_xlabel("BPM index"); axes[0].set_ylabel("horizontal position [µm]")
    axes[0].set_title("one faulty shot hidden among normal orbits", color=NAVY)
    b = np.linspace(0, score.max(), 50)
    axes[1].hist(score[~lab], b, color=GREY, label="normal"); axes[1].hist(score[lab], b, color=RED, label="faulty BPM")
    axes[1].set_yscale("log"); axes[1].set_xlabel("anomaly score (PCA residual) [µm]")
    axes[1].legend(frameon=False); axes[1].set_title("unsupervised score separates them", color=NAVY)
    save(fig, "anomaly_detection.svg")


# ---------------------------------------------------------------------------- RL loop
def fig_rl():
    fig, ax = plt.subplots(figsize=(8, 3.6)); ax.set_axis_off()
    def box(x, y, w, h, text, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.25",
                                    fc=c, ec="none", alpha=0.95))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=13, weight="bold")
    box(0, 0, 3.8, 1.6, "Agent\n(policy = neural net)", BLUE)
    box(6.2, 0, 3.8, 1.6, "Environment\n(machine or Cheetah)", TEAL)
    def arrow(p, q, rad):
        ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=22, lw=2.2, color=NAVY))
    arrow((3.95, 1.35), (6.05, 1.35), -0.45)
    arrow((6.05, 0.25), (3.95, 0.25), -0.45)
    ax.text(5.0, 2.45, "action: change magnet settings", ha="center", fontsize=12, color=NAVY)
    ax.text(5.0, -0.95, "observation: beam image   +   reward: −beam size", ha="center", fontsize=12, color=NAVY)
    ax.set_xlim(-0.3, 10.3); ax.set_ylim(-1.3, 2.9)
    save(fig, "rl_loop.svg")


# ============================================================================ running 1-D example
# beam size sigma_x at the screen versus the strength of Q1 (Q2 = Q3 = 0)
K1D = np.linspace(-15, 25, 400)


def sx_1d(k):
    return optics.sigma_screen([k, 0.0, 0.0])[0] * 1e6


def data_1d(n=25, seed=4, noise=0.06):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-15, 25, n))
    y = np.array([sx_1d(k) for k in x]) * (1 + noise * rng.standard_normal(n))
    return x, y


def fig_loss():
    x, y = data_1d()
    t = K1D
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    c1 = np.polyfit(x, y, 1)
    c2 = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, tol=1e-7,
                                                      n_iter_no_change=200, random_state=0)).fit(x[:, None], y / 100)
    models = [("model A: a straight line", lambda u: np.polyval(c1, u), RED),
              ("model B: a flexible curve", lambda u: c2.predict(np.asarray(u)[:, None]) * 100, TEAL)]
    for ax, (title, f, col) in zip(axes, models):
        pred = f(x)
        for xi, yi, pi in zip(x, y, pred):
            ax.plot([xi, xi], [yi, pi], color=ORANGE, lw=1.5)
        ax.plot(t, f(t), color=col, lw=2.5)
        ax.plot(x, y, "o", color=NAVY, ms=6, zorder=3)
        mse = np.mean((pred - y) ** 2)
        ax.set_title(f"{title}\nmean squared error = {mse:,.0f} µm²", color=NAVY, fontsize=13)
        ax.set_xlabel(r"$k_1$ of Q1 [1/m$^2$]  (input $x$)")
    axes[0].set_ylabel(r"$\sigma_x$ at screen [µm]  (target $y$)")
    save(fig, "loss_residuals.svg")


def fig_gd1d():
    L = lambda th: (th - 2.0) ** 2 + 0.5
    dL = lambda th: 2 * (th - 2.0)
    th = np.linspace(-3, 7, 300)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, (eta, title) in zip(axes, [(0.05, "learning rate too small"), (0.3, "learning rate OK"),
                                       (1.02, "learning rate too large")]):
        ax.plot(th, L(th), color=GREY, lw=2)
        p = [-2.0]
        for _ in range(8):
            p.append(p[-1] - eta * dL(p[-1]))
        p = np.array(p)
        ax.plot(p, L(p), "-o", color=ORANGE, ms=6, lw=1.5)
        ax.plot(p[0], L(p[0]), "o", color=NAVY, ms=9)
        ax.set_title(f"{title}  (η = {eta})", color=NAVY, fontsize=13)
        ax.set_xlabel(r"parameter $\theta$"); ax.set_ylim(0, 30); ax.set_xlim(-3, 7)
    axes[0].set_ylabel(r"loss $L(\theta)$")
    save(fig, "gd_learning_rate.svg")


def fig_neuron():
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_axis_off()
    ins = [(r"$x_1 = k_{Q1}$", 2.6), (r"$x_2 = k_{Q2}$", 1.3), (r"$x_3 = k_{Q3}$", 0.0)]
    for (lab, y), w in zip(ins, ["$w_1$", "$w_2$", "$w_3$"]):
        ax.add_patch(Circle((0.6, y), 0.32, color=BLUE))
        ax.text(0.05, y, lab, ha="right", va="center", fontsize=15, color=NAVY)
        ax.add_patch(FancyArrowPatch((0.95, y), (3.55, 1.3), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=GREY))
        ax.text(2.1, y + (1.3 - y) * 0.45 + 0.18, w, ha="center", fontsize=15, color=ORANGE)
    ax.add_patch(Circle((4.0, 1.3), 0.45, color=TEAL))
    ax.text(4.0, 1.3, r"$\Sigma$", ha="center", va="center", fontsize=24, color="white")
    ax.text(4.0, 0.45, "+ bias $b$", ha="center", fontsize=13, color=NAVY)
    ax.add_patch(FancyArrowPatch((4.5, 1.3), (5.9, 1.3), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=GREY))
    ax.text(5.2, 1.5, "$z$", ha="center", fontsize=15, color=NAVY)
    ax.add_patch(FancyBboxPatch((6.0, 0.8), 1.4, 1.0, boxstyle="round,pad=0.05", fc=ORANGE, ec="none"))
    zz = np.linspace(-1, 1, 50)
    ax.plot(6.2 + 1.0 * (zz + 1) / 2, 0.95 + 0.6 * np.maximum(zz, 0), color="white", lw=3)
    ax.text(6.7, 2.05, "activation\n(e.g. ReLU)", ha="center", fontsize=12, color=NAVY)
    ax.add_patch(FancyArrowPatch((7.5, 1.3), (8.8, 1.3), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=GREY))
    ax.text(9.0, 1.3, "output $a$", ha="left", va="center", fontsize=15, color=NAVY)
    ax.text(5.0, -0.7, r"$z = w_1 x_1 + w_2 x_2 + w_3 x_3 + b \qquad a = \max(0, z)$",
            ha="center", fontsize=17, color=NAVY)
    ax.set_xlim(-1.8, 10.8); ax.set_ylim(-1.1, 3.2)
    save(fig, "neuron.svg")


def fig_activations():
    z = np.linspace(-4, 4, 300)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    for ax, (name, f, col) in zip(axes, [("ReLU:  max(0, z)", lambda u: np.maximum(0, u), ORANGE),
                                          ("sigmoid:  1 / (1 + e$^{-z}$)", lambda u: 1 / (1 + np.exp(-u)), BLUE),
                                          ("tanh(z)", np.tanh, TEAL)]):
        ax.axhline(0, color="#ccc", lw=1); ax.axvline(0, color="#ccc", lw=1)
        ax.plot(z, f(z), color=col, lw=3); ax.set_title(name, color=NAVY, fontsize=13)
        ax.set_xlabel("z")
    save(fig, "activations.svg")


def fig_relu_sum():
    """A 1-hidden-layer ReLU network with 6 neurons fitted to the running example."""
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
    W1, b1 = m.coefs_[0][0], m.intercepts_[0]
    W2, b2 = m.coefs_[1][:, 0], m.intercepts_[1][0]
    t = K1D / 20.0
    parts = [W2[j] * np.maximum(0, W1[j] * t + b1[j]) * 500 for j in range(len(b1))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    cols = plt.cm.tab10(np.arange(len(parts)))
    for j, part in enumerate(parts):
        axes[0].plot(K1D, part, color=cols[j], lw=2, label=f"neuron {j+1}")
    axes[0].axhline(0, color="#ccc", lw=1)
    axes[0].set_title("what each hidden neuron contributes\n(a 'hinge' = scaled, shifted ReLU)", color=NAVY, fontsize=13)
    axes[0].set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); axes[0].legend(fontsize=8, frameon=False, ncol=2)
    total = np.sum(parts, axis=0) + b2 * 500
    axes[1].plot(K1D, [sx_1d(k) for k in K1D], "--", color=GREY, lw=2, label="true σx (simulation)")
    axes[1].plot(K1D, total, color=ORANGE, lw=2.5, label="sum of the 6 hinges + bias")
    axes[1].plot(x, y, "o", color=NAVY, ms=3, label="training points")
    axes[1].set_title("their weighted sum = the network output", color=NAVY, fontsize=13)
    axes[1].set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); axes[1].set_ylabel("σx [µm]")
    axes[1].legend(fontsize=9, frameon=False)
    save(fig, "relu_sum.svg")


def fig_training():
    x, y = data_1d(n=40, seed=7)
    xs, ys = ((x - 5) / 12)[:, None], (y - y.mean()) / y.std()
    m = MLPRegressor(hidden_layer_sizes=(32, 32), activation="relu", learning_rate_init=0.003,
                     max_iter=1, warm_start=True, random_state=1)
    t = ((K1D - 5) / 12)[:, None]
    snaps, losses = {}, []
    for ep in range(1, 3001):
        m.fit(xs, ys)
        losses.append(np.mean((m.predict(xs) * y.std() + y.mean() - y) ** 2))
        if ep in (1, 5, 15, 3000):
            snaps[ep] = m.predict(t) * y.std() + y.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1]))
    axes[0].plot(x, y, "o", color=NAVY, ms=5, label="training data")
    for (ep, pred), col in zip(snaps.items(), [GREY, BLUE, TEAL, ORANGE]):
        axes[0].plot(K1D, pred, color=col, lw=2.2, label=f"after {ep} epoch{'s' if ep > 1 else ''}")
    axes[0].set_xlabel(r"$k_1$ of Q1 [1/m$^2$]"); axes[0].set_ylabel("σx [µm]")
    axes[0].set_title("the network's prediction during training", color=NAVY, fontsize=13)
    axes[0].legend(fontsize=9, frameon=False)
    axes[1].semilogy(np.arange(1, len(losses) + 1), losses, color=ORANGE, lw=2)
    axes[1].set_xscale("log"); axes[1].set_xlabel("epoch (one pass over the data)")
    axes[1].set_ylabel("loss (mean squared error) [µm²]")
    axes[1].set_title("the loss goes down", color=NAVY, fontsize=13)
    save(fig, "nn_training.svg")


def fig_bias_variance():
    rng = np.random.default_rng(3)
    f = lambda t: np.sin(2 * np.pi * t)
    xtr = np.sort(rng.uniform(0, 1, 15)); ytr = f(xtr) + 0.25 * rng.standard_normal(15)
    xte = rng.uniform(0, 1, 500); yte = f(xte) + 0.25 * rng.standard_normal(500)
    degs = np.arange(0, 15)
    tr, te = [], []
    for d in degs:
        c = np.polyfit(xtr, ytr, d)
        tr.append(np.mean((np.polyval(c, xtr) - ytr) ** 2)); te.append(np.mean((np.polyval(c, xte) - yte) ** 2))
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.semilogy(degs, tr, "o-", color=BLUE, lw=2, label="error on training data")
    ax.semilogy(degs, te, "o-", color=RED, lw=2, label="error on new (test) data")
    ax.axvspan(2.5, 5.5, color=TEAL, alpha=0.15)
    ax.text(4, 3e-3, "sweet\nspot", ha="center", color=TEAL, fontsize=12)
    ax.set_xlabel("model complexity (polynomial degree)"); ax.set_ylabel("mean squared error")
    ax.set_ylim(1e-3, 1e2); ax.legend(frameon=False, fontsize=11)
    save(fig, "train_vs_test.svg")


def fig_gp():
    rng = np.random.default_rng(2)
    xg = np.linspace(-30, 30, 300)[:, None]
    truth = lambda k: np.log10(optics.sigma_screen([k, -13.5, 12.0]).sum() * 1e6)
    Xd = np.array([-22.0, -8.0, 10.0, 24.0])[:, None]
    yd = np.array([truth(k) for k in Xd[:, 0]])
    kern = ConstantKernel(0.25, "fixed") * Matern(length_scale=8.0, length_scale_bounds="fixed", nu=2.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    prior = GaussianProcessRegressor(kern, optimizer=None)
    sp = prior.sample_y(xg, 5, random_state=1) + yd.mean()
    axes[0].plot(xg[:, 0], sp, lw=1.5, alpha=0.8)
    axes[0].fill_between(xg[:, 0], yd.mean() - 1.0, yd.mean() + 1.0, color=BLUE, alpha=0.12)
    axes[0].set_title("before any measurement:\nmany smooth curves are possible", color=NAVY, fontsize=13)
    post = GaussianProcessRegressor(kern, optimizer=None, alpha=1e-4).fit(Xd, yd - yd.mean())
    mu, sd = post.predict(xg, return_std=True); mu += yd.mean()
    ss = post.sample_y(xg, 5, random_state=1) + yd.mean()
    axes[1].plot(xg[:, 0], ss, lw=1, alpha=0.6)
    axes[1].fill_between(xg[:, 0], mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.15, label="±2σ uncertainty")
    axes[1].plot(xg[:, 0], mu, color=BLUE, lw=3, label="mean prediction")
    axes[1].plot(Xd[:, 0], yd, "o", color=NAVY, ms=9, zorder=5, label="4 measurements")
    axes[1].set_title("after 4 measurements:\nonly curves through the points remain", color=NAVY, fontsize=13)
    axes[1].legend(fontsize=9, frameon=False, loc="lower left")
    for ax in axes:
        ax.set_xlabel(r"$k_1$ of Q1 [1/m$^2$]")
    axes[0].set_ylabel("log$_{10}$ beam size")
    save(fig, "gaussian_process.svg")


def fig_pca2d():
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 300)
    X = np.c_[a * 120, a * 85] + rng.normal(0, 8, (300, 2))
    out = np.array([90.0, -70.0])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(X[:, 0], X[:, 1], "o", color=GREY, ms=4, alpha=0.6, label="normal shots")
    pca = PCA(1).fit(X); d = pca.components_[0]; m = pca.mean_
    ln = np.array([m - 400 * d, m + 400 * d])
    ax.plot(ln[:, 0], ln[:, 1], color=BLUE, lw=2.5, label="1st principal component")
    proj = m + d * np.dot(out - m, d)
    ax.plot(*out, "o", color=RED, ms=11, label="faulty shot")
    ax.plot([out[0], proj[0]], [out[1], proj[1]], color=RED, lw=2, ls="--")
    ax.text((out[0] + proj[0]) / 2 + 12, (out[1] + proj[1]) / 2 - 10, "reconstruction\nerror", color=RED, fontsize=11)
    ax.set_xlabel("BPM 3 reading [µm]"); ax.set_ylabel("BPM 4 reading [µm]")
    ax.set_xlim(-350, 350); ax.set_ylim(-260, 260); ax.legend(fontsize=9, frameon=False, loc="upper left")
    save(fig, "pca_2d.svg")


def fig_iforest():
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, (120, 2)); out = np.array([3.2, 2.6])
    inl = X[np.argmin(np.linalg.norm(X, axis=1))]
    pts = np.vstack([X, out])

    def isolate(target, seed, draw_ax=None):
        r = np.random.default_rng(seed); box = [-4.0, 4.0, -4.0, 4.0]; cur = pts.copy(); n = 0
        while len(cur) > 1 and n < 40:
            dim = r.integers(2)
            lo, hi = cur[:, dim].min(), cur[:, dim].max()
            if hi - lo < 1e-12:
                break
            cut = r.uniform(lo, hi)
            if draw_ax is not None:
                if dim == 0:
                    draw_ax.plot([cut, cut], box[2:], color=BLUE, lw=1.2)
                else:
                    draw_ax.plot(box[:2], [cut, cut], color=BLUE, lw=1.2)
            side = target[dim] < cut
            cur = cur[(cur[:, dim] < cut) == side]
            if dim == 0:
                box = [box[0], cut, box[2], box[3]] if side else [cut, box[1], box[2], box[3]]
            else:
                box = [box[0], box[1], box[2], cut] if side else [box[0], box[1], cut, box[3]]
            n += 1
        return n

    avg_out = np.mean([isolate(out, s) for s in range(300)])
    avg_in = np.mean([isolate(inl, s) for s in range(300)])
    # pick a typical tree for each drawing
    s_out = min(range(300), key=lambda s: abs(isolate(out, s) - avg_out))
    s_in = min(range(300), key=lambda s: abs(isolate(inl, s) - avg_in))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, target, seed, avg, title in [(axes[0], out, s_out, avg_out, "anomaly: isolated after few random cuts"),
                                         (axes[1], inl, s_in, avg_in, "normal point: needs many cuts")]:
        ax.plot(pts[:, 0], pts[:, 1], "o", color=GREY, ms=4)
        n = isolate(target, seed, draw_ax=ax)
        ax.plot(*target, "o", color=RED, ms=10, zorder=5)
        ax.set_title(f"{title}\n{n} cuts in this tree · {avg:.1f} on average (300 trees)", color=NAVY, fontsize=12)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "isolation_forest.svg")


if __name__ == "__main__":
    fig_beamline(); fig_overfit(); fig_gd(); fig_nn(); fig_surrogate(); fig_bo(); fig_anomaly(); fig_rl()
    fig_loss(); fig_gd1d(); fig_neuron(); fig_activations(); fig_relu_sum(); fig_training()
    fig_bias_variance(); fig_gp(); fig_pca2d(); fig_iforest()
