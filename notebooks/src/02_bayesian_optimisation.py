# %% [markdown]
# # Exercise 2 — Tuning the beamline with Bayesian optimisation
#
# **ML for Accelerator Physics · beginner course · H. Guler (IJCLab)**
#
# {{COLAB}}
#
# On a real accelerator we usually have **no gradient**, each measurement is **noisy**, and
# each trial costs **beam time**. Bayesian optimisation (BO) is designed for exactly this:
# it finds a good setting with as few evaluations as possible.
#
# The recipe, repeated at every step:
#
# 1. fit a **Gaussian process** (GP) to all measurements so far — it gives a *prediction*
#    **and** an *uncertainty* everywhere;
# 2. use an **acquisition function** to decide where to measure next
#    (trade-off between *exploitation* — "it looks good here" — and *exploration* — "I don't know here");
# 3. measure, add the point, repeat.
#
# In this notebook you will code this loop yourself (≈ 30 lines), first in 1-D to see what
# happens, then on all three quadrupoles, and compare it with random search.

# %%
import sys
if "google.colab" in sys.modules:
    !pip install -q "cheetah-accelerator>=0.8,<0.9"

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import cheetah
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

SEED = 0
torch.manual_seed(SEED)

# %% [markdown]
# ## 1 · The machine (a simulated one)
#
# Same beamline as before. We add **3 % measurement noise**, like a real screen, and we
# count how many times the "machine" is used.

# %%
segment = cheetah.Segment(elements=[
    cheetah.Drift(length=torch.tensor(0.5)),
    cheetah.Quadrupole(length=torch.tensor(0.12), name="Q1"),
    cheetah.Drift(length=torch.tensor(0.4)),
    cheetah.Quadrupole(length=torch.tensor(0.12), name="Q2"),
    cheetah.Drift(length=torch.tensor(0.4)),
    cheetah.Quadrupole(length=torch.tensor(0.12), name="Q3"),
    cheetah.Drift(length=torch.tensor(1.5)),
])
incoming = cheetah.ParameterBeam.from_twiss(
    beta_x=torch.tensor(5.0), alpha_x=torch.tensor(-1.0), emittance_x=torch.tensor(2e-9),
    beta_y=torch.tensor(5.0), alpha_y=torch.tensor(-1.0), emittance_y=torch.tensor(2e-9),
    energy=torch.tensor(100e6),
)
K_MAX = 30.0
NOISE = 0.03
rng = np.random.default_rng(SEED)

def true_size(k):
    """Noise-free σx + σy in µm for k = (k1_Q1, k1_Q2, k1_Q3)."""
    segment.Q1.k1 = torch.tensor(float(k[0]))
    segment.Q2.k1 = torch.tensor(float(k[1]))
    segment.Q3.k1 = torch.tensor(float(k[2]))
    out = segment.track(incoming)
    return (out.sigma_x.item() + out.sigma_y.item()) * 1e6

def measure(k):
    """What the operator sees on the screen: noisy beam size in µm."""
    return true_size(k) * (1 + NOISE * rng.standard_normal())

print(f"all quads off : {measure([0, 0, 0]):.0f} µm")
print(f"a good setting: {measure([4.9, -13.5, 11.9]):.1f} µm")

# %% [markdown]
# As in Exercise 1, the beam size spans orders of magnitude, so we optimise its
# **logarithm**: `objective = log10(size)`. Smaller is better.

# %% [markdown]
# ## 2 · Warm-up in 1-D: tune only Q1
#
# We fix Q2 = −13.5 and Q3 = 12 and look for the best Q1. In 1-D we can *see* everything.

# %%
def objective_1d(k1):
    return np.log10(measure([k1, -13.5, 12.0]))

k_grid = np.linspace(-K_MAX, K_MAX, 400)
truth_1d = np.log10([true_size([k, -13.5, 12.0]) for k in k_grid])

# %% [markdown]
# ### The Gaussian process
#
# A GP is a probabilistic regression model: for every input it returns a mean μ(x) and a
# standard deviation σ(x). Its **kernel** encodes our prior belief: "nearby settings give
# similar beams". We use a Matérn kernel plus a `WhiteKernel` that learns the noise level.
# Inputs are rescaled to [−1, 1].

# %%
def fit_gp(X, y):
    d = X.shape[1]
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(1e-2, 1e1), nu=2.5)
              + WhiteKernel(1e-3, (1e-6, 1e-1)))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3,
                                  random_state=SEED)
    gp.fit(X / K_MAX, y)
    return gp

def gp_predict(gp, X):
    mu, sd = gp.predict(X / K_MAX, return_std=True)
    return mu, sd

# %% [markdown]
# ### The acquisition function: Expected Improvement
#
# Let y* be the best value observed so far. **Expected Improvement** (EI) is the average
# amount by which a new measurement at x would beat y*, according to the GP:
#
# $$ \mathrm{EI}(x) = (y^* - \mu)\,\Phi(z) + \sigma\,\varphi(z), \qquad z = \frac{y^* - \mu}{\sigma} $$
#
# (Φ, φ: cumulative and density of the normal distribution). EI is large where μ is low
# (exploitation) **or** where σ is large (exploration).

# %%
def expected_improvement(mu, sd, y_best, xi=0.01):
    sd = np.maximum(sd, 1e-9)
    z = (y_best - mu - xi) / sd
    return (y_best - mu - xi) * norm.cdf(z) + sd * norm.pdf(z)

# %% [markdown]
# ### Run it step by step
#
# Start from 3 random measurements, then let EI choose 7 more points.

# %%
X1 = rng.uniform(-K_MAX, K_MAX, size=(3, 1))
y1 = np.array([objective_1d(x[0]) for x in X1])

fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
for it, ax in enumerate(axes.T.ravel()):
    gp = fit_gp(X1, y1)
    mu, sd = gp_predict(gp, k_grid[:, None])
    ei = expected_improvement(mu, sd, y1.min())
    x_next = k_grid[np.argmax(ei)]

    ax.plot(k_grid, truth_1d, "k--", lw=1, label="truth (unknown)")
    ax.plot(k_grid, mu, "C0", label="GP mean")
    ax.fill_between(k_grid, mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.2, label="±2σ")
    ax.plot(X1[:, 0], y1, "ko", ms=4, label="measurements")
    ax2 = ax.twinx(); ax2.fill_between(k_grid, 0, ei, color="C1", alpha=0.3); ax2.set_yticks([])
    ax.axvline(x_next, color="C1", lw=1.5)
    ax.set_title(f"step {it}: {len(y1)} points — next: k1 = {x_next:.1f}", fontsize=9)
    ax.set_ylim(1.5, 3.6)

    X1 = np.vstack([X1, [[x_next]]])
    y1 = np.append(y1, objective_1d(x_next))
axes[0, 0].legend(fontsize=7, loc="upper left")
for ax in axes[-1]: ax.set_xlabel("k1 of Q1 [1/m²]")
fig.supylabel("log10(σx+σy / µm)"); plt.tight_layout(); plt.show()
print(f"best k1 found: {X1[np.argmin(y1), 0]:.2f}  →  {10**y1.min():.1f} µm (noisy)")

# %% [markdown]
# Observe how the orange EI curve (right axis) first explores the unknown regions, then
# concentrates around the minimum — and how the shaded uncertainty shrinks near measurements.

# %% [markdown]
# ### ✏️ Exercise 1 — another acquisition function
#
# A simple alternative is the **lower confidence bound** (for minimisation):
#
# $$ \mathrm{LCB}(x) = \mu(x) - \kappa\,\sigma(x) $$
#
# and the next point is where LCB is **smallest**.
#
# 1. Write `lcb(mu, sd, kappa)`.
# 2. Plot μ, μ ± 2σ and LCB for κ = 0.1 and κ = 5 using the GP fitted on the **first
#    3 points** of the run above. Where would each choice measure next? Which κ explores more?

# %%
def lcb(mu, sd, kappa=2.0):
    ### BEGIN SOLUTION
    return mu - kappa * sd
    ### END SOLUTION
    ### EXERCISE: return ...   # TODO

gp3 = fit_gp(X1[:3], y1[:3])
mu, sd = gp_predict(gp3, k_grid[:, None])
plt.figure(figsize=(7, 3.5))
plt.plot(k_grid, mu, "C0", label="GP mean")
plt.fill_between(k_grid, mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.2)
plt.plot(X1[:3, 0], y1[:3], "ko")
for kappa, c in [(0.1, "C2"), (5.0, "C3")]:
    a = lcb(mu, sd, kappa)
    plt.plot(k_grid, a, c, label=f"LCB κ={kappa}")
    plt.axvline(k_grid[np.argmin(a)], color=c, ls=":")
plt.xlabel("k1 of Q1 [1/m²]"); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()

# %% [markdown] solution
# **Answer.** With κ = 0.1, LCB ≈ μ: the next point is next to the best measurement so far
# (pure *exploitation*, risk of getting stuck). With κ = 5 the uncertainty dominates and the
# next point is far from existing measurements (*exploration*). EI makes this trade-off
# automatically; with LCB, κ is a knob you choose.

# %% [markdown]
# ## 3 · Now the real problem: three quadrupoles
#
# Same loop, in 3-D. To maximise EI we simply evaluate it on many random candidates
# (cheap: the GP is fast, the machine is not).

# %%
def bayes_opt(f, n_init=5, n_iter=35, dim=3, n_candidates=5000, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(-K_MAX, K_MAX, size=(n_init, dim))
    y = np.array([f(x) for x in X])
    for _ in range(n_iter):
        gp = fit_gp(X, y)
        cand = r.uniform(-K_MAX, K_MAX, size=(n_candidates, dim))
        # also look close to the current best point (helps fine-tuning)
        cand[: n_candidates // 5] = np.clip(
            X[np.argmin(y)] + 3.0 * r.standard_normal((n_candidates // 5, dim)), -K_MAX, K_MAX)
        mu, sd = gp_predict(gp, cand)
        x_next = cand[np.argmax(expected_improvement(mu, sd, y.min()))]
        X = np.vstack([X, x_next])
        y = np.append(y, f(x_next))
    return X, y

def random_search(f, n=40, dim=3, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(-K_MAX, K_MAX, size=(n, dim))
    return X, np.array([f(x) for x in X])

objective = lambda k: np.log10(measure(k))

X_bo, y_bo = bayes_opt(objective)
best = X_bo[np.argmin(y_bo)]
print("BO best setting:", np.round(best, 2), f"→ true size {true_size(best):.1f} µm "
      f"after {len(y_bo)} measurements")

# %% [markdown]
# ### ✏️ Exercise 2 — is BO better than random search?
#
# A single run proves nothing (noise, random start). Run **both** methods with 5 different
# seeds, 40 measurements each, and plot for each method the *true* beam size of the best
# point found so far versus the number of measurements (mean over seeds, and the spread).
#
# Hint: for a run `(X, y)`, the best point after `i+1` measurements is `X[np.argmin(y[:i+1])]`.

# %%
def best_so_far(X, y):
    return np.array([true_size(X[np.argmin(y[: i + 1])]) for i in range(len(y))])

curves = {"random search": [], "Bayesian optimisation": []}
### BEGIN SOLUTION
for seed in range(5):
    curves["random search"].append(best_so_far(*random_search(objective, n=40, seed=seed)))
    curves["Bayesian optimisation"].append(best_so_far(*bayes_opt(objective, seed=seed)))
### END SOLUTION
### EXERCISE: # TODO: fill curves[...] with one best_so_far curve per seed and per method

plt.figure(figsize=(6, 4))
for name, c in curves.items():
    c = np.array(c)
    n = np.arange(1, c.shape[1] + 1)
    plt.plot(n, np.median(c, 0), label=name)
    plt.fill_between(n, c.min(0), c.max(0), alpha=0.2)
plt.yscale("log"); plt.xlabel("number of measurements"); plt.ylabel("best true σx+σy [µm]")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.show()

# %% [markdown] solution
# **Answer.** After the few random initial points, BO quickly reaches beam sizes that random
# search typically does not find within 40 measurements, and the spread between runs is
# smaller. Note that BO may still end in the "62 µm" family of solutions rather than in the
# better one located at the power-supply limit: no optimiser is magic, the budget matters.

# %% [markdown]
# ### ✏️ Exercise 3 — noise
#
# Set `NOISE = 0.20` (20 % shot-to-shot jitter), re-run one BO and one random search, and
# compare the *noisy* best value `10**y.min()` with the *true* size of that point. What do you see?
# Do not forget to set `NOISE` back to 0.03 afterwards.

# %%
NOISE = 0.20
### BEGIN SOLUTION
for name, (X, y) in [("random", random_search(objective, seed=1)), ("BO", bayes_opt(objective, seed=1))]:
    b = X[np.argmin(y)]
    print(f"{name:7s}: best measured {10**y.min():6.1f} µm  — true size there {true_size(b):6.1f} µm")
### END SOLUTION
### EXERCISE: # TODO
NOISE = 0.03

# %% [markdown] solution
# **Answer.** With large noise the *best measured* value is optimistic: it is often a lucky
# fluctuation (the "winner's curse"), and the true size at that point is larger. The GP
# helps because its `WhiteKernel` learns the noise level and the mean μ averages
# neighbouring measurements: a better recommendation is the point with the lowest **GP mean**,
# not the lowest single measurement.

# %% [markdown]
# ## Take-away
#
# * BO = **surrogate with uncertainty** (GP) + **acquisition function** (where to look next).
# * Very sample-efficient for a handful (≲ 20) of parameters; handles noise naturally.
# * Its cost grows with the number of points (GP fit ∝ n³) and it struggles in high dimension.
# * In practice use a mature library: [Xopt](https://github.com/xopt-org/Xopt) (accelerator
#   community), [BoTorch](https://botorch.org).
#
# **In the literature:** Duris *et al.*, PRL **124**, 124801 (2020) — BO of the LCLS FEL;
# Roussel *et al.*, PRAB **27**, 084801 (2024) — review of BO for accelerator physics.
