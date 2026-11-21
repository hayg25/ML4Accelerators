# %% [markdown]
# # Exercise 5 — A surrogate model of a reactor-physics code
#
# **ML for Accelerator Physics · nuclear-energy track · H. Guler (IJCLab)**
#
# {{COLAB}}
#
# **Why this matters.** Core design, fuel-loading optimisation, safety margins and uncertainty
# studies all need *thousands* of neutronics calculations. A high-fidelity Monte Carlo run
# (Serpent, OpenMC, MCNP, TRIPOLI-4…) or a coupled multiphysics calculation can take hours.
# A **surrogate model** trained on a few dozen well-chosen runs can answer in microseconds,
# **and tell you how confident it is** — which is exactly what a Gaussian process does.
#
# In this notebook you will:
#
# 1. explore a (toy) pin-cell neutronics code and compute reactivity coefficients,
# 2. design 40 "expensive" runs with a **Latin hypercube**,
# 3. train a **Gaussian process** surrogate and check it honestly,
# 4. use it for a **critical boron search** and an **uncertainty propagation** with 100 000 samples,
# 5. compare with a neural network trained on the same 40 runs.
#
# No installation needed: only numpy, scipy, scikit-learn and matplotlib (preinstalled on Colab).

# %%
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.optimize import brentq
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # hyper-parameters at their bounds: harmless here

SEED = 0
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 1 · The "expensive code"
#
# We cannot run Serpent on Colab, so we use a **toy pin-cell model** of a pressurised-water
# reactor based on the textbook **six-factor formula**:
#
# $$ k_\text{eff} = \underbrace{\eta\, f\, p\, \varepsilon}_{k_\infty} \times P_\text{fast NL}\, P_\text{thermal NL} $$
#
# * η — neutrons per thermal absorption in fuel (depends on **enrichment**)
# * f — thermal utilisation (fuel vs. moderator, structures, **boron**)
# * p — resonance-escape probability (moderation ratio, **Doppler** effect of fuel temperature)
# * ε — fast-fission factor; P — non-leakage probabilities of a large core
#
# Inputs (5): enrichment [%], moderator-to-fuel volume ratio Vm/Vf, fuel temperature Tf [K],
# moderator temperature Tm [°C], soluble boron [ppm]. Output: k_eff.
#
# To mimic a Monte Carlo code, each run returns k_eff with a **statistical uncertainty of
# 30 pcm** (1 pcm = 10⁻⁵), and we count the runs: **pretend each one costs 1 hour of cluster time**.
#
# > ⚠️ The model is hand-made for teaching: physically sensible trends and orders of magnitude
# > (k∞ ≈ 1.17 without boron, critical boron ≈ 1650 ppm, boron worth ≈ −8 pcm/ppm,
# > Doppler ≈ −3 pcm/K), but **not** a design tool.

# %%
NA = 0.6022                       # Avogadro × 1e-24 → densities in 1/(barn·cm)

def water_density(Tm_C):
    """Approximate density of pressurised water (155 bar) in g/cm³, 250–330 °C."""
    return 0.7650 - 1.85e-3 * (Tm_C - 300) - 1.2e-5 * (Tm_C - 300) ** 2

def pincell_keff(enr, vmvf, Tf, Tm_C, boron, factors=False):
    """Toy PWR pin-cell + leakage model (noise-free). Works with numpy arrays."""
    e = enr / 100.0
    rho_f = 10.4                                           # UO2 density g/cm³
    M_U = 1 / (e / 235.04 + (1 - e) / 238.05)
    N_UO2 = rho_f * NA / (M_U + 32.0)
    N25, N28 = N_UO2 * e * M_U / 235.04, N_UO2 * (1 - e) * M_U / 238.05
    rho_m = water_density(Tm_C)
    N_w = rho_m * NA / 18.015
    N_B10 = boron * 1e-6 * rho_m * NA / 10.81 * 0.199      # natural boron, ppm by mass
    g = np.sqrt(np.pi) / 2 * np.sqrt(293.6 / (Tm_C + 303.15))   # 1/v Maxwellian average
    sf25, sa25, sa28 = 571.1 * g, 665.9 * g, 2.68 * g      # thermal cross sections [barn]
    saH, saO, sa10 = 0.332 * g, 1.9e-4 * g, 3840.0 * g
    Sa_fuel = N25 * sa25 + N28 * sa28 + 2 * N_UO2 * saO
    Sa_mod = N_w * (2 * saH + saO) + N_B10 * sa10
    Sa_par = 0.20 * Sa_fuel                                # cladding, structures, Xe/Sm (lumped)
    f = Sa_fuel / (Sa_fuel + Sa_par + 1.10 * vmvf * Sa_mod)
    eta = 2.43 * N25 * sf25 / Sa_fuel
    SM = 2 / (0.41 * rho_f)                                # surface/mass of a 4.1 mm pellet
    Ieff = (4.45 + 26.6 * np.sqrt(SM)) * (1 + (6.1e-3 + 4.7e-3 * SM) * (np.sqrt(Tf) - np.sqrt(300.0)))
    p = np.exp(-N28 * Ieff / (1.35 * rho_m * vmvf))        # resonance escape (Doppler via Ieff)
    eps = 1 + 0.07 / (1 + 0.6 * vmvf)
    kinf = eta * f * p * eps
    B2 = (2.405 / 170.0) ** 2 + (np.pi / 370.0) ** 2       # buckling of a large PWR core
    keff = kinf * np.exp(-B2 * 27.0 / rho_m ** 2) / (1 + (1 - f) * 2.85 / rho_m ** 2 * B2)
    if factors:
        return dict(eta=eta, f=f, p=p, eps=eps, kinf=kinf, keff=keff)
    return keff

N_RUNS = 0
MC_SIGMA = 30e-5                                           # 30 pcm statistical uncertainty

def run_code(x):
    """One 'expensive' run. x = (enr %, Vm/Vf, Tf K, Tm °C, boron ppm) → k_eff ± 30 pcm."""
    global N_RUNS
    N_RUNS += 1
    return float(pincell_keff(*x) + MC_SIGMA * rng.standard_normal())

NOMINAL = np.array([3.5, 2.0, 900.0, 305.0, 1500.0])
NAMES = ["enrichment [%]", "Vm/Vf", "T_fuel [K]", "T_mod [°C]", "boron [ppm]"]
BOUNDS = np.array([[2.0, 5.0], [1.4, 2.6], [600.0, 1200.0], [280.0, 320.0], [0.0, 2500.0]])

for k, v in pincell_keff(*NOMINAL, factors=True).items():
    print(f"{k:5s} = {v:.4f}")

# %% [markdown]
# Reactivity is usually expressed as ρ = (k − 1)/k, in **pcm** (10⁻⁵).

# %%
def pcm(k):
    return (np.asarray(k) - 1) / np.asarray(k) * 1e5

# %% [markdown]
# ### ✏️ Exercise 1 — physics first: reactivity coefficients
#
# Using **noise-free** `pincell_keff` (cheap here), compute around the nominal point:
#
# * the **boron worth** in pcm/ppm (ρ difference for +100 ppm, divided by 100),
# * the **Doppler coefficient** in pcm/K (ρ difference for +100 K of fuel temperature, divided by 100),
#
# and plot k_eff versus boron concentration from 0 to 2500 ppm.

# %%
### BEGIN SOLUTION
def coefficient(i, delta):
    x1 = NOMINAL.copy(); x1[i] += delta
    return (pcm(pincell_keff(*x1)) - pcm(pincell_keff(*NOMINAL))) / delta

boron_worth = coefficient(4, 100.0)
doppler = coefficient(2, 100.0)
### END SOLUTION
### EXERCISE: boron_worth = ...   # TODO [pcm/ppm]
### EXERCISE: doppler = ...       # TODO [pcm/K]
print(f"boron worth : {boron_worth:6.2f} pcm/ppm")
print(f"Doppler     : {doppler:6.2f} pcm/K")

B = np.linspace(0, 2500, 100)
plt.figure(figsize=(5.5, 3.5))
plt.plot(B, pincell_keff(*NOMINAL[:4], B))
plt.axhline(1, color="k", lw=0.8, ls="--")
plt.xlabel("boron [ppm]"); plt.ylabel("k_eff"); plt.grid(alpha=0.3); plt.show()

# %% [markdown] solution
# **Answer.** ≈ −8 pcm/ppm and ≈ −3 pcm/K: both negative, i.e. stabilising feedbacks, with the
# orders of magnitude of a real PWR. Here we could afford finite differences because the model
# is instantaneous. With a 1-hour Monte Carlo run with ±30 pcm noise, a +100 K Doppler
# difference (≈ 300 pcm) needs several runs per point… and a full core design needs this for
# many conditions. This is where a surrogate pays off.

# %% [markdown]
# ## 2 · Design of experiments: where to spend 40 runs?
#
# With a budget of 40 runs in 5 dimensions, a regular grid is impossible (even 3 values per
# input = 3⁵ = 243 runs). A **Latin hypercube** spreads the points so that every input is
# sampled evenly over its range.

# %%
N_TRAIN = 40
sampler = qmc.LatinHypercube(d=5, seed=SEED)
X_train = qmc.scale(sampler.random(N_TRAIN), BOUNDS[:, 0], BOUNDS[:, 1])

N_RUNS = 0
y_train = np.array([run_code(x) for x in X_train])
print(f"{N_RUNS} runs done  (≈ {N_RUNS} h of cluster time)")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, (i, j) in zip(axes, [(0, 1), (0, 4), (2, 3)]):
    ax.scatter(X_train[:, i], X_train[:, j], c=y_train, cmap="viridis", s=25)
    ax.set_xlabel(NAMES[i]); ax.set_ylabel(NAMES[j])
plt.suptitle("40 training runs (colour = k_eff)"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3 · Gaussian-process surrogate
#
# Same tool as in the Bayesian-optimisation exercise. Two details matter here:
#
# * **ARD kernel**: one length scale *per input*. A short length scale = k_eff changes quickly
#   with that input. After fitting, the length scales tell you which inputs matter.
# * **WhiteKernel**: learns the Monte Carlo noise, so the GP does not chase statistical fluctuations.
#
# Inputs are rescaled to [0, 1] before fitting.

# %%
def to_unit(X):
    return (np.asarray(X) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])

kernel = (ConstantKernel(1.0, (1e-2, 1e4))
          * Matern(length_scale=np.ones(5), length_scale_bounds=(1e-2, 1e2), nu=2.5)
          + WhiteKernel(1e-4, (1e-12, 1e-2)))
gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=5, random_state=SEED)
gp.fit(to_unit(X_train), y_train)
print(gp.kernel_)

ls = gp.kernel_.k1.k2.length_scale
for n, l in sorted(zip(NAMES, ls), key=lambda t: t[1]):
    print(f"  length scale {n:16s}: {l:7.2f}")

# %% [markdown]
# ### ✏️ Exercise 2 — is the surrogate any good?
#
# In real life you would hold back a few runs for validation. Here the code is cheap, so we
# can afford an honest test on 300 new random settings (noise-free reference).
#
# 1. Predict mean **and** standard deviation (`return_std=True`) on `X_test`.
# 2. Compute the RMS error in **pcm** and make a parity plot.
# 3. Check the uncertainty: which fraction of test points fall within ±2σ of the prediction?
#    (For a well-calibrated model: ≈ 95 %.)

# %%
X_test = qmc.scale(qmc.LatinHypercube(d=5, seed=1).random(300), BOUNDS[:, 0], BOUNDS[:, 1])
k_true = pincell_keff(*X_test.T)
### BEGIN SOLUTION
k_pred, k_std = gp.predict(to_unit(X_test), return_std=True)
err_pcm = (k_pred - k_true) * 1e5
rms = np.sqrt(np.mean(err_pcm ** 2))
coverage = np.mean(np.abs(k_pred - k_true) < 2 * k_std)

plt.figure(figsize=(4.5, 4.5))
plt.errorbar(k_true, k_pred, yerr=2 * k_std, fmt=".", alpha=0.5, ms=3)
plt.plot([0.7, 1.5], [0.7, 1.5], "k--", lw=1)
plt.xlabel("k_eff (code)"); plt.ylabel("k_eff (GP surrogate)"); plt.grid(alpha=0.3); plt.show()
### END SOLUTION
### EXERCISE: k_pred, k_std = ...    # TODO
### EXERCISE: rms = ...              # TODO: RMS error in pcm
### EXERCISE: coverage = ...         # TODO: fraction within ±2σ
print(f"RMS error: {rms:.0f} pcm over a k_eff range of {(k_true.max()-k_true.min())*1e5:.0f} pcm")
print(f"fraction of test points within ±2σ: {coverage*100:.0f} %")

# %% [markdown] solution
# **Answer.** With only 40 runs the GP reproduces k_eff over the whole 5-D domain to within a
# few hundred pcm, and its error bars are roughly calibrated. The ARD length scales are
# physically meaningful: boron and enrichment (and the moderation ratio) have the shortest
# length scales, the moderator and fuel temperatures the longest — within these ranges,
# the temperatures matter least. A designer would now add runs where the GP is most uncertain
# (Exercise 6).

# %% [markdown]
# ### ✏️ Exercise 3 — why not a neural network?
#
# Train an `MLPRegressor` (e.g. two hidden layers of 32 neurons, inputs scaled) on the **same
# 40 runs** and compute its RMS error on the test set. Compare with the GP.

# %%
### BEGIN SOLUTION
mlp = make_pipeline(StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, tol=1e-7,
                                 n_iter_no_change=200, random_state=SEED))
mlp.fit(X_train, y_train)
rms_mlp = np.sqrt(np.mean(((mlp.predict(X_test) - k_true) * 1e5) ** 2))
### END SOLUTION
### EXERCISE: rms_mlp = ...   # TODO
print(f"RMS error  GP: {rms:.0f} pcm   |   MLP: {rms_mlp:.0f} pcm   (same 40 runs)")

# %% [markdown] solution
# **Answer.** The neural network is much worse with 40 points: it has thousands of parameters
# and no notion of smoothness or uncertainty. Gaussian processes shine for **small, expensive
# datasets** (tens to a few thousand points); neural networks for **large** ones (they scale
# better, see the accelerator surrogate exercise with 2400 runs). Choose the tool from the
# data budget.

# %% [markdown]
# ## 4 · Using the surrogate
#
# ### 4a · Critical boron search
#
# At nominal conditions, which boron concentration makes the reactor exactly critical
# (k_eff = 1)? With the surrogate this is a root-finding problem costing **zero** extra runs,
# and the GP gives an uncertainty band.

# %%
def gp_at_boron(b):
    x = NOMINAL.copy(); x[4] = b
    m, s = gp.predict(to_unit(x[None]), return_std=True)
    return m[0], s[0]

b_grid = np.linspace(0, 2500, 200)
m, s = np.array([gp_at_boron(b) for b in b_grid]).T
cb_gp = brentq(lambda b: gp_at_boron(b)[0] - 1, 0, 2500)
cb_lo = brentq(lambda b: sum(gp_at_boron(b) * np.array([1, -2])) - 1, 0, 2500)
cb_hi = brentq(lambda b: sum(gp_at_boron(b) * np.array([1, 2])) - 1, 0, 2500)
cb_true = brentq(lambda b: pincell_keff(*NOMINAL[:4], b) - 1, 0, 2500)

plt.figure(figsize=(6, 3.5))
plt.plot(b_grid, pincell_keff(*NOMINAL[:4], b_grid), "k--", lw=1, label="code (noise-free)")
plt.plot(b_grid, m, label="GP mean"); plt.fill_between(b_grid, m - 2 * s, m + 2 * s, alpha=0.25, label="±2σ")
plt.axhline(1, color="grey", lw=0.8); plt.axvline(cb_gp, color="C1", lw=1)
plt.xlabel("boron [ppm]"); plt.ylabel("k_eff"); plt.legend(); plt.grid(alpha=0.3); plt.show()
print(f"critical boron (GP)  : {cb_gp:.0f} ppm   (range {min(cb_lo, cb_hi):.0f}–{max(cb_lo, cb_hi):.0f} ppm)")
print(f"critical boron (code): {cb_true:.0f} ppm")

# %% [markdown]
# ### 4b · Uncertainty propagation
#
# Manufacturing tolerances and operating conditions are not exactly known. Assume (1σ):
#
# | input | nominal | uncertainty |
# |:--|--:|--:|
# | enrichment | 3.5 % | ±0.03 % |
# | Vm/Vf | 2.0 | ±0.02 |
# | T_fuel | 900 K | ±50 K |
# | T_mod | 305 °C | ±2 °C |
# | boron | 1500 ppm | ±15 ppm |
#
# What is the resulting uncertainty on k_eff? Monte Carlo propagation needs ~10⁵ evaluations:
# **10⁵ hours** with the code, **< 1 s** with the surrogate.

# %%
SIGMA_IN = np.array([0.03, 0.02, 50.0, 2.0, 15.0])
N_UQ = 100_000
X_uq = NOMINAL + SIGMA_IN * rng.standard_normal((N_UQ, 5))

t0 = time.perf_counter()
k_uq = gp.predict(to_unit(X_uq))
print(f"{N_UQ} surrogate evaluations in {time.perf_counter() - t0:.2f} s")
print(f"k_eff = {k_uq.mean():.5f} ± {k_uq.std()*1e5:.0f} pcm (surrogate)")
k_uq_ref = pincell_keff(*X_uq.T)
print(f"k_eff = {k_uq_ref.mean():.5f} ± {k_uq_ref.std()*1e5:.0f} pcm (code, only possible because it is a toy)")

plt.figure(figsize=(5.5, 3.5))
plt.hist(k_uq, bins=80, alpha=0.6, density=True, label="surrogate")
plt.hist(k_uq_ref, bins=80, alpha=0.6, density=True, label="code")
plt.xlabel("k_eff"); plt.legend(); plt.show()

# %% [markdown]
# ### ✏️ Exercise 4 — which uncertainty dominates?
#
# Propagate the uncertainty of **one input at a time** (the others fixed at nominal) with the
# surrogate, and print the resulting standard deviation of k_eff in pcm for each input.
# Which tolerance would you tighten first to reduce the uncertainty on k_eff?

# %%
### BEGIN SOLUTION
contrib = {}
for i, name in enumerate(NAMES):
    X1 = np.tile(NOMINAL, (20_000, 1))
    X1[:, i] += SIGMA_IN[i] * rng.standard_normal(20_000)
    contrib[name] = gp.predict(to_unit(X1)).std() * 1e5
for name, s_ in sorted(contrib.items(), key=lambda t: -t[1]):
    print(f"{name:16s}: {s_:5.0f} pcm")
### END SOLUTION
### EXERCISE: # TODO: loop over the 5 inputs, vary only input i, print the std of k_eff in pcm

# %% [markdown] solution
# **Answer.** Compare the individual contributions with the total: they add (approximately)
# in quadrature (here ≈ 230, 170, 120, 100 and 80 pcm → ≈ 340 pcm in total). In this toy model
# the **enrichment** tolerance dominates, followed by the fuel temperature: tightening the
# enrichment specification is the most effective lever. Note a subtlety: with ±2σ GP errors of a few
# hundred pcm over the whole domain, small contributions (a few tens of pcm) are at the limit
# of what this surrogate can resolve — for a real study you would train a **local** surrogate
# around the nominal point, or add runs there.

# %% [markdown]
# ### ✏️ Exercise 5 (bonus) — active learning
#
# Add 10 runs, one at a time, each at the candidate point (among 5000 random ones) where the GP
# standard deviation is **largest**; refit after each run. Recompute the RMS test error.
# This "fill in where you know least" loop is the basis of adaptive design of experiments —
# and the exploration half of Bayesian optimisation.

# %%
### BEGIN SOLUTION
X_al, y_al = X_train.copy(), y_train.copy()
gp_al = gp
for _ in range(10):
    cand = qmc.scale(rng.random((5000, 5)), BOUNDS[:, 0], BOUNDS[:, 1])
    _, s_c = gp_al.predict(to_unit(cand), return_std=True)
    x_new = cand[np.argmax(s_c)]
    X_al = np.vstack([X_al, x_new]); y_al = np.append(y_al, run_code(x_new))
    gp_al = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2,
                                     random_state=SEED).fit(to_unit(X_al), y_al)
rms_al = np.sqrt(np.mean(((gp_al.predict(to_unit(X_test)) - k_true) * 1e5) ** 2))
print(f"RMS error with {len(y_al)} runs (active learning): {rms_al:.0f} pcm  (was {rms:.0f} pcm with 40)")
### END SOLUTION
### EXERCISE: # TODO

# %% [markdown]
# ## Take-away
#
# * A surrogate replaces an expensive code by a fast approximation **learned from a few runs**.
# * With small budgets, **Gaussian processes** beat neural networks and give error bars;
#   always validate on runs the model has not seen.
# * Once trained: criticality searches, reactivity coefficients, uncertainty propagation and
#   optimisation become essentially free.
# * The physics stays in the code: the surrogate only *interpolates* it, inside the training domain.
#
# **Going further:** surrogate- and RL-based fuel assembly optimisation (Radaideh *et al.*,
# Nucl. Eng. Des. **372**, 110966, 2021); review of ML in nuclear physics (Boehnlein *et al.*,
# Rev. Mod. Phys. **94**, 031003, 2022).
