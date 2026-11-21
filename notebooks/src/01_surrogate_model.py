# %% [markdown]
# # Exercise 1 — A neural-network surrogate of the beamline
#
# **ML for Accelerator Physics · beginner course · H. Guler (IJCLab)**
#
# {{COLAB}}
#
# A **surrogate model** is a fast approximation of an expensive simulation (or of the
# machine itself), *learned from examples*. Once trained, it answers in microseconds —
# useful for optimisation, uncertainty studies, online "virtual diagnostics"…
#
# In this notebook you will:
#
# 1. generate a training set with Cheetah (inputs: 3 quadrupole strengths → outputs: σx, σy),
# 2. train a **linear model** (baseline) and a small **neural network** (MLP),
# 3. evaluate them properly on unseen data,
# 4. explore two classic pitfalls: *too little data* and *extrapolation*.
#
# This is **supervised learning**: we learn a mapping from inputs to known outputs.

# %%
import sys
if "google.colab" in sys.modules:
    !pip install -q "cheetah-accelerator>=0.8,<0.9"

# %%
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import cheetah
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# %% [markdown]
# ## 1 · The "expensive" simulator
#
# Same beamline as in Exercise 0. Here it takes ~1 ms per evaluation; real start-to-end
# simulations (space charge, CSR, wakefields…) can take minutes to hours.

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

def simulate(k):
    """k = (k1_Q1, k1_Q2, k1_Q3) in 1/m²  →  (sigma_x, sigma_y) in µm."""
    segment.Q1.k1 = torch.tensor(float(k[0]))
    segment.Q2.k1 = torch.tensor(float(k[1]))
    segment.Q3.k1 = torch.tensor(float(k[2]))
    out = segment.track(incoming)
    return out.sigma_x.item() * 1e6, out.sigma_y.item() * 1e6

print(simulate([5.0, -13.5, 12.0]))

# %% [markdown]
# ## 2 · Generate the dataset
#
# We sample quadrupole settings uniformly in [−30, 30]³ and record the beam sizes.
# Each row of `X` is one machine setting; each row of `Y` the measured beam.

# %%
K_MAX = 30.0
N = 3000

X = rng.uniform(-K_MAX, K_MAX, size=(N, 3))
t0 = time.perf_counter()
Y = np.array([simulate(k) for k in X])
t_sim = (time.perf_counter() - t0) / N
print(f"X: {X.shape}, Y: {Y.shape} — {t_sim*1e3:.2f} ms per simulation")

# %% [markdown]
# ### ✏️ Exercise 1 — look at your data first
#
# Plot the histogram of σx. What do you notice about the range of values?
# Then plot the histogram of **log10(σx)**.

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
### BEGIN SOLUTION
ax[0].hist(Y[:, 0], bins=60)
ax[1].hist(np.log10(Y[:, 0]), bins=60)
### END SOLUTION
### EXERCISE: # TODO: ax[0].hist(...)  and  ax[1].hist(...)
ax[0].set_xlabel("σx [µm]"); ax[1].set_xlabel("log10(σx / µm)")
plt.tight_layout(); plt.show()

# %% [markdown] solution
# **Answer.** σx spans more than two orders of magnitude (≈ 20 µm to several mm) with a
# long tail. A regression model trained on raw values would focus on the huge beams and
# ignore the small ones — which are exactly those we care about. Taking the **logarithm**
# makes the distribution much more regular: *choosing a good representation of the data
# is often more important than choosing the model.*

# %% [markdown]
# We will predict `log10(σ)` and convert back afterwards. We keep 20 % of the data aside as a
# **test set** that the models never see during training.

# %%
Y_log = np.log10(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_log, test_size=0.2, random_state=SEED)
print(len(X_train), "training samples,", len(X_test), "test samples")

# %% [markdown]
# ## 3 · Baseline: linear regression
#
# Always start with the simplest model. If it already works, you are done!

# %%
linear = make_pipeline(StandardScaler(), LinearRegression())
linear.fit(X_train, Y_train)
print(f"linear model — R² on test set: {r2_score(Y_test, linear.predict(X_test)):.3f}")

# %% [markdown]
# ## 4 · A small neural network
#
# A **multi-layer perceptron** (MLP) stacks layers of `weighted sum → non-linearity`.
# Here: 3 inputs → 3 hidden layers of 64 neurons → 2 outputs. Training adjusts 8706 weights
# by gradient descent to minimise the mean squared error.
#
# `StandardScaler` rescales each input to zero mean and unit variance — neural networks
# train much better on normalised inputs.

# %%
def make_mlp():
    """3 inputs → 3 hidden layers of 64 ReLU neurons → 2 outputs."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="relu", max_iter=2000,
                     tol=1e-6, n_iter_no_change=50, random_state=SEED),
    )

mlp = make_mlp()
t0 = time.perf_counter()
mlp.fit(X_train, Y_train)
print(f"training took {time.perf_counter() - t0:.1f} s")

Y_pred = mlp.predict(X_test)
print(f"MLP — R² on test set: {r2_score(Y_test, Y_pred):.4f}")

# %% [markdown]
# ### ✏️ Exercise 2 — parity plot and error in physical units
#
# 1. Make a scatter plot of predicted vs. true σx **in µm** (log-log axes), with the
#    diagonal y = x.
# 2. Compute the **median relative error** |σ_pred − σ_true| / σ_true on the test set.

# %%
### BEGIN SOLUTION
true_um = 10 ** Y_test
pred_um = 10 ** Y_pred
rel_err = np.abs(pred_um - true_um) / true_um

plt.figure(figsize=(4.5, 4.5))
plt.loglog(true_um[:, 0], pred_um[:, 0], ".", ms=3, alpha=0.5)
lims = [true_um.min(), true_um.max()]
plt.plot(lims, lims, "k--", lw=1)
plt.xlabel("true σx [µm] (Cheetah)"); plt.ylabel("predicted σx [µm] (MLP)")
plt.grid(alpha=0.3, which="both"); plt.show()
print(f"median relative error: σx {np.median(rel_err[:, 0])*100:.1f} %, "
      f"σy {np.median(rel_err[:, 1])*100:.1f} %")
### END SOLUTION
### EXERCISE: # TODO: convert Y_test and Y_pred back to µm, make the parity plot, compute the error

# %% [markdown]
# ## 5 · Why bother? Speed.
#
# The surrogate evaluates many settings at once (vectorised).

# %%
X_big = rng.uniform(-K_MAX, K_MAX, size=(100_000, 3))
t0 = time.perf_counter()
_ = mlp.predict(X_big)
t_nn = (time.perf_counter() - t0) / len(X_big)
print(f"simulator: {t_sim*1e6:8.1f} µs / evaluation")
print(f"surrogate: {t_nn*1e6:8.2f} µs / evaluation  → ×{t_sim/t_nn:.0f} faster")

# %% [markdown]
# Here the "simulator" is already fast, so the gain is modest. For a simulation that takes
# 10 minutes, a surrogate that answers in microseconds changes what is possible.

# %% [markdown]
# ### ✏️ Exercise 3 — how much data do we need?
#
# Train a fresh MLP (`make_mlp()`) with 50, 100, 300, 1000 and all training samples and plot the test R²
# versus the training-set size (log x-axis). *This is called a learning curve.*

# %%
sizes = [50, 100, 300, 1000, len(X_train)]
scores = []
### BEGIN SOLUTION
for n in sizes:
    model = make_mlp()
    model.fit(X_train[:n], Y_train[:n])
    scores.append(r2_score(Y_test, model.predict(X_test)))
    print(f"n = {n:5d}  R² = {scores[-1]:.4f}")
### END SOLUTION
### EXERCISE: # TODO: for each n, train a new model on X_train[:n], Y_train[:n] and append its test R²

plt.figure(figsize=(5, 3))
plt.semilogx(sizes, scores, "o-"); plt.grid(alpha=0.3)
plt.xlabel("number of training samples"); plt.ylabel("test R²"); plt.show()

# %% [markdown] solution
# **Answer.** With a few tens of samples the network is poor; quality improves steadily
# with data and saturates after a few thousand points for this 3-D problem. The number of
# samples needed grows quickly with the number of inputs (the *curse of dimensionality*):
# on a real machine, data acquisition is often the bottleneck.

# %% [markdown]
# ### ✏️ Exercise 4 — the extrapolation trap
#
# Train a new MLP (`make_mlp()`) **only on settings with |k| ≤ 20** for all three quadrupoles, then test it
# separately on
#
# * test points inside this region,
# * test points where at least one |k| > 20.
#
# Compare the two R² scores. What do you conclude?

# %%
inside_train = np.all(np.abs(X_train) <= 20, axis=1)
inside_test = np.all(np.abs(X_test) <= 20, axis=1)
### BEGIN SOLUTION
model_in = make_mlp()
model_in.fit(X_train[inside_train], Y_train[inside_train])
r2_in = r2_score(Y_test[inside_test], model_in.predict(X_test[inside_test]))
r2_out = r2_score(Y_test[~inside_test], model_in.predict(X_test[~inside_test]))
for name, mask in [("inside ", inside_test), ("outside", ~inside_test)]:
    err = np.abs(10 ** model_in.predict(X_test[mask]) - 10 ** Y_test[mask]) / 10 ** Y_test[mask]
    print(f"median relative error {name}: {np.median(err)*100:.1f} %")
### END SOLUTION
### EXERCISE: r2_in, r2_out = ..., ...   # TODO
print(f"trained on {inside_train.sum()} points with |k| ≤ 20")
print(f"R² inside the training domain : {r2_in:.3f}")
print(f"R² outside (extrapolation)    : {r2_out:.3f}")

# %% [markdown] solution
# **Answer.** Inside the training domain the model is excellent; outside, the error is
# several times larger — and it grows the further you go from the training data. A neural network is an *interpolator*: it knows nothing about regions it has
# never seen, and it will still return a confident-looking number. In operation, always
# check that inputs lie within the training domain — or use models that also return an
# **uncertainty** (Gaussian processes, ensembles…, see Exercise 2).

# %% [markdown]
# ### ✏️ Exercise 5 (bonus) — optimise with the surrogate
#
# Use the surrogate to find the setting with the smallest σx + σy among 100 000 random
# candidates (`X_big`). Then **check** this setting with the real simulator. Do they agree?

# %%
### BEGIN SOLUTION
pred = 10 ** mlp.predict(X_big)
best = X_big[np.argmin(pred.sum(axis=1))]
print("best candidate:", np.round(best, 2))
print(f"surrogate says : σx+σy = {pred.sum(axis=1).min():.1f} µm")
print(f"simulator says : σx+σy = {sum(simulate(best)):.1f} µm")
### END SOLUTION
### EXERCISE: # TODO

# %% [markdown] solution
# The surrogate points to a good region of parameter space, but its prediction near the
# optimum is less accurate (few training points have very small beams). In practice a
# surrogate is used to *propose* candidates, which are then verified — and the new data
# point is added to the training set. Doing this in a principled way is exactly
# **Bayesian optimisation**.

# %% [markdown]
# ## Take-away
#
# * A surrogate is a **learned function** replacing an expensive one: same inputs, same outputs.
# * Look at the data, choose a good representation (here log σ), **always keep a test set**.
# * Start with a simple baseline; the neural network must beat it.
# * Neural networks interpolate, they do **not** extrapolate — know your training domain.
#
# **In the literature:** virtual diagnostic of the longitudinal phase space at LCLS
# (Emma *et al.*, PRAB **21**, 112802, 2018); surrogate-based multi-objective optimisation
# (Edelen *et al.*, PRAB **23**, 044601, 2020).
