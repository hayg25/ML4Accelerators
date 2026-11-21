# %% [markdown]
# # Exercise 0 — Your first beamline with Cheetah
#
# **ML for Accelerator Physics · beginner course · H. Guler (IJCLab)**
#
# {{COLAB}}
#
# In this notebook you will:
#
# 1. build a small electron transport line (3 quadrupoles) in **Cheetah**,
# 2. track a beam through it and read the beam size at the end,
# 3. tune the quadrupoles **by hand**, then
# 4. let **PyTorch compute gradients through the simulation** and tune them automatically.
#
# > **Cheetah** is a fast, *differentiable* beam dynamics code written in PyTorch
# > (Kaiser *et al.*, PRAB **27**, 054601, 2024 — [github.com/desy-ml/cheetah](https://github.com/desy-ml/cheetah)).
# > Because it is written in PyTorch, every quantity it computes can be differentiated
# > with respect to every input — exactly what machine-learning tools need.
#
# Cells marked **✏️ Exercise** contain a `...` or a `# TODO` for you to fill in.
# Solutions are in `notebooks/solutions/`.

# %% [markdown]
# ## 0 · Setup
#
# On Google Colab, the first cell installs Cheetah (≈ 30 s). Locally, install it once with
# `pip install cheetah-accelerator`.

# %%
import sys
if "google.colab" in sys.modules:
    !pip install -q "cheetah-accelerator>=0.8,<0.9"

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import cheetah

torch.manual_seed(0)
np.random.seed(0)
print("torch", torch.__version__, "· cheetah", cheetah.__version__)

# %% [markdown]
# ## 1 · Build the beamline
#
# Our toy beamline is a short electron transport line ending on a diagnostic screen:
#
# ```
#  beam ─ 0.5 m ─[Q1]─ 0.4 m ─[Q2]─ 0.4 m ─[Q3]─ 1.5 m ─▶ screen
# ```
#
# * A **drift** is an empty section of beam pipe: the beam just flies (and spreads).
# * A **quadrupole** focuses the beam in one plane and *defocuses* it in the other.
#   Its strength is `k1` (unit 1/m²). `k1 > 0` focuses horizontally (x), `k1 < 0` vertically (y).
#
# In Cheetah, a beamline is a `Segment`, i.e. a list of elements.

# %%
def make_beamline():
    """Return the 3-quadrupole transport line used in all exercises."""
    return cheetah.Segment(elements=[
        cheetah.Drift(length=torch.tensor(0.5)),
        cheetah.Quadrupole(length=torch.tensor(0.12), name="Q1"),
        cheetah.Drift(length=torch.tensor(0.4)),
        cheetah.Quadrupole(length=torch.tensor(0.12), name="Q2"),
        cheetah.Drift(length=torch.tensor(0.4)),
        cheetah.Quadrupole(length=torch.tensor(0.12), name="Q3"),
        cheetah.Drift(length=torch.tensor(1.5)),
    ])

segment = make_beamline()
print([el.name for el in segment.elements])

# %% [markdown]
# ## 2 · Define the incoming beam
#
# We describe the beam by its **Twiss parameters** (β, α) and its **emittance** ε.
# The RMS beam size is simply σ = √(β ε). A `ParameterBeam` only stores these
# statistical moments — it is very fast, perfect for optimisation.

# %%
incoming = cheetah.ParameterBeam.from_twiss(
    beta_x=torch.tensor(5.0), alpha_x=torch.tensor(-1.0), emittance_x=torch.tensor(2e-9),
    beta_y=torch.tensor(5.0), alpha_y=torch.tensor(-1.0), emittance_y=torch.tensor(2e-9),
    energy=torch.tensor(100e6),   # 100 MeV electrons
)
print(f"incoming beam: sigma_x = {incoming.sigma_x.item()*1e6:.0f} µm, "
      f"sigma_y = {incoming.sigma_y.item()*1e6:.0f} µm")

# %% [markdown]
# ## 3 · Track the beam
#
# With all quadrupoles off (`k1 = 0`), the beam simply drifts and grows.

# %%
outgoing = segment.track(incoming)
print(f"at the screen: sigma_x = {outgoing.sigma_x.item()*1e6:.0f} µm, "
      f"sigma_y = {outgoing.sigma_y.item()*1e6:.0f} µm")

# %% [markdown]
# Let us wrap this in a function that takes the three quadrupole strengths and returns
# the beam sizes in **micrometres**. We will reuse it everywhere.

# %%
def beam_size(k1, k2, k3):
    """Beam sizes (sigma_x, sigma_y) in µm at the screen for quadrupole strengths in 1/m²."""
    segment.Q1.k1 = torch.tensor(float(k1))
    segment.Q2.k1 = torch.tensor(float(k2))
    segment.Q3.k1 = torch.tensor(float(k3))
    out = segment.track(incoming)
    return out.sigma_x.item() * 1e6, out.sigma_y.item() * 1e6

print(beam_size(0, 0, 0))
print(beam_size(10, 0, 0))

# %% [markdown]
# ### ✏️ Exercise 1 — scan one quadrupole
#
# Scan `k1` of **Q1** from −30 to +30 1/m² (Q2 = Q3 = 0) and plot σx and σy versus k1.
#
# * For which value is σx smallest?
# * What happens to σy at the same time? Why?

# %%
k_scan = np.linspace(-30, 30, 121)
### BEGIN SOLUTION
sizes = np.array([beam_size(k, 0, 0) for k in k_scan])
sx, sy = sizes[:, 0], sizes[:, 1]
### END SOLUTION
### EXERCISE: sx = ...   # TODO: list/array of sigma_x for each k in k_scan
### EXERCISE: sy = ...   # TODO: same for sigma_y

plt.figure(figsize=(6, 3.5))
plt.plot(k_scan, sx, label=r"$\sigma_x$")
plt.plot(k_scan, sy, label=r"$\sigma_y$")
plt.xlabel(r"$k_1$ of Q1  [1/m$^2$]"); plt.ylabel("beam size at screen [µm]")
plt.legend(); plt.grid(alpha=0.3); plt.show()

# %% [markdown] solution
# **Answer.** σx is minimal around k1 ≈ +6 1/m²: Q1 then focuses horizontally so that
# the beam waist falls near the screen. At the same time σy grows: a quadrupole that
# focuses in x *defocuses* in y. This is why we need **several** quadrupoles with
# alternating signs to make the beam small in both planes — and why tuning is a
# genuinely multi-dimensional problem.

# %% [markdown]
# ## 4 · Look *inside* the beamline
#
# To see the beam envelope along the line, we track element by element and split
# long drifts into small pieces.

# %%
def envelope(k1, k2, k3, n_split=20):
    """Return s [m], sigma_x [µm], sigma_y [µm] along the beamline."""
    beam_size(k1, k2, k3)                      # sets the magnet strengths
    beam, s = incoming, 0.0
    S, SX, SY = [0.0], [beam.sigma_x.item()*1e6], [beam.sigma_y.item()*1e6]
    for el in segment.elements:
        L = float(el.length)
        n = n_split if isinstance(el, cheetah.Drift) else 1
        piece = cheetah.Drift(length=torch.tensor(L / n)) if n > 1 else el
        for _ in range(n):
            beam = piece.track(beam)
            s += L / n
            S.append(s); SX.append(beam.sigma_x.item()*1e6); SY.append(beam.sigma_y.item()*1e6)
    return np.array(S), np.array(SX), np.array(SY)

def plot_envelope(k1, k2, k3, title=""):
    s, sx, sy = envelope(k1, k2, k3)
    plt.figure(figsize=(7, 3))
    plt.plot(s, sx, label=r"$\sigma_x$"); plt.plot(s, sy, label=r"$\sigma_y$")
    for q0 in (0.5, 1.02, 1.54):               # quadrupole positions
        plt.axvspan(q0, q0 + 0.12, color="grey", alpha=0.25)
    plt.xlabel("s [m]"); plt.ylabel("beam size [µm]"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.show()

plot_envelope(0, 0, 0, "all quadrupoles off")

# %% [markdown]
# ### ✏️ Exercise 2 — tune by hand
#
# Try to find settings (k1, k2, k3) in [−30, 30] such that **σx + σy < 80 µm** at the screen.
# Hint: alternate the signs (focus x, focus y, focus x…).
#
# On Colab you can also use the sliders in the next cell.

# %%
### BEGIN SOLUTION
my_k = (5.0, -13.5, 12.0)
### END SOLUTION
### EXERCISE: my_k = (0.0, 0.0, 0.0)   # TODO: your best guess
print("sigma_x, sigma_y =", np.round(beam_size(*my_k), 1), "µm  →  sum =",
      round(sum(beam_size(*my_k)), 1), "µm")
plot_envelope(*my_k, f"k = {my_k}")

# %% notest
# Optional: interactive sliders (works in Colab and Jupyter)
from ipywidgets import interact, FloatSlider
sl = dict(min=-30, max=30, step=0.5, continuous_update=False)
interact(lambda k1, k2, k3: plot_envelope(k1, k2, k3, f"σx+σy = {sum(beam_size(k1,k2,k3)):.1f} µm"),
         k1=FloatSlider(value=0, **sl), k2=FloatSlider(value=0, **sl), k3=FloatSlider(value=0, **sl));

# %% [markdown]
# How many trials did you need? On a real machine, every trial costs beam time and an
# operator's attention — this is precisely where machine learning helps.

# %% [markdown]
# ## 5 · Let the gradient do the work
#
# Because Cheetah is written in PyTorch, we can ask for the **derivative of the beam
# size with respect to each quadrupole strength**, and walk "downhill":
#
# $$ k \leftarrow k - \eta \, \frac{\partial (\sigma_x+\sigma_y)}{\partial k} $$
#
# This is **gradient descent**, the very same algorithm used to train neural networks.
# Here we use its popular variant **Adam**.
#
# We declare the three strengths as `torch.nn.Parameter` so that PyTorch tracks gradients.
# (Avoid starting exactly at k = 0: the gradient of a quadrupole is not defined there.)

# %% notest
def gradient_tuning(k_start, lr=0.5, steps=300, k_max=30.0):
    seg = make_beamline()
    seg.Q1.k1 = torch.nn.Parameter(torch.tensor(float(k_start[0])))
    seg.Q2.k1 = torch.nn.Parameter(torch.tensor(float(k_start[1])))
    seg.Q3.k1 = torch.nn.Parameter(torch.tensor(float(k_start[2])))
    params = [seg.Q1.k1, seg.Q2.k1, seg.Q3.k1]
    optimizer = torch.optim.Adam(params, lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        out = seg.track(incoming)
        loss = (out.sigma_x + out.sigma_y) * 1e6        # µm
        loss.backward()                                 # gradients through the simulation!
        optimizer.step()
        with torch.no_grad():                           # respect the power-supply limits
            for p in params:
                p.clamp_(-k_max, k_max)
        history.append(loss.item())
    return np.array([p.item() for p in params]), np.array(history)

k_opt, hist = gradient_tuning([5.0, -5.0, 5.0])
print("optimum found:", np.round(k_opt, 2), "→ σx+σy =", round(hist[-1], 1), "µm")
plt.figure(figsize=(5, 3)); plt.semilogy(hist); plt.grid(alpha=0.3)
plt.xlabel("Adam step"); plt.ylabel("σx + σy  [µm]"); plt.show()
plot_envelope(*k_opt, "after gradient-based tuning")

# %% [markdown]
# ### ✏️ Exercise 3 — local minima
#
# Run `gradient_tuning` from several starting points, for example
# `[5, -5, 5]`, `[-5, 5, -5]`, `[25, -20, 20]`, `[1, 1, 1]`, and compare the final beam sizes.
#
# * Do you always reach the same solution?
# * What does this tell you about gradient-based optimisation?

# %% notest
starts = [[5, -5, 5], [-5, 5, -5], [25, -20, 20], [1, 1, 1]]
### BEGIN SOLUTION
for k0 in starts:
    k_opt, hist = gradient_tuning([float(x) for x in k0])
    print(f"start {k0!s:>14} → k = {np.round(k_opt, 1)}  σx+σy = {hist[-1]:.1f} µm")
### END SOLUTION
### EXERCISE: # TODO: loop over `starts`, call gradient_tuning and print the result

# %% [markdown] solution
# **Answer.** No. Most starting points end in a *local* minimum (σx+σy ≈ 63 µm, k ≈ ±(4.9, −13.5, 11.9)),
# while starts close to the power-supply limit reach a better solution (≈ 32 µm) with
# |k1(Q1)| = 30 1/m² — *at the limit*. Gradient descent is fast and scales to many
# parameters, but it is **local**: it only knows the slope where it stands. It also needs
# a *differentiable model* of the machine — on the real accelerator we usually do not have
# gradients. Exercise 2 (Bayesian optimisation) addresses both issues.

# %% [markdown]
# ## Take-away
#
# * A beamline is a function: **magnet settings → beam properties**.
# * Tuning = finding the inputs that optimise the output. By hand, this is slow.
# * A *differentiable* simulator gives you gradients for free → fast local optimisation.
# * Next: learn this function from data (**surrogate model**, Exercise 1) and optimise it
#   without gradients and with few evaluations (**Bayesian optimisation**, Exercise 2).
