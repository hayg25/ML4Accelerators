# %% [markdown]
# # Exercise 6 — Neutron/gamma pulse-shape discrimination
#
# **ML for Accelerator Physics · nuclear-energy track · H. Guler (IJCLab)**
#
# {{COLAB}}
#
# **Why this matters.** Fast neutrons are measured with organic scintillators (liquid or plastic)
# in neutron dosimetry and radiation protection around reactors and accelerators, in reactor
# instrumentation, in safeguards (counting fission neutrons from nuclear material) and in fusion
# and fission experiments. These detectors also see **gammas** — usually many more of them. The
# two are separated by the *shape* of the light pulse: proton recoils (from neutrons) produce
# relatively more **slow, delayed light** than electrons (from gammas). This is called
# **pulse-shape discrimination (PSD)**.
#
# The standard method compares the charge in the tail of the pulse with the total charge. It
# works well at high energy and degrades at low energy, where only a few tens of photoelectrons
# are collected. Machine-learning PSD is an active topic (e.g. Karmakar *et al.*, Pramana **97**,
# 157 (2023); a benchmark of ~60 PSD methods: arXiv:2508.02750 (2025)).
#
# In this notebook you will:
#
# 1. simulate digitised pulses photoelectron by photoelectron,
# 2. implement and **optimise** the classical charge-comparison method,
# 3. train a neural network on the raw waveforms and compare **honestly**,
# 4. add a realistic complication — **pile-up** — and see which method copes.
#
# Only numpy, scikit-learn and matplotlib: nothing to install.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve

SEED = 0
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 1 · Simulating the pulses
#
# A liquid scintillator (EJ-301 / BC-501A-like) read by a photomultiplier and a 500 MS/s digitiser
# (one sample every 2 ns, 300 ns window):
#
# * deposited energy E in **keVee** (electron-equivalent), ~1 photoelectron (p.e.) per keVee;
# * each p.e. arrives at a random time drawn from the scintillation decay: a fast component
#   (3.2 ns) and two slow ones (32 ns and 270 ns). **Neutrons have a larger slow fraction**
#   (30 % here) than **gammas** (12 %);
# * each p.e. produces a single-photoelectron pulse with gain fluctuations; electronic noise is added.
#
# At low energy, a pulse has only a few tens of p.e.: the tail/total ratio fluctuates a lot —
# this is the physical limit of PSD.
#
# Energies are drawn log-uniformly between 30 and 1500 keVee for both particles, so that the
# classifier must use the **shape**, not the amplitude.

# %%
DT, N_S = 2.0, 150                                  # ns per sample, samples per waveform
T = np.arange(N_S) * DT
TAU = np.array([3.2, 32.0, 270.0])                  # decay constants [ns]
FRACTIONS = {0: np.array([0.88, 0.07, 0.05]),       # 0 = gamma
             1: np.array([0.70, 0.17, 0.13])}       # 1 = neutron
PE_PER_KEV, NOISE, T_TRIGGER = 1.0, 0.3, 20.0
_tb = np.arange(-4, 5) * DT
_SPE = np.exp(-0.5 * (_tb / 1.5) ** 2); _SPE /= _SPE.sum()   # single-p.e. response (σ = 1.5 ns)

def pulse(kind, energy, rng):
    n_pe = rng.poisson(PE_PER_KEV * energy)
    comp = rng.choice(3, n_pe, p=FRACTIONS[kind])
    t = T_TRIGGER + rng.uniform(0, DT) + rng.exponential(TAU[comp]) + rng.normal(0, 0.5, n_pe)
    gain = rng.gamma(1 / 0.16, 0.16, n_pe)          # p.e. gain fluctuations (≈ 40 %)
    h = np.bincount(np.clip((t / DT).astype(int), 0, N_S - 1), weights=gain, minlength=N_S)
    h[-1] = 0.0                                     # light arriving after the window is lost
    return np.convolve(h, _SPE, mode="same") + rng.normal(0, NOISE, N_S)

def make_pulses(n, rng, e_range=(30, 1500)):
    """n waveforms (n, 150), labels (0 = gamma, 1 = neutron), true energies [keVee]."""
    y = rng.integers(0, 2, n)
    energy = np.exp(rng.uniform(np.log(e_range[0]), np.log(e_range[1]), n))
    X = np.array([pulse(k, e, rng) for k, e in zip(y, energy)])
    return X, y, energy

X_train, y_train, E_train = make_pulses(60000, rng)
X_test, y_test, E_test = make_pulses(30000, np.random.default_rng(1))
print(X_train.shape, "— neutron fraction:", y_train.mean().round(2))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for k, c in [(0, "C0"), (1, "C3")]:
    i = np.where((y_test == k) & (E_test > 800))[0][0]
    j = np.where((y_test == k) & (E_test < 60))[0][0]
    ax[0].plot(T, X_test[i], c, label=f"{['gamma', 'neutron'][k]}, {E_test[i]:.0f} keVee")
    ax[1].plot(T, X_test[j], c, label=f"{['gamma', 'neutron'][k]}, {E_test[j]:.0f} keVee")
for a in ax:
    a.set_xlabel("time [ns]"); a.set_ylabel("signal [p.e./sample]"); a.legend(); a.set_xlim(0, 200)
ax[0].set_title("high energy"); ax[1].set_title("low energy: a handful of photoelectrons")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✏️ Exercise 1 — where is the difference?
#
# Plot the **average normalised pulse** (each pulse divided by its sum) for gammas and for
# neutrons with E > 300 keVee, on a logarithmic y-axis. Where do they differ?

# %%
### BEGIN SOLUTION
sel = E_train > 300
plt.figure(figsize=(6, 3.5))
for k, name, c in [(0, "gamma", "C0"), (1, "neutron", "C3")]:
    P = X_train[sel & (y_train == k)]
    plt.semilogy(T, (P / P.sum(1, keepdims=True)).mean(0), c, label=name)
plt.ylim(1e-4, 0.5); plt.xlabel("time [ns]"); plt.ylabel("mean normalised pulse")
plt.legend(); plt.grid(alpha=0.3); plt.show()
### END SOLUTION
### EXERCISE: # TODO

# %% [markdown] solution
# **Answer.** The rising edge and the peak are almost identical; neutron pulses have clearly
# more light in the **tail**, from ~10 ns after the peak onwards. Any discrimination method
# must measure "how much tail" — as precisely as possible with few photoelectrons.

# %% [markdown]
# ## 2 · The classical method: charge comparison
#
# PSD parameter = tail charge / total charge, both integrated from the pulse maximum (the tail
# starts `gate` ns after the maximum). The total charge also gives the energy.

# %%
def charge_comparison(X, gate=24.0):
    """Return (tail/total, total charge) for each waveform."""
    peak = np.argmax(X, axis=1)
    psd, total = np.zeros(len(X)), np.zeros(len(X))
    for i, (x, p) in enumerate(zip(X, peak)):
        tot = x[max(p - 2, 0): p + 100].sum()
        tail = x[p + int(gate / DT): p + 100].sum()
        psd[i], total[i] = tail / tot, tot
    return psd, total

psd, Q = charge_comparison(X_test)
plt.figure(figsize=(7, 4))
plt.hist2d(Q / PE_PER_KEV, psd, bins=[np.geomspace(25, 1600, 120), np.linspace(-0.1, 0.6, 120)],
           cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
plt.xscale("log"); plt.xlabel("total charge ≈ energy [keVee]"); plt.ylabel("tail / total")
plt.title("the classic PSD plot (no labels used)"); plt.colorbar(label="pulses"); plt.show()

# %% [markdown]
# Two bands — gammas below, neutrons above — that **merge at low energy**. The quality of the
# separation in an energy slice is quantified by the **figure of merit**
#
# $$ \mathrm{FOM} = \frac{|\mu_n - \mu_\gamma|}{\mathrm{FWHM}_n + \mathrm{FWHM}_\gamma} $$
#
# (FOM ≳ 1.27 is usually considered good separation).

# %%
def fom(psd, y):
    mu_n, mu_g = psd[y == 1].mean(), psd[y == 0].mean()
    return abs(mu_n - mu_g) / (2.355 * (psd[y == 1].std() + psd[y == 0].std()))

# %% [markdown]
# ### ✏️ Exercise 2 — optimise the gate
#
# For pulses with 100–200 keVee in the **training** set, compute the FOM for tail-gate starts of
# 6, 10, 14, 18, 24, 30 and 40 ns. Which gate is best? Keep it as `BEST_GATE`.

# %%
gates = [6, 10, 14, 18, 24, 30, 40]
sl = (E_train > 100) & (E_train < 200)
### BEGIN SOLUTION
foms = [fom(charge_comparison(X_train[sl], g)[0], y_train[sl]) for g in gates]
BEST_GATE = gates[int(np.argmax(foms))]
### END SOLUTION
### EXERCISE: foms = ...        # TODO: one FOM per gate
### EXERCISE: BEST_GATE = ...   # TODO
for g, f in zip(gates, foms):
    print(f"gate {g:3d} ns : FOM = {f:.2f}")
print("best gate:", BEST_GATE, "ns")

# %% [markdown] solution
# **Answer.** A short delay (≈ 10–14 ns) is best here: it starts integrating the slow light as
# early as possible, which maximises the number of "tail" photoelectrons — what limits PSD at
# low energy is photoelectron statistics. Always optimise the classical method before comparing
# it with ML: an un-tuned baseline makes any new method look good.

# %% [markdown]
# ## 3 · A neural network on the raw waveform
#
# Input: 90 samples starting just before the maximum, normalised by their sum (shape only),
# plus the log of the total charge (so that the network knows how noisy the shape is).

# %%
def features(X):
    peak = np.argmax(X, axis=1)
    F = np.zeros((len(X), 90))
    for i, (x, p) in enumerate(zip(X, peak)):
        seg = x[max(p - 3, 0): max(p - 3, 0) + 90]
        F[i, :len(seg)] = seg
    s = F.sum(1, keepdims=True)
    return np.hstack([F / s, np.log(np.abs(s) + 1)])

def make_model():
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                       early_stopping=True, random_state=SEED))

E_BINS = [30, 50, 80, 130, 200, 350, 600, 1500]
def auc_per_energy(score, y, E):
    return [roc_auc_score(y[(E >= a) & (E < b)], score[(E >= a) & (E < b)])
            for a, b in zip(E_BINS[:-1], E_BINS[1:])]

# %% [markdown]
# ### ✏️ Exercise 3 — network vs. optimised charge comparison
#
# 1. Train `make_model()` on `features(X_train)`, `y_train`.
# 2. Compute the neutron probability on the test set.
# 3. Compare the **ROC AUC** per energy bin (`auc_per_energy`) with the charge comparison at
#    `BEST_GATE`. (AUC = 1: perfect separation; 0.5: coin flip.)

# %%
### BEGIN SOLUTION
model = make_model().fit(features(X_train), y_train)
p_nn = model.predict_proba(features(X_test))[:, 1]
psd_best, _ = charge_comparison(X_test, BEST_GATE)
auc_nn = auc_per_energy(p_nn, y_test, E_test)
auc_cc = auc_per_energy(psd_best, y_test, E_test)
### END SOLUTION
### EXERCISE: auc_nn = ...   # TODO
### EXERCISE: auc_cc = ...   # TODO
print(f"{'energy [keVee]':>15s} {'charge comp.':>13s} {'network':>9s}")
for a, b, c, n in zip(E_BINS[:-1], E_BINS[1:], auc_cc, auc_nn):
    print(f"{a:6d} – {b:5d}   {c:12.3f} {n:9.3f}")

# %% [markdown] solution
# **Answer.** On these clean pulses the network is **about as good** as the optimised charge
# comparison — slightly better at the lowest energies, equal above. This is an important,
# honest result: for a signal described by a simple physical model (two light components +
# photoelectron statistics), the tail/total ratio is already close to the best possible
# discriminant. ML does not create information that is not in the data.

# %% [markdown]
# ## 4 · Real life: pile-up
#
# At high count rates — near a reactor, a strong source, or an accelerator target — a second
# (usually gamma) pulse often arrives in the same 300 ns window. Pile-up adds light in the
# tail, so a gamma can look like a neutron and a neutron's shape is distorted.

# %%
def add_pileup(X, rng, prob=0.15):
    """Add a random gamma pulse, delayed by 30–240 ns, to a fraction `prob` of the waveforms."""
    X = X.copy()
    for i in np.where(rng.random(len(X)) < prob)[0]:
        delay = rng.integers(15, 120)               # in samples
        extra = pulse(0, np.exp(rng.uniform(np.log(30), np.log(1500))), rng)
        extra = np.roll(extra, delay)
        extra[:delay] = 0.0
        X[i] += extra
    return X

X_test_pu = add_pileup(X_test, np.random.default_rng(7))
i = np.where(np.abs(X_test_pu - X_test).sum(1) > 50)[0][0]
plt.figure(figsize=(6, 3)); plt.plot(T, X_test_pu[i], label="with pile-up"); plt.plot(T, X_test[i], label="original")
plt.xlabel("time [ns]"); plt.legend(); plt.show()

# %% [markdown]
# ### ✏️ Exercise 4 — who survives pile-up?
#
# On `X_test_pu`, compute the AUC per energy bin for
#
# 1. the charge comparison at `BEST_GATE`,
# 2. the network trained above (on clean pulses),
# 3. a new network trained on `add_pileup(X_train, np.random.default_rng(8))`.

# %%
### BEGIN SOLUTION
psd_pu, _ = charge_comparison(X_test_pu, BEST_GATE)
auc_cc_pu = auc_per_energy(psd_pu, y_test, E_test)
auc_nn_pu = auc_per_energy(model.predict_proba(features(X_test_pu))[:, 1], y_test, E_test)
model_pu = make_model().fit(features(add_pileup(X_train, np.random.default_rng(8))), y_train)
auc_nnpu_pu = auc_per_energy(model_pu.predict_proba(features(X_test_pu))[:, 1], y_test, E_test)
### END SOLUTION
### EXERCISE: auc_cc_pu = ...     # TODO
### EXERCISE: auc_nn_pu = ...     # TODO
### EXERCISE: auc_nnpu_pu = ...   # TODO
centres = np.sqrt(np.array(E_BINS[:-1]) * np.array(E_BINS[1:]))
plt.figure(figsize=(6, 4))
plt.semilogx(centres, auc_cc, "k--", label="charge comparison, clean pulses")
plt.semilogx(centres, auc_cc_pu, "o-", label="charge comparison, pile-up")
plt.semilogx(centres, auc_nn_pu, "o-", label="network (trained clean), pile-up")
plt.semilogx(centres, auc_nnpu_pu, "o-", label="network (trained with pile-up), pile-up")
plt.xlabel("energy [keVee]"); plt.ylabel("ROC AUC"); plt.ylim(0.7, 1.005)
plt.grid(alpha=0.3, which="both"); plt.legend(fontsize=8); plt.show()

# %% [markdown] solution
# **Answer.** With 15 % pile-up both the charge comparison and the network trained on clean pulses
# degrade, even at high energy where separation was perfect. The network **trained with pile-up
# examples** learns to recognise a second pulse in the tail and recovers most of the
# performance. The classical answer would be a dedicated pile-up rejection algorithm (at the
# cost of losing events); ML can learn to handle the distortion directly — provided the
# training data contain it.

# %% [markdown]
# ### ✏️ Exercise 5 (bonus) — what matters in practice: gamma rejection
#
# In a strong gamma field, misidentifying 1 gamma in 100 can swamp the neutron signal. Compute,
# for each energy bin on the **clean** test set, the **neutron efficiency** when the threshold is
# set so that only **0.1 %** of gammas are misidentified (use `roc_curve` and interpolate the
# true-positive rate at a false-positive rate of 10⁻³). Compare charge comparison and network.
# Down to which energy can you count neutrons with > 50 % efficiency?

# %%
def efficiency_at(score, y, fpr_target=1e-3):
    fpr, tpr, _ = roc_curve(y, score)
    return np.interp(fpr_target, fpr, tpr)

### BEGIN SOLUTION
for a, b in zip(E_BINS[:-1], E_BINS[1:]):
    m = (E_test >= a) & (E_test < b)
    print(f"{a:5d}–{b:5d} keVee: neutron efficiency  charge comp. {efficiency_at(psd_best[m], y_test[m]):.2f}"
          f"  |  network {efficiency_at(p_nn[m], y_test[m]):.2f}")
### END SOLUTION
### EXERCISE: # TODO

# %% [markdown] solution
# **Answer.** At 10⁻³ gamma misidentification both methods keep > 50 % neutron efficiency only
# above ≈ 130 keVee, and they are comparable (the charge comparison is even slightly better in
# some bins). These numbers are noisy: with ~2000 gammas per bin, only a couple of events
# define the threshold — estimating very small error rates needs large calibration samples. The AUC hides this:
# what matters for an application is the performance **at the operating point** (here: very
# strong gamma rejection). Lowering this threshold needs more photoelectrons (better light
# collection, better scintillator) — physics, not software.

# %% [markdown]
# ## Take-away
#
# * PSD is a classification problem; the classical tail/total method is physics-based and,
#   once **optimised**, close to the best possible on clean signals.
# * ML helps when signals carry **distortions that are hard to model by hand** (pile-up, baseline
#   shifts, afterpulses, detector ageing) — if the training data contain them.
# * Evaluate at the **operating point** that matters (gamma rejection), not only with global scores.
# * Labels in real life come from separate measurements (gamma-only source such as Cs-137,
#   time-of-flight with Cf-252) — their quality limits what the network can learn.
#
# **Further reading:** Karmakar *et al.*, Pramana **97**, 157 (2023); CNN-based PSD with SiPMs,
# arXiv:1807.06853; PSD methods benchmark and open toolbox, arXiv:2508.02750 (2025).
