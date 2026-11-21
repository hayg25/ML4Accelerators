# %% [markdown]
# # Exercise 4 — Identifying radionuclides in gamma-ray spectra
#
# **ML for Accelerator Physics · nuclear-energy track · H. Guler (IJCLab)**
#
# {{COLAB}}
#
# **Why this matters.** Radionuclide identification from gamma spectra is everyday work in
# radiation protection, nuclear-plant operation, **decommissioning** (characterising activated
# concrete and steel), waste management, environmental monitoring after an incident, and nuclear
# security (portal monitors, hand-held identifiers). Field instruments often use cheap,
# low-resolution **NaI(Tl)** scintillators: peaks overlap, statistics are poor, the gain drifts
# with temperature — and the operator needs an answer *now*.
#
# Neural networks trained on **simulated spectra** are an active research topic for exactly this
# task (e.g. Kamuda *et al.*, IEEE TNS **64**, 1858 (2017); Kamuda, Zhao & Huff, NIM A **954**,
# 161385 (2020)).
#
# In this notebook you will:
#
# 1. simulate realistic NaI spectra (photopeaks, Compton continuum, natural background, Poisson noise),
# 2. implement the classical **region-of-interest (ROI)** method,
# 3. train a neural network for **multi-label** identification (several nuclides at once),
# 4. compare it fairly with a physics-based **full-spectrum fit**,
# 5. study low counting statistics and **gain drift** — and fix the latter with data augmentation.
#
# Only numpy, scikit-learn and matplotlib: nothing to install.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from scipy.optimize import nnls

SEED = 0
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 1 · A small spectrum simulator
#
# Detector: 3″×3″ NaI(Tl), 512 channels from 0 to 3 MeV (≈ 5.9 keV/channel), resolution
# 7 % FWHM at 662 keV (FWHM ∝ √E).
#
# For each gamma line of energy E₀ the detector response is the sum of
#
# * a **photopeak** (Gaussian) — fraction *photofraction(E₀)*, decreasing with energy,
# * a **Compton continuum** up to the Compton edge, shaped by the Klein–Nishina cross-section,
#   plus a small backscatter peak and multiple-scattering contribution,
# * weighted by the **intrinsic efficiency** of the crystal.
#
# The library contains 9 nuclides relevant for reactors, decommissioning and calibration:
# activation products (Co-60, Co-58, Mn-54, Eu-152), fission products (Cs-137, Cs-134, I-131),
# an actinide (Am-241) and a calibration source (Ba-133). Natural background (K-40, U and Th
# chains) is always present. Line energies and intensities are taken from nuclear data tables
# (main lines only).
#
# *You do not need to understand every line of the next cell — run it and read the plots.*

# %%
N_CH, E_MAX = 512, 3000.0
E = (np.arange(N_CH) + 0.5) * E_MAX / N_CH          # channel centres [keV]
MEC2 = 511.0

LIBRARY = {   # (energy keV, emission probability per decay)
    "Am-241": [(26.3, 0.024), (59.54, 0.359)],
    "Ba-133": [(30.9, 0.62), (80.99, 0.329), (276.40, 0.0716), (302.85, 0.1834), (356.01, 0.6205), (383.85, 0.0894)],
    "Co-58":  [(511.0, 0.30), (810.76, 0.9945)],
    "Co-60":  [(1173.23, 0.9985), (1332.49, 0.9998)],
    "Cs-134": [(563.25, 0.0834), (569.33, 0.1537), (604.72, 0.9762), (795.86, 0.8546), (801.95, 0.0869), (1365.19, 0.0302)],
    "Cs-137": [(32.2, 0.07), (661.66, 0.851)],
    "Eu-152": [(40.1, 0.59), (121.78, 0.2853), (244.70, 0.0755), (344.28, 0.2659), (778.90, 0.1293),
               (964.08, 0.1451), (1085.84, 0.1013), (1112.08, 0.1367), (1408.01, 0.2087)],
    "I-131":  [(80.19, 0.026), (284.31, 0.0612), (364.49, 0.815), (636.99, 0.0716), (722.91, 0.0177)],
    "Mn-54":  [(834.85, 0.9998)],
}
BACKGROUND = [(295.2, 0.18), (351.9, 0.36), (583.2, 0.30), (609.3, 0.45), (911.2, 0.26),
              (1120.3, 0.15), (1460.8, 1.0), (1764.5, 0.15), (2614.5, 0.36)]
NUCLIDES = list(LIBRARY)

def fwhm(e):
    return 0.07 * np.sqrt(662.0 * e)

def efficiency(e):
    Ek = np.array([30, 50, 100, 200, 300, 500, 662, 1000, 1500, 2000, 3000.0])
    mass_att = np.array([18.0, 20.0, 1.6, 0.34, 0.17, 0.096, 0.077, 0.058, 0.047, 0.042, 0.037])  # NaI cm²/g
    mu = 3.67 * np.exp(np.interp(np.log(e), np.log(Ek), np.log(mass_att)))
    return (1 - np.exp(-mu * 7.62)) * np.exp(-(25.0 / e) ** 3)   # crystal × entrance window

def photofraction(e):
    return 1 / (1 + (e / 650.0) ** 1.5)

_S = fwhm(np.maximum(E, 10)) / 2.355
_SMEAR = np.exp(-0.5 * ((E[:, None] - E[None, :]) / _S[None, :]) ** 2)
_SMEAR /= _SMEAR.sum(0, keepdims=True)                 # resolution matrix

def gaussian(e0):
    g = np.exp(-0.5 * ((E - e0) / (fwhm(e0) / 2.355)) ** 2)
    return g / g.sum()

def compton(e0):
    eps = e0 / MEC2
    tmax = e0 * 2 * eps / (1 + 2 * eps)                # Compton edge
    s = np.minimum(E, 0.9999 * tmax) / e0
    kn = 2 + s ** 2 / (eps ** 2 * (1 - s) ** 2) + s / (1 - s) * (s - 2 / eps)   # Klein–Nishina dσ/dT
    kn = np.where(E < tmax, kn, 0.0)
    if kn.sum() <= 0:
        c = np.zeros(N_CH); c[0] = 1.0
        return c
    multi = np.where(E < e0, 1.0, 0.0)                 # multiple scattering fills the valley
    back = gaussian(max(e0 - tmax, 30.0))              # backscatter peak
    return _SMEAR @ (0.80 * kn / kn.sum() + 0.13 * multi / multi.sum() + 0.07 * back)

def line_response(e0):
    pf = photofraction(e0)
    return efficiency(e0) * (pf * gaussian(e0) + (1 - pf) * compton(e0))

def template(lines):
    t = sum(I * line_response(e0) for e0, I in lines)
    return t / t.sum()

T_ISO = np.array([template(LIBRARY[n]) for n in NUCLIDES])           # (9, 512)
_cont = np.exp(-E / 250.0) + 0.15 * np.exp(-E / 900.0)
T_BKG = 0.4 * template(BACKGROUND) + 0.6 * _cont / _cont.sum()

def apply_gain(t, gain):
    """A gain change moves a line from energy E to gain×E on the energy axis."""
    out = np.interp(E / gain, E, t, left=0.0, right=0.0)
    return out / out.sum()

def make_spectra(n, rng, counts=(500, 20000), gain=(1.0, 1.0), library=None):
    """n random spectra with 0–3 nuclides on top of natural background.
    Returns counts X (n, 512), labels Y (n, 9), total counts N."""
    lib = T_ISO if library is None else library
    X, Y, Ns = np.zeros((n, N_CH)), np.zeros((n, len(lib)), dtype=int), np.zeros(n)
    for j in range(n):
        k = rng.choice([0, 1, 2, 3], p=[0.10, 0.45, 0.30, 0.15])
        idx = rng.choice(len(lib), k, replace=False)
        Y[j, idx] = 1
        ratio = np.exp(rng.uniform(np.log(0.3), np.log(5.0)))          # source / background
        lam = T_BKG.copy()
        if k:
            lam = (T_BKG + ratio * rng.dirichlet(np.ones(k)) @ lib[idx]) / (1 + ratio)
        g = rng.uniform(*gain)
        if g != 1.0:
            lam = apply_gain(lam, g)
        Ns[j] = np.exp(rng.uniform(np.log(counts[0]), np.log(counts[1])))
        X[j] = rng.poisson(Ns[j] * lam)
    return X, Y, Ns

# %% [markdown]
# ### Look at the physics

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for i, n in enumerate(NUCLIDES):
    ax[0].semilogy(E, T_ISO[i], lw=1.2, label=n)
ax[0].semilogy(E, T_BKG, "k", lw=2, label="background")
ax[0].set_ylim(1e-5, 0.1); ax[0].set_xlim(0, 2000); ax[0].legend(ncol=5, fontsize=8)
ax[0].set_title("noise-free detector response (normalised)"); ax[0].set_xlabel("energy [keV]")

Xs, Ys, Ns = make_spectra(3, np.random.default_rng(3), counts=(3000, 3000))
for j in range(3):
    lab = ", ".join(NUCLIDES[k] for k in np.where(Ys[j])[0]) or "background only"
    ax[1].step(E, Xs[j], where="mid", lw=0.9, label=lab)
ax[1].set_xlim(0, 2000); ax[1].set_xlabel("energy [keV]"); ax[1].set_ylabel("counts")
ax[1].set_title("what the instrument sees: 3000 counts, Poisson noise"); ax[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✏️ Exercise 1 — where are the traps?
#
# With a 7 % resolution (FWHM ≈ 55 keV at 800 keV), which **pairs of lines** will be hard to
# separate? Look at the library and at the plot, and list at least three problematic cases.

# %% [markdown] solution
# **Answer.**
#
# * **Co-58 (811 keV) vs Mn-54 (835 keV)**: 24 keV apart, less than half a FWHM.
# * **Ba-133 (356 keV) vs I-131 (364 keV)**: the main lines coincide; only the weaker lines
#   (Ba-133: 81, 303 keV; I-131: 284, 637 keV) can tell them apart.
# * **Cs-134 (605 keV) vs natural Bi-214 (609 keV)** in the background, and **Cs-137 (662 keV)
#   vs I-131 (637 keV) / Cs-134 (605 keV)**.
# * **Eu-152 (344 keV)** sits next to Ba-133/I-131 lines.
#
# A method looking at one line at a time will confuse these; one has to use the *whole* pattern
# of lines and continuum.

# %% [markdown]
# ## 2 · Datasets
#
# 20 000 training spectra and 3 000 test spectra, with 500 to 20 000 counts each.

# %%
X_train, Y_train, N_train = make_spectra(20000, rng)
X_test, Y_test, N_test = make_spectra(3000, np.random.default_rng(1))
print(X_train.shape, Y_train.shape, "— nuclides per spectrum:", np.bincount(Y_train.sum(1)))

# %% [markdown]
# ## 3 · The classical method: regions of interest
#
# For each nuclide, take its strongest line E₀, sum the counts in a window of ±1 FWHM
# (**gross**), estimate the background under the peak from two side windows (**B**), and
# declare the nuclide present if the net counts are significant:
#
# $$ \frac{\text{gross} - B}{\sqrt{\text{gross} + B}} > k $$
#
# (k = 3: the usual "3σ" decision threshold).

# %%
MAIN_LINE = {n: max([l for l in LIBRARY[n] if l[0] > 50],          # gamma lines only (no X-rays)
                    key=lambda l: l[1] * efficiency(l[0]) * photofraction(l[0]))[0]
             for n in NUCLIDES}
print(MAIN_LINE)

def roi_significance(spectrum, e0):
    w = fwhm(e0)
    peak = (E > e0 - w) & (E < e0 + w)
    left = (E > e0 - 1.75 * w) & (E <= e0 - w)
    right = (E >= e0 + w) & (E < e0 + 1.75 * w)
    gross = spectrum[peak].sum()
    bkg_density = (spectrum[left].sum() + spectrum[right].sum()) / max(left.sum() + right.sum(), 1)
    B = bkg_density * peak.sum()
    return (gross - B) / np.sqrt(gross + B + 1e-9)

# %% [markdown]
# ### ✏️ Exercise 2 — evaluate the ROI method
#
# Compute the ROI decisions (k = 3) for all test spectra and nuclides, then the **F1 score** per
# nuclide (`f1_score(Y_test, Y_roi, average=None)`; 1 = perfect).

# %%
### BEGIN SOLUTION
S_roi = np.array([[roi_significance(x, MAIN_LINE[n]) for n in NUCLIDES] for x in X_test])
Y_roi = (S_roi > 3).astype(int)
f1_roi = f1_score(Y_test, Y_roi, average=None, zero_division=0)
### END SOLUTION
### EXERCISE: Y_roi = ...    # TODO: array (3000, 9) of 0/1 decisions
### EXERCISE: f1_roi = ...   # TODO
for n, f in zip(NUCLIDES, f1_roi):
    print(f"{n:7s} F1 = {f:.2f}")
print(f"all nuclides correct in {np.mean((Y_roi == Y_test).all(1))*100:.0f} % of spectra")

# %% [markdown] solution
# **Answer.** Isolated lines (Am-241, Co-60) work reasonably well. Overlapping cases fail:
# Co-58/Mn-54 and Ba-133/I-131 trigger each other (false positives), Cs-134 fires on the
# natural Bi-214 line. A single threshold k is also a compromise between missed sources at low
# counts and false alarms at high counts. Expert analysts fix this with peak fitting,
# multiplet deconvolution and nuclide libraries with all lines — which is complex to automate.

# %% [markdown]
# ## 4 · A neural network
#
# Input: the 512 channel contents, normalised to the total number of counts and square-rooted
# (this "stabilises" the Poisson noise). Output: 9 independent yes/no answers — a **multi-label**
# classification (several nuclides may be present at once). `MLPClassifier` handles this
# directly when `Y` has one column per nuclide.

# %%
def features(X):
    return np.sqrt(X / X.sum(axis=1, keepdims=True))

def make_model():
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60,
                                       early_stopping=True, random_state=SEED))

# %% [markdown]
# ### ✏️ Exercise 3 — train and compare
#
# Train `make_model()` on `features(X_train)`, predict the test set, and print the per-nuclide F1
# next to the ROI result. (Training takes ~1 min on Colab.)

# %%
### BEGIN SOLUTION
model = make_model().fit(features(X_train), Y_train)
Y_nn = model.predict(features(X_test))
f1_nn = f1_score(Y_test, Y_nn, average=None, zero_division=0)
### END SOLUTION
### EXERCISE: model = ...   # TODO
### EXERCISE: Y_nn = ...    # TODO
### EXERCISE: f1_nn = ...   # TODO
print(f"{'nuclide':8s} {'F1 ROI':>7s} {'F1 NN':>7s}")
for n, a, b in zip(NUCLIDES, f1_roi, f1_nn):
    print(f"{n:8s} {a:7.2f} {b:7.2f}")
print(f"all nuclides correct: ROI {np.mean((Y_roi == Y_test).all(1))*100:.0f} %  |  "
      f"NN {np.mean((Y_nn == Y_test).all(1))*100:.0f} %")

# %% [markdown] solution
# **Answer.** The network is better (or equal) for every nuclide, especially for the overlapping pairs:
# it uses all the lines, their ratios and the shape of the continuum at once. Note that it was
# trained only on **simulated** spectra: the quality of the simulation (detector model,
# background, shielding, scattering in the source) becomes the limiting factor in practice.

# %% [markdown]
# ## 5 · Be fair: a stronger classical method
#
# The ROI method is a weak baseline. A spectroscopist would rather fit the **whole spectrum** as a
# non-negative combination of the known detector responses (the templates of section 1) plus
# background, and test whether each fitted contribution is significant:
#
# $$ x \approx \sum_i c_i\, T_i + c_B\, T_B, \qquad c_i \ge 0, \qquad z_i = c_i / \sigma(c_i) $$
#
# (weighted non-negative least squares; σ from the Poisson-weighted fit). This is physics-based,
# fast and interpretable.

# %%
A_FIT = np.vstack([T_ISO, T_BKG]).T                    # (512 channels, 9 nuclides + background)

def full_spectrum_fit(x, templates=A_FIT):
    """Significance z_i of each nuclide from a Poisson-weighted non-negative fit."""
    w2 = 1.0 / (x + 1.0)
    w = np.sqrt(w2)
    c, _ = nnls(templates * w[:, None], x * w)
    sigma = np.sqrt(np.diag(np.linalg.pinv((templates * w2[:, None]).T @ templates)))
    return (c / sigma)[: len(NUCLIDES)]

# %% [markdown]
# ### ✏️ Exercise 4 — full-spectrum fit vs. neural network
#
# Compute `full_spectrum_fit` for every test spectrum, decide "present" if z > 3, and compare the
# per-nuclide F1 and the fraction of fully correct spectra with the neural network.

# %%
### BEGIN SOLUTION
Z_fit = np.array([full_spectrum_fit(x) for x in X_test])
Y_fit = (Z_fit > 3).astype(int)
f1_fit = f1_score(Y_test, Y_fit, average=None, zero_division=0)
### END SOLUTION
### EXERCISE: Y_fit = ...    # TODO
### EXERCISE: f1_fit = ...   # TODO
print(f"{'nuclide':8s} {'ROI':>6s} {'fit':>6s} {'NN':>6s}")
for n, a, b, c in zip(NUCLIDES, f1_roi, f1_fit, f1_nn):
    print(f"{n:8s} {a:6.2f} {b:6.2f} {c:6.2f}")
for name, Yp in [("ROI", Y_roi), ("full-spectrum fit", Y_fit), ("neural network", Y_nn)]:
    print(f"{name:18s}: {np.mean((Yp == Y_test).all(1))*100:3.0f} % fully correct")

# %% [markdown] solution
# **Answer.** The full-spectrum fit is as good as the network — here even slightly better.
# No surprise: our test spectra are generated with **exactly** the templates the fit uses, i.e.
# the physics model is perfect. This matches a recent benchmark (Phan *et al.*,
# arXiv:2508.08306, 2025): when the spectral signatures are well known, statistical
# full-spectrum unmixing is very hard to beat; end-to-end ML becomes attractive when
# measurement conditions are uncertain. **Always compare with the best classical method, not
# the weakest one.** The next exercises show where the balance changes.

# %% [markdown]
# ### ✏️ Exercise 5 — how many counts do you need?
#
# Plot the fraction of test spectra with **all** nuclides correctly identified versus the total
# number of counts (use 6 bins in log scale of `N_test`), for the three methods.

# %%
bins = np.geomspace(500, 20000, 7)
centres = np.sqrt(bins[1:] * bins[:-1])
### BEGIN SOLUTION
which = np.digitize(N_test, bins) - 1
acc_roi = [np.mean((Y_roi == Y_test).all(1)[which == b]) for b in range(6)]
acc_nn = [np.mean((Y_nn == Y_test).all(1)[which == b]) for b in range(6)]
acc_fit = [np.mean((Y_fit == Y_test).all(1)[which == b]) for b in range(6)]
### END SOLUTION
### EXERCISE: acc_roi = ...   # TODO: list of 6 accuracies
### EXERCISE: acc_fit = ...   # TODO
### EXERCISE: acc_nn = ...    # TODO
plt.figure(figsize=(5.5, 3.5))
plt.semilogx(centres, acc_roi, "o-", label="ROI (3σ)")
plt.semilogx(centres, acc_fit, "o-", label="full-spectrum fit")
plt.semilogx(centres, acc_nn, "o-", label="neural network")
plt.xlabel("total counts in the spectrum"); plt.ylabel("fraction fully correct")
plt.ylim(0, 1); plt.grid(alpha=0.3, which="both"); plt.legend(); plt.show()

# %% [markdown] solution
# **Answer.** The two methods that use the whole spectrum improve with statistics and are
# already usable with ~1000 counts — a short measurement or a weak source. Surprisingly, the
# ROI method gets **worse** with more counts: at high statistics, any interfering line or
# Compton feature inside a window becomes "significant", so false alarms multiply. A fixed 3σ
# rule ignores interferences; using the whole spectrum (by a fit or by a network) does not.

# %% [markdown]
# ### ✏️ Exercise 6 — the instrument drifts
#
# The gain of a scintillator changes with temperature: a **+4 % gain shift** moves the 662 keV
# peak to ≈ 688 keV. Generate a test set with `gain=(1.04, 1.04)` and evaluate the network
# trained above **and** the full-spectrum fit (which uses the nominal templates). Then retrain the
# network on spectra with a **random gain between 0.95 and 1.05** (*data augmentation*) and
# evaluate again on the drifted test set.

# %%
X_drift, Y_drift, _ = make_spectra(3000, np.random.default_rng(2), gain=(1.04, 1.04))
### BEGIN SOLUTION
acc_before = np.mean((model.predict(features(X_drift)) == Y_drift).all(1))
Z_drift = np.array([full_spectrum_fit(x) for x in X_drift])
acc_fit_drift = np.mean(((Z_drift > 3).astype(int) == Y_drift).all(1))
X_aug, Y_aug, _ = make_spectra(20000, np.random.default_rng(4), gain=(0.95, 1.05))
model_aug = make_model().fit(features(X_aug), Y_aug)
acc_after = np.mean((model_aug.predict(features(X_drift)) == Y_drift).all(1))
acc_nominal = np.mean((model_aug.predict(features(X_test)) == Y_test).all(1))
### END SOLUTION
### EXERCISE: acc_before = ...   # TODO: nominal model on drifted data
### EXERCISE: acc_fit_drift = ... # TODO: full-spectrum fit (nominal templates) on drifted data
### EXERCISE: acc_after = ...    # TODO: augmented model on drifted data
### EXERCISE: acc_nominal = ...  # TODO: augmented model on nominal data
print(f"full-spectrum fit, drifted data: {acc_fit_drift*100:.0f} % fully correct")
print(f"nominal model,   drifted data: {acc_before*100:.0f} % fully correct")
print(f"augmented model, drifted data: {acc_after*100:.0f} % fully correct")
print(f"augmented model, nominal data: {acc_nominal*100:.0f} % fully correct")

# %% [markdown] solution
# **Answer.** A 4 % drift — barely visible by eye — makes **both** the nominal network and the
# full-spectrum fit collapse: both rely on the exact channel positions. Training the network on
# spectra with random gains makes it robust, at almost no cost on nominal data. (The fit could
# also be repaired, by adding the gain as a free parameter: more physics modelling. Both routes
# are legitimate; ML makes it cheap to include *many* nuisance effects at once.) General lesson: a model only knows the
# conditions it was trained on (**domain shift**); include the expected variations of the real
# instrument in the training data (gain, resolution, background, shielding…).

# %% [markdown]
# ### ✏️ Exercise 7 (bonus) — a nuclide the network has never seen
#
# Na-22 (lines at 511 and 1275 keV) is not in the library. Build its template with
# `template([(511.0, 1.80), (1274.5, 0.9994)])`, generate 500 spectra containing only Na-22 and
# background (`lam = (T_BKG + ratio * t_na) / (1 + ratio)`, then Poisson), and look at what the
# network answers. What does this tell you?

# %%
### BEGIN SOLUTION
t_na = template([(511.0, 1.80), (1274.5, 0.9994)])
r5 = np.random.default_rng(5)
X_na = []
for _ in range(500):
    ratio = np.exp(r5.uniform(np.log(1.0), np.log(5.0)))
    lam = (T_BKG + ratio * t_na) / (1 + ratio)
    X_na.append(r5.poisson(np.exp(r5.uniform(np.log(2000), np.log(20000))) * lam))
X_na = np.array(X_na)
pred_na = model.predict(features(X_na))
print(f"spectra with at least one (wrong) nuclide reported: {pred_na.any(1).mean()*100:.0f} %")
for n, frac in zip(NUCLIDES, pred_na.mean(0)):
    print(f"  {n:7s} reported in {frac*100:4.0f} % of the Na-22 spectra")
### END SOLUTION
### EXERCISE: # TODO

# %% [markdown] solution
# **Answer.** The network can only answer with the classes it knows: Na-22 is reported as a
# mixture of known nuclides with similar lines: Co-58 (which also has a 511 keV annihilation
# line) and Co-60 (lines close to 1275 keV) — in essentially every spectrum, with high
# confidence. A classifier is a **closed-world** model. In operation you need an extra check — for
# instance, compare the measured spectrum with the one reconstructed from the identified
# nuclides and raise a flag if the residual is too large (the same idea as the PCA
# reconstruction error of the BPM exercise).

# %% [markdown]
# ## Take-away
#
# * Supervised learning on **simulated** spectra clearly beats simple rule-based identification
#   (ROI), especially for overlapping lines and low statistics…
# * …but a physics-based **full-spectrum fit** is just as good when the detector model is exact.
#   ML pays off when conditions vary (drift, shielding, background) and are hard to model.
# * The model is only as good as the **simulation** and the **range of conditions** it was
#   trained on — augment the training data with realistic variations (drift, background).
# * Classifiers know only their classes: always add a sanity check for the unknown.
#
# **Further reading:** Kamuda, Stinnett & Sullivan, IEEE TNS **64**, 1858 (2017); Kamuda, Zhao &
# Huff, NIM A **954**, 161385 (2020); Turner *et al.*, Sensors **21**, 5238 (2021);
# Phan *et al.*, arXiv:2508.08306 (2025) — ML vs. statistical unmixing benchmark.
