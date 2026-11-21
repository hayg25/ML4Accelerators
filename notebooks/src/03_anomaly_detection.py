# %% [markdown]
# # Exercise 3 — Finding faulty beam position monitors (unsupervised learning)
#
# **ML for Accelerator Physics · beginner course · H. Guler (IJCLab)**
#
# {{COLAB}}
#
# Beam position monitors (**BPMs**) measure the beam centroid along the machine. A large
# machine has hundreds of them, and some occasionally misbehave (electronics offsets, dead
# channels, noise). Faulty readings spoil orbit correction and optics measurements.
#
# Nobody labels faults in the logged data → we cannot train a classifier. Instead we use
# **unsupervised anomaly detection**: learn what *normal* data look like, and flag what does not fit.
#
# You will:
#
# 1. simulate shot-by-shot BPM readings with Cheetah (orbit jitter from two steering magnets),
# 2. corrupt a few readings with realistic faults (hidden labels, used only for evaluation),
# 3. detect them with **PCA** (reconstruction error) and an **Isolation Forest**,
# 4. find *which* BPM is faulty.

# %%
import sys
if "google.colab" in sys.modules:
    !pip install -q "cheetah-accelerator>=0.8,<0.9"

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import cheetah
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# %% [markdown]
# ## 1 · A transport line with steerers and BPMs
#
# Two horizontal steering magnets (correctors) followed by five FODO cells
# (focusing quad `QF`, defocusing quad `QD`), with a BPM after each quadrupole: **10 BPMs**.

# %%
elements = [
    cheetah.HorizontalCorrector(length=torch.tensor(0.02), name="HCM1"),
    cheetah.Drift(length=torch.tensor(0.5)),
    cheetah.HorizontalCorrector(length=torch.tensor(0.02), name="HCM2"),
    cheetah.Drift(length=torch.tensor(0.5)),
]
for cell in range(5):
    elements += [
        cheetah.Quadrupole(length=torch.tensor(0.12), k1=torch.tensor(10.0), name=f"QF{cell}"),
        cheetah.BPM(name=f"BPM{2 * cell}"),
        cheetah.Drift(length=torch.tensor(1.0)),
        cheetah.Quadrupole(length=torch.tensor(0.12), k1=torch.tensor(-10.0), name=f"QD{cell}"),
        cheetah.BPM(name=f"BPM{2 * cell + 1}"),
        cheetah.Drift(length=torch.tensor(1.0)),
    ]
segment = cheetah.Segment(elements=elements)
bpm_names = [el.name for el in segment.elements if isinstance(el, cheetah.BPM)]
print(len(bpm_names), "BPMs:", bpm_names)

incoming = cheetah.ParameterBeam.from_twiss(
    beta_x=torch.tensor(5.0), alpha_x=torch.tensor(0.0), emittance_x=torch.tensor(2e-9),
    beta_y=torch.tensor(5.0), alpha_y=torch.tensor(0.0), emittance_y=torch.tensor(2e-9),
    energy=torch.tensor(100e6),
)

def read_bpms(angle1, angle2):
    """Horizontal beam position (µm) at every BPM for given corrector kicks (rad)."""
    segment.HCM1.angle = torch.tensor(float(angle1))
    segment.HCM2.angle = torch.tensor(float(angle2))
    beam, readings = incoming, []
    for el in segment.elements:
        beam = el.track(beam)
        if isinstance(el, cheetah.BPM):
            readings.append(beam.mu_x.item() * 1e6)
    return np.array(readings)

plt.figure(figsize=(7, 3))
for a1, a2 in [(50e-6, 0), (0, 50e-6), (30e-6, -40e-6)]:
    plt.plot(read_bpms(a1, a2), "o-", label=f"kicks = ({a1*1e6:.0f}, {a2*1e6:.0f}) µrad")
plt.xlabel("BPM index"); plt.ylabel("x [µm]"); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()

# %% [markdown]
# ## 2 · Simulate 2000 shots of "logged machine data"
#
# Every shot, the two steerers jitter randomly (σ = 30 µrad, e.g. power-supply ripple), and
# each BPM adds 5 µm of electronic noise.

# %%
N_SHOTS, BPM_NOISE = 2000, 5.0
kicks = rng.normal(0.0, 30e-6, size=(N_SHOTS, 2))
X_clean = np.array([read_bpms(*k) for k in kicks])
X_clean += rng.normal(0.0, BPM_NOISE, size=X_clean.shape)
print("data matrix:", X_clean.shape, " (shots × BPMs)")

# %% [markdown]
# Now we corrupt 4 % of the shots. In each corrupted shot, **one** BPM has one of three faults:
#
# * **offset** — reading shifted by 60 µm (calibration jump),
# * **dead** — reading stuck at 0,
# * **noisy** — 15× more electronic noise.
#
# `y_true` (1 = faulty shot) and `bad_bpm` are the *hidden truth*: the algorithms never see them.

# %%
X = X_clean.copy()
y_true = np.zeros(N_SHOTS, dtype=int)
bad_bpm = np.full(N_SHOTS, -1)
fault_type = np.array([""] * N_SHOTS, dtype=object)

faulty = rng.choice(N_SHOTS, size=int(0.04 * N_SHOTS), replace=False)
for i in faulty:
    j = rng.integers(len(bpm_names))
    kind = rng.choice(["offset", "dead", "noisy"])
    if kind == "offset":
        X[i, j] += 60.0 * rng.choice([-1, 1])
    elif kind == "dead":
        X[i, j] = 0.0
    else:
        X[i, j] += rng.normal(0.0, 15 * BPM_NOISE)
    y_true[i], bad_bpm[i], fault_type[i] = 1, j, kind
print(f"{y_true.sum()} faulty shots out of {N_SHOTS}")

# %% [markdown]
# ### ✏️ Exercise 1 — can you see the faults?
#
# Plot 30 normal orbits in grey and 5 faulty orbits in red (use `y_true` only for colouring).
# Is it easy to spot the faulty BPM by eye?

# %%
plt.figure(figsize=(7, 3.5))
### BEGIN SOLUTION
for i in np.where(y_true == 0)[0][:30]:
    plt.plot(X[i], color="grey", alpha=0.4)
for i in np.where(y_true == 1)[0][:5]:
    plt.plot(X[i], color="red")
    plt.plot(bad_bpm[i], X[i, bad_bpm[i]], "rx", ms=10)
### END SOLUTION
### EXERCISE: # TODO
plt.xlabel("BPM index"); plt.ylabel("x [µm]"); plt.grid(alpha=0.3); plt.show()

# %% [markdown] solution
# **Answer.** Hardly: a 60 µm offset or a "dead" BPM is well within the range of normal
# orbits (±200 µm). Looking at one BPM at a time (a threshold on each channel) cannot work.
# What *is* abnormal is that the faulty reading is **inconsistent with the other BPMs**:
# physically, the orbit is entirely determined by the two kicks.

# %% [markdown]
# ## 3 · Method 1 — PCA: normal orbits live in a small subspace
#
# 10 BPM readings, but only **2 physical causes** (the two kicks). So normal orbits all lie
# (up to noise) on a 2-D plane inside the 10-D space. **Principal component analysis** (PCA)
# finds this plane from the data alone.
#
# A shot that is *far* from the plane is suspicious: we use the **reconstruction error**
# (distance to the plane) as an anomaly score.

# %%
pca_full = PCA().fit(X)
plt.figure(figsize=(5, 3))
plt.semilogy(np.arange(1, 11), pca_full.explained_variance_, "o-")
plt.xlabel("principal component"); plt.ylabel("variance [µm²]"); plt.grid(alpha=0.3); plt.show()

# %% [markdown]
# Two components carry almost all the variance; the rest is noise (and faults). The data
# themselves tell us the machine has two "knobs" — without any physics model!

# %% [markdown]
# ### ✏️ Exercise 2 — reconstruction error
#
# 1. Fit a `PCA(n_components=2)` on `X`.
# 2. Project and reconstruct: `X_rec = pca.inverse_transform(pca.transform(X))`.
# 3. The score of a shot is `np.sqrt(((X - X_rec)**2).sum(axis=1))`.
# 4. Plot the histogram of scores for normal and faulty shots (log y-axis).

# %%
### BEGIN SOLUTION
pca = PCA(n_components=2).fit(X)
X_rec = pca.inverse_transform(pca.transform(X))
residual = X - X_rec
score_pca = np.sqrt((residual ** 2).sum(axis=1))
### END SOLUTION
### EXERCISE: pca = ...        # TODO
### EXERCISE: X_rec = ...      # TODO
### EXERCISE: residual = X - X_rec
### EXERCISE: score_pca = ...  # TODO

bins = np.linspace(0, score_pca.max(), 60)
plt.figure(figsize=(6, 3))
plt.hist(score_pca[y_true == 0], bins=bins, alpha=0.6, label="normal")
plt.hist(score_pca[y_true == 1], bins=bins, alpha=0.6, label="faulty")
plt.yscale("log"); plt.xlabel("PCA reconstruction error [µm]"); plt.legend(); plt.show()
print(f"ROC AUC (PCA) = {roc_auc_score(y_true, score_pca):.3f}   (1 = perfect, 0.5 = random)")

# %% [markdown]
# ## 4 · Method 2 — Isolation Forest
#
# An **Isolation Forest** builds random decision trees that split the data with random cuts.
# Anomalies are "few and different", so they get isolated after only a few cuts.
# It needs no assumption about linearity. It was used to detect faulty BPMs at the LHC
# (Fol *et al.*, PRAB **23**, 102805, 2020).

# %%
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED).fit(X)
score_iso = -iso.score_samples(X)          # higher = more anomalous
print(f"ROC AUC (Isolation Forest on raw readings) = {roc_auc_score(y_true, score_iso):.3f}")

# %% [markdown]
# ### ✏️ Exercise 3 — combine physics insight and ML
#
# The raw readings are dominated by the (normal) orbit jitter, which hides the faults.
# Train the Isolation Forest on the **PCA residuals** (`residual`) instead of the raw
# readings. Compare the ROC AUC.

# %%
### BEGIN SOLUTION
iso_res = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED).fit(residual)
score_iso_res = -iso_res.score_samples(residual)
### END SOLUTION
### EXERCISE: score_iso_res = ...   # TODO
print(f"ROC AUC (Isolation Forest on PCA residuals) = {roc_auc_score(y_true, score_iso_res):.3f}")

# %% [markdown] solution
# **Answer.** Much better. Removing the "explained" part of the signal (the 2-D orbit) leaves
# the part that *no normal shot can produce*. Feature engineering guided by physics often
# beats a more sophisticated model on raw data.

# %% [markdown]
# ### ✏️ Exercise 4 — from a score to a decision
#
# An operator needs a yes/no flag. Choose a threshold on `score_pca` (e.g. the 97th percentile
# of all scores) and compute the confusion matrix, precision and recall.
# How do they change if you lower the threshold?

# %%
### BEGIN SOLUTION
for pct in (97, 95, 90):
    thr = np.percentile(score_pca, pct)
    y_pred = (score_pca > thr).astype(int)
    print(f"threshold = {pct}th percentile ({thr:5.1f} µm): "
          f"precision = {precision_score(y_true, y_pred):.2f}, recall = {recall_score(y_true, y_pred):.2f}")
print(confusion_matrix(y_true, (score_pca > np.percentile(score_pca, 97)).astype(int)))
### END SOLUTION
### EXERCISE: # TODO

# %% [markdown] solution
# **Answer.** A lower threshold catches more faults (higher recall) but raises more false
# alarms (lower precision). The right compromise depends on the cost of each error — an
# operations decision, not an ML one. Some faults (e.g. a "dead" BPM where the orbit is
# naturally close to 0, or a small "noisy" excursion) are intrinsically undetectable.

# %% [markdown]
# ### ✏️ Exercise 5 — which BPM is faulty?
#
# For each flagged shot, the faulty BPM is likely the one with the **largest absolute
# residual**. Compute the fraction of correctly identified BPMs among the truly faulty shots,
# and break it down by `fault_type`.

# %%
### BEGIN SOLUTION
guess = np.argmax(np.abs(residual), axis=1)
ok = guess[y_true == 1] == bad_bpm[y_true == 1]
print(f"correct BPM identified in {ok.mean()*100:.0f} % of faulty shots")
for kind in ("offset", "dead", "noisy"):
    m = fault_type[y_true == 1] == kind
    print(f"  {kind:6s}: {ok[m].mean()*100:5.1f} %  ({m.sum()} shots)")
### END SOLUTION
### EXERCISE: # TODO

# %% [markdown]
# ## Take-away
#
# * **Unsupervised** learning needs no labels: learn "normal", flag the rest.
# * Correlations between sensors are the key: a reading is suspicious if it is *inconsistent*
#   with the others, not because it is large.
# * Simple, interpretable methods (PCA) are strong baselines; physics-informed features help a lot.
# * Turning a score into an alarm is a trade-off between missed faults and false alarms.
#
# **In the literature:** faulty BPM detection at the LHC with Isolation Forest (Fol *et al.*,
# PRAB **23**, 102805, 2020); SRF cavity fault classification at CEBAF (Tennant *et al.*,
# PRAB **23**, 114601, 2020).
