"""Minimal linear transverse optics (numpy) used to generate the slide figures.

Same thick-element transfer matrices as Cheetah's 'linear' tracking for drifts
and quadrupoles (k1 in 1/m^2, k1 > 0 focuses in x).
"""
import numpy as np


def drift(L):
    return np.array([[1.0, L], [0.0, 1.0]])


def quad(L, k1):
    """2x2 matrices (Mx, My) of a thick quadrupole."""
    def m(k):
        if abs(k) < 1e-12:
            return drift(L)
        if k > 0:
            s = np.sqrt(k)
            return np.array([[np.cos(s * L), np.sin(s * L) / s],
                             [-s * np.sin(s * L), np.cos(s * L)]])
        s = np.sqrt(-k)
        return np.array([[np.cosh(s * L), np.sinh(s * L) / s],
                         [s * np.sinh(s * L), np.cosh(s * L)]])
    return m(k1), m(-k1)


# Beamline used in the course and the notebooks (electron transport line)
#  D0 | Q1 | D1 | Q2 | D2 | Q3 | D3 | screen
LAYOUT = dict(d0=0.5, lq=0.12, d1=0.4, d2=0.4, d3=1.5)
BEAM = dict(beta=5.0, alpha=-1.0, emit=2e-9)   # same in x and y


def sigma_screen(k, layout=LAYOUT, beam=BEAM):
    """RMS beam sizes (sx, sy) in metres at the screen for k = (k1_Q1, k1_Q2, k1_Q3)."""
    b, a, e = beam["beta"], beam["alpha"], beam["emit"]
    g = (1 + a * a) / b
    S0 = e * np.array([[b, -a], [-a, g]])
    out = []
    for plane in (0, 1):
        M = drift(layout["d0"])
        for i, ki in enumerate(k):
            M = quad(layout["lq"], ki)[plane] @ M
            M = drift([layout["d1"], layout["d2"], layout["d3"]][i]) @ M
        S = M @ S0 @ M.T
        out.append(np.sqrt(S[0, 0]))
    return np.array(out)
