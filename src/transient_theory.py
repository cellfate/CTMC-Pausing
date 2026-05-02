import numpy as np
from .utils import compute_gate_equilibrium_weights
from .lifetime_kernels import survival_shifted_two_exp

def compute_transient_pmf_and_moments(Q, lambdas, L, W_means, v_means, Tterm_means,
                              t_grid, Nmax, pi0=None):
    """
    Evolve the PMF P_n(j,t) (conditioned on observation background state j at time t) via:
      dP_0/dt = Q P_0 - A(t) P_0
      dP_n/dt = A(t) P_{n-1} + Q P_n - A(t) P_n,  n>=1
    and the mean vector m(j,t) via:
      dm/dt = A(t) 1 + Q m
    where A_ii(t) = lambda_i H_i(t).
    Euler step with per-state renormalization across n.
    """
    Q = np.asarray(Q, float)
    lam = np.asarray(lambdas, float)
    d = Q.shape[0]
    dt = float(np.diff(t_grid)[0])
    T = len(t_grid)

    # H_i(t)
    c = L / np.maximum(v_means, 1e-9)
    H = np.zeros((d, T))
    for i in range(d):
        H[i, :] = survival_shifted_two_exp(t_grid, c[i], W_means[i], Tterm_means[i])
    A = (lam[:, None] * H)  # (d x T)

    # PMFs P_n(j,t) and mean m(j,t)
    P = np.zeros((Nmax + 1, d, T))
    P[0, :, 0] = 1.0
    m = np.zeros((d, T))

    for k in range(T - 1):
        # P0
        P[0, :, k+1] = P[0, :, k] + dt * (Q @ P[0, :, k] - A[:, k] * P[0, :, k])
        # Pn
        for n in range(1, Nmax + 1):
            P[n, :, k+1] = P[n, :, k] + dt * (A[:, k] * P[n-1, :, k] + (Q @ P[n, :, k]) - A[:, k] * P[n, :, k])

        # numerical hygiene: clamp negatives & renormalize over n for each obs-state
        P[:, :, k+1] = np.maximum(P[:, :, k+1], 0.0)
        mass = P[:, :, k+1].sum(axis=0, keepdims=True)
        P[:, :, k+1] /= np.where(mass <= 0, 1.0, mass)

        # mean vector
        m[:, k+1] = m[:, k] + dt * (A[:, k] + (Q @ m[:, k]))

    # initial observation background distribution
    if pi0 is None:
        Aeq = np.vstack([Q.T, np.ones(d)])
        beq = np.zeros(d + 1); beq[-1] = 1.0
        pi0, *_ = np.linalg.lstsq(Aeq, beq, rcond=None)
        pi0 = np.maximum(pi0, 0.0); pi0 = pi0 / pi0.sum()

    # unconditional PMF and means
    p_nt = np.tensordot(pi0, P, axes=(0, 1))  # (Nmax+1, T)
    n_vals = np.arange(Nmax + 1)[:, None]
    mean_from_p = (n_vals * p_nt).sum(axis=0)
    mean_vec = pi0 @ m

    norm_err = float(np.max(np.abs(p_nt.sum(axis=0) - 1.0)))
    return P, p_nt,  mean_from_p, mean_vec, pi0, norm_err
