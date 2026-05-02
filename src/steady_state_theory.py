import numpy as np
from typing import List, Callable, Tuple
from scipy.linalg import expm
from .utils import compute_gate_equilibrium_weights

def compute_limiting_pmf_and_moments(Q: np.ndarray,
                             lambdas: np.ndarray,
                             H_funcs: List[Callable[[float], float]],
                             Nmax: int = 30,
                             tol: float = 1e-10,
                             u_max: float = 2e4,
                             du: float = 1.0) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Compute limiting distribution p∞(n) and moments via matrix-Poisson:
      B = ∫_0^∞ e^{Q u} diag(λ_i H_i(u)) du,
      p∞(n) = π^T e^{-B} B^n / n! · 1,
      E[N] = π^T B 1, Var(N) = E[N] + π^T B^2 1 - (E[N])^2.
    H_funcs: list of callables u -> H_i(u).
    """
    Q = np.asarray(Q, float)
    lam = np.asarray(lambdas, float).reshape(-1)
    d = Q.shape[0]
    pi = compute_gate_equilibrium_weights(Q)

    # numeric quadrature for B
    us = np.arange(0.0, float(u_max) + float(du), float(du))
    B = np.zeros_like(Q)
    for k, u in enumerate(us):
        Hdiag = np.diag([lam[i] * float(H_funcs[i](u)) for i in range(d)])
        Eu = expm(Q * u)
        w = 0.5 if (k == 0 or k == len(us) - 1) else 1.0
        B += w * (Eu @ Hdiag) * du
        if k > 10 and all(H_funcs[i](u) < tol for i in range(d)):
            break

    one = np.ones(d, dtype=float)
    EminusB = expm(-B)
    v = EminusB @ one

    ps = np.empty(Nmax + 1, dtype=float)
    ps[0] = np.dot(pi, v).item()
    for n in range(1, Nmax + 1):
        v = (B @ v) / n
        ps[n] = np.dot(pi, v).item()

    EN = np.dot(pi, B @ one).item()
    EN2_fact = np.dot(pi, B @ (B @ one)).item()
    VarN = EN + EN2_fact - EN * EN

    s = ps.sum()
    if s > 0:
        ps /= s

    return ps, EN, VarN, B, pi
