import numpy as np

def compute_gate_equilibrium_weights(Q: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, float)
    d = Q.shape[0]
    Aeq = np.vstack([Q.T, np.ones(d)])
    beq = np.zeros(d + 1); beq[-1] = 1.0
    pi, *_ = np.linalg.lstsq(Aeq, beq, rcond=None)
    pi = np.clip(np.asarray(pi, float).reshape(-1), 0.0, None)
    s = pi.sum()
    return pi / (s if s > 0 else 1.0)

def count_modes_from_pmf(p, prominence=1e-12, include_boundary=True):
    """
    Count modes of a discrete pmf p over n=0,1,2,...
    A peak at n=0 is included if include_boundary=True and p[0] > p[1] + prominence.
    Returns (k_total, peaks) where peaks is a list of indices (including 0 if counted).
    """
    p = np.asarray(p, float)
    peaks = []
    # Left boundary at n=0
    if include_boundary and p[0] > p[1] + prominence:
        peaks.append(0)
    # Interior peaks
    for i in range(1, len(p) - 1):
        if p[i] > p[i - 1] + prominence and p[i] > p[i + 1] + prominence:
            peaks.append(i)
    return len(peaks), peaks
