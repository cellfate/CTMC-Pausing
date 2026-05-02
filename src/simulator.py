import numpy as np, heapq
from dataclasses import dataclass
from typing import List
from .utils import compute_gate_equilibrium_weights


@dataclass
class SimResult:
    S: np.ndarray      # arrival times
    C: np.ndarray      # completion times
    states: np.ndarray # arrival states
    
# -----------------------------
# Event-based simulator (single trajectory)
# -----------------------------
def simulate_single_path(Q, lambdas, L, W_means, v_means, 
                         Tterm_means, T_max, i0=None, pi0=None, rng=None):                     
    """
    Simulate one trajectory up to T_max.
    - Q: (d x d) generator, off-diagonals >=0, rows sum to 0
    - lambdas[i]: initiation rate when background state is i
    - Lifetime depends only on *arrival* state i
    """                    
    rng = np.random.default_rng() if rng is None else rng
    Q = np.asarray(Q, float)
    lam = np.asarray(lambdas, float)
    d = Q.shape[0]
    nu = -np.diag(Q)

    # init background
    if i0 is None:
        if pi0 is None:
            pi0 = np.ones(d) / d
        i = int(rng.choice(d, p=pi0))
    else:
        i = int(i0)

    t = 0.0
    comp_heap = []  # min-heap of completion times

    S_list, C_list, st_list = [], [], []

    def exp_time(rate):
        return np.inf if rate <= 0 else rng.exponential(1.0 / rate)

    while True:
        rate_ab = lam[i] + nu[i]                     # 合并时钟
        dt_ab = rng.exponential(1.0 / rate_ab) if rate_ab > 0 else np.inf
        dt_cmp = (comp_heap[0] - t) if comp_heap else np.inf
        dt = dt_ab if dt_ab <= dt_cmp else dt_cmp
        if not np.isfinite(dt) or (t + dt) > T_max:
            break
        t += dt

        if dt_cmp <= dt_ab:
            # —— 完成事件 ——（可能同一时刻有多个，按需要 while 弹出）
            heapq.heappop(comp_heap)
            # 如果会有多重并发完成，可在这里 while 顶堆==t 继续弹
        else:
            # —— 合并时钟主导：判定到达 or 背景跳转 ——
            if rng.random() < (lam[i] / rate_ab):
                # 到达：寿命 = Exp(W_i) + L/v_i + Exp(Tterm_i)
                w  = rng.exponential(W_means[i]) if W_means[i] > 0 else 0.0
                c  = L / max(v_means[i], 1e-9)
                tt = rng.exponential(Tterm_means[i]) if Tterm_means[i] > 0 else 0.0
                life = w + c + tt
                S_list.append(t); C_list.append(t + life); st_list.append(i)
                heapq.heappush(comp_heap, t + life)
            else:
                # 背景跳转：按 Q 行归一化选新态
                if nu[i] > 0:
                    probs = np.maximum(Q[i,:], 0.0).copy()
                    probs[i] = 0.0
                    probs /= probs.sum()
                    i = rng.choice(len(probs), p=probs)

    return SimResult(S=np.array(S_list, float), C=np.array(C_list, float), states=np.array(st_list, int))

def simulate_ensemble_N_grid(Q, lambdas, L, W_means, v_means, Tterm_means,
                             T_max, edges, R, pi0=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if pi0 is None: pi0 = compute_gate_equilibrium_weights(Q)
    Nmat = np.zeros((R, len(edges)-1), dtype=float)
    for r in range(R):
        sim = simulate_single_path(Q, lambdas, L, W_means, v_means, Tterm_means,
                                   T_max=T_max, pi0=pi0, rng=rng)
        starts, _ = np.histogram(sim.S, bins=edges)
        ends, _   = np.histogram(sim.C, bins=edges)
        Nmat[r,:] = np.cumsum(starts - ends)
    return Nmat