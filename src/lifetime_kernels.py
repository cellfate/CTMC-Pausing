import numpy as np

# Utility: Survival H_i(t) for tau_i = Exp(w) + c (det) + Exp(tterm)
# -----------------------------
def survival_shifted_two_exp(t, c, w_mean, tterm_mean):
    """
    Survival H(t) for W ~ Exp(1/w_mean), T ~ Exp(1/tterm_mean), tau = W + c + T.
    c is deterministic (L/v). Works elementwise on t (numpy array).
    """
    t = np.asarray(t, float)
    mu = 1.0 / w_mean if w_mean > 0 else np.inf
    nu = 1.0 / tterm_mean if tterm_mean > 0 else np.inf

    out = np.ones_like(t)
    s = t - c
    mask = s >= 0
    s_pos = s[mask]

    if np.isfinite(mu) and np.isfinite(nu):
        if abs(mu - nu) < 1e-12:
            r = mu  # limit mu -> nu
            out[mask] = np.exp(-r * s_pos) * (1.0 + r * s_pos)  # Gamma(k=2, rate=r) survival
        else:
            out[mask] = (nu * np.exp(-mu * s_pos) - mu * np.exp(-nu * s_pos)) / (nu - mu)
    elif np.isfinite(mu) and not np.isfinite(nu):
        out[mask] = np.exp(-mu * s_pos)
    elif not np.isfinite(mu) and np.isfinite(nu):
        out[mask] = np.exp(-nu * s_pos)
    else:
        out[mask] = 0.0

    return np.clip(out, 0.0, 1.0)


# def build_survival_funcs_from_shifted_two_exp(L, W_means, v_means, Tterm_means):
#     W_means = np.asarray(W_means, float).reshape(-1)
#     v_means = np.asarray(v_means, float).reshape(-1)
#     Tterm_means = np.asarray(Tterm_means, float).reshape(-1)
#     c_vec = L / np.maximum(v_means, 1e-9)
#     funcs = []
#     for i in range(len(W_means)):
#         wi, vi, ti, ci = W_means[i], v_means[i], Tterm_means[i], c_vec[i]
#         funcs.append(lambda u, wi=wi, ti=ti, ci=ci:
#                      float(survival_shifted_two_exp(np.array([u]), ci, wi, ti)[0]))
#     return funcs


def build_survival_funcs_from_shifted_two_exp(L, W_means, v_means, Tterm_means):
    W_means = np.asarray(W_means, float).reshape(-1)
    v_means = np.asarray(v_means, float).reshape(-1)
    Tterm_means = np.asarray(Tterm_means, float).reshape(-1)
    c_vec = L / np.maximum(v_means, 1e-9)

    funcs = []
    for i in range(len(W_means)):
        wi, vi, ti, ci = W_means[i], v_means[i], Tterm_means[i], c_vec[i]

        # ✅ 关键改动：既能接受标量 u，也能接受数组 u
        funcs.append(
            lambda u, wi=wi, ti=ti, ci=ci:
                survival_shifted_two_exp(u, ci, wi, ti)
        )

    return funcs
