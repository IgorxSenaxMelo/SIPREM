"""Fuzzy PROMETHEE do SiPREM."""

from __future__ import annotations
import math
import numpy as np

from .fuzzy_numbers import trap_add, trap_sub, trap_scalar_div, trap_mul_geldermann, defuzz_coa


def pref_scalar(d, pref_type, q, p):
    d = max(0.0, float(d))

    if pref_type == 1:  # Usual
        return float(d > 0)
    if pref_type == 2:  # U-shape
        return 0.0 if d <= q else 1.0
    if pref_type == 3:  # V-shape
        if d <= 0:
            return 0.0
        if d <= p:
            return d/p
        return 1.0
    if pref_type == 4:  # Level
        if d <= q:
            return 0.0
        if d <= p:
            return 0.5
        return 1.0
    if pref_type == 5:  # Linear
        if d <= q:
            return 0.0
        if d <= p:
            return (d-q)/(p-q)
        return 1.0
    if pref_type == 6:  # Gaussian
        if d <= 0:
            return 0.0
        sigma = p/3
        return 1 - math.exp(-(d*d)/(2*sigma*sigma))

    raise ValueError(f"Tipo de preferência não implementado: {pref_type}")


def pref_trap(D, pref_type, q, p):
    out = np.array([pref_scalar(x, pref_type, q, p) for x in D], dtype=float)
    return np.sort(out)


def run_fuzzy_promethee_trap_reference(F, w_fuzzy, crit_dir, pref_types, q_vals, p_vals):
    """Versão original (laços Python). Mantida como referência para os testes."""
    F = np.asarray(F, dtype=float)
    w_fuzzy = np.asarray(w_fuzzy, dtype=float)
    n_req, n_crit, n_comp = F.shape

    if n_comp != 4:
        raise ValueError("F deve conter trapézios [a,b,c,d].")

    PI = np.zeros((n_req, n_req, 4), dtype=float)

    for i in range(n_req):
        for j in range(n_req):
            if i == j:
                continue

            pi_ij = np.zeros(4, dtype=float)
            for k in range(n_crit):
                Ai = F[i, k]
                Bj = F[j, k]

                if crit_dir[k] == 1:
                    D = trap_sub(Ai, Bj)
                else:
                    D = trap_sub(Bj, Ai)

                Pk = pref_trap(D, pref_types[k], q_vals[k], p_vals[k])
                term = trap_mul_geldermann(w_fuzzy[k], Pk)
                pi_ij = trap_add(pi_ij, term)

            PI[i, j] = pi_ij

    phi_plus_fuzzy = np.zeros((n_req, 4))
    phi_minus_fuzzy = np.zeros((n_req, 4))
    phi_net_fuzzy = np.zeros((n_req, 4))
    phi_plus = np.zeros(n_req)
    phi_minus = np.zeros(n_req)
    phi_net = np.zeros(n_req)

    for i in range(n_req):
        sum_p = np.zeros(4)
        sum_m = np.zeros(4)

        for j in range(n_req):
            if i == j:
                continue
            sum_p = trap_add(sum_p, PI[i, j])
            sum_m = trap_add(sum_m, PI[j, i])

        avg_p = trap_scalar_div(sum_p, n_req-1)
        avg_m = trap_scalar_div(sum_m, n_req-1)
        net_f = trap_sub(avg_p, avg_m)

        phi_plus_fuzzy[i] = avg_p
        phi_minus_fuzzy[i] = avg_m
        phi_net_fuzzy[i] = net_f

        phi_plus[i] = defuzz_coa(avg_p)
        phi_minus[i] = defuzz_coa(avg_m)
        phi_net[i] = defuzz_coa(net_f)

    return phi_plus, phi_minus, phi_net, phi_plus_fuzzy, phi_minus_fuzzy, phi_net_fuzzy


def rankdata_average(values):
    """Equivalente ao tiedrank do MATLAB para empates: rank médio, crescente."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j+1]] == x[order[i]]:
            j += 1
        ranks[order[i:j+1]] = (i+1 + j+1) / 2
        i = j + 1
    return ranks


def rank_from_net_flow(phi_net):
    n = len(phi_net)
    return n + 1 - rankdata_average(phi_net)


# ---------------------------------------------------------------------------
# Versão vetorizada (mesmos resultados da referência, ~100x mais rápida)
# ---------------------------------------------------------------------------

def pref_array(D, pref_type, q, p):
    """Função de preferência aplicada a um array inteiro de diferenças."""
    d = np.maximum(np.asarray(D, dtype=float), 0.0)

    if pref_type == 1:  # Usual
        return (d > 0).astype(float)
    if pref_type == 2:  # U-shape
        return np.where(d <= q, 0.0, 1.0)
    if pref_type == 3:  # V-shape
        ratio = d / p if p > 0 else np.ones_like(d)
        return np.where(d <= 0, 0.0, np.where(d <= p, ratio, 1.0))
    if pref_type == 4:  # Level
        return np.where(d <= q, 0.0, np.where(d <= p, 0.5, 1.0))
    if pref_type == 5:  # Linear
        ratio = (d - q) / (p - q) if p > q else np.ones_like(d)
        return np.where(d <= q, 0.0, np.where(d <= p, ratio, 1.0))
    if pref_type == 6:  # Gaussiana
        if p <= 0:
            return (d > 0).astype(float)
        sigma = p / 3
        return np.where(d <= 0, 0.0, 1 - np.exp(-(d * d) / (2 * sigma * sigma)))

    raise ValueError(f"Tipo de preferência não implementado: {pref_type}")


def pairwise_preference(F_k, direction, pref_type, q, p):
    """Preferência trapezoidal P_k(i,j) para todos os pares. Retorna (n, n, 4)."""
    F_k = np.asarray(F_k, dtype=float)
    A = F_k[:, None, :]
    B = F_k[None, :, :]
    X, Y = (A, B) if direction == 1 else (B, A)

    D = np.stack([
        X[..., 0] - Y[..., 3],
        X[..., 1] - Y[..., 2],
        X[..., 2] - Y[..., 1],
        X[..., 3] - Y[..., 0],
    ], axis=-1)

    P = np.sort(pref_array(D, pref_type, q, p), axis=-1)
    idx = np.arange(F_k.shape[0])
    P[idx, idx, :] = 0.0
    return P


def mul_geldermann_array(A, B):
    """trap_mul_geldermann vetorizado (último eixo = [a,b,c,d])."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    ml1, mu1 = A[..., 1], A[..., 2]
    alpha1, beta1 = A[..., 1] - A[..., 0], A[..., 3] - A[..., 2]
    ml2, mu2 = B[..., 1], B[..., 2]
    alpha2, beta2 = B[..., 1] - B[..., 0], B[..., 3] - B[..., 2]

    ml = ml1 * ml2
    mu = mu1 * mu2
    alpha = np.maximum(ml1 * alpha2 + ml2 * alpha1 - alpha1 * alpha2, 0.0)
    beta = np.maximum(mu1 * beta2 + mu2 * beta1 + beta1 * beta2, 0.0)

    C = np.stack([ml - alpha, ml, mu, mu + beta], axis=-1)
    C[C < 0] = 0.0
    return np.sort(C, axis=-1)


def defuzz_coa_array(T):
    """defuzz_coa vetorizado para um array (..., 4)."""
    T = np.asarray(T, dtype=float)
    a, b, c, d = T[..., 0], T[..., 1], T[..., 2], T[..., 3]
    den = 3 * (c + d - a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        coa = ((c + d) ** 2 - c * d - (a + b) ** 2 + a * b) / den
    out = np.where(np.abs(den) < 1e-12, T.mean(axis=-1), coa)
    return np.where(np.abs(d - a) < 1e-12, b, out)


def _trap_sub_array(A, B):
    return np.stack([A[..., 0] - B[..., 3], A[..., 1] - B[..., 2],
                     A[..., 2] - B[..., 1], A[..., 3] - B[..., 0]], axis=-1)


def run_fuzzy_promethee_full(F, w_fuzzy, crit_dir, pref_types, q_vals, p_vals):
    """Fuzzy PROMETHEE vetorizado.

    Retorna um dicionário com os fluxos agregados e também os fluxos líquidos
    unicritério (n_req x n_crit), usados no GAIA e na explicação por critério.
    """
    F = np.asarray(F, dtype=float)
    w_fuzzy = np.asarray(w_fuzzy, dtype=float)
    n_req, n_crit, n_comp = F.shape

    if n_comp != 4:
        raise ValueError("F deve conter trapézios [a,b,c,d].")
    if n_req < 2:
        raise ValueError("São necessários pelo menos 2 requisitos.")

    PI = np.zeros((n_req, n_req, 4), dtype=float)
    uni_net = np.zeros((n_req, n_crit), dtype=float)

    for k in range(n_crit):
        P = pairwise_preference(F[:, k, :], crit_dir[k], pref_types[k], q_vals[k], p_vals[k])
        PI += mul_geldermann_array(w_fuzzy[k], P)

        uni_p = P.sum(axis=1) / (n_req - 1)
        uni_m = P.sum(axis=0) / (n_req - 1)
        uni_net[:, k] = defuzz_coa_array(_trap_sub_array(uni_p, uni_m))

    avg_p = PI.sum(axis=1) / (n_req - 1)
    avg_m = PI.sum(axis=0) / (n_req - 1)
    net_f = _trap_sub_array(avg_p, avg_m)

    return {
        "phi_plus": defuzz_coa_array(avg_p),
        "phi_minus": defuzz_coa_array(avg_m),
        "phi_net": defuzz_coa_array(net_f),
        "phi_plus_fuzzy": avg_p,
        "phi_minus_fuzzy": avg_m,
        "phi_net_fuzzy": net_f,
        "unicriterion_net": uni_net,
    }


def run_fuzzy_promethee_trap(F, w_fuzzy, crit_dir, pref_types, q_vals, p_vals):
    """Mesma assinatura/retorno da versão original, agora vetorizada."""
    out = run_fuzzy_promethee_full(F, w_fuzzy, crit_dir, pref_types, q_vals, p_vals)
    return (out["phi_plus"], out["phi_minus"], out["phi_net"],
            out["phi_plus_fuzzy"], out["phi_minus_fuzzy"], out["phi_net_fuzzy"])


def promethee_i_relations(phi_plus, phi_minus, eps=1e-10):
    """Relações do PROMETHEE I: 'P' (i domina j), 'P-' (i dominado), 'I', 'R'."""
    plus = np.asarray(phi_plus, dtype=float)
    minus = np.asarray(phi_minus, dtype=float)

    pi, pj = plus[:, None], plus[None, :]
    mi, mj = minus[:, None], minus[None, :]

    dom = (pi > pj + eps) & (mi < mj - eps)
    sub = (pi < pj - eps) & (mi > mj + eps)
    ind = (np.abs(pi - pj) <= eps) & (np.abs(mi - mj) <= eps)

    rel = np.full(dom.shape, "R", dtype=object)
    rel[ind] = "I"
    rel[sub] = "P-"
    rel[dom] = "P"
    np.fill_diagonal(rel, "I")
    return rel
