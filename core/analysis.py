"""Análises complementares do SiPREM (sem dependência de Streamlit/Plotly)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data_loader import CRITERIA
from .promethee import promethee_i_relations, rankdata_average

CLASS_ORDER = ["Key", "Grupo 1", "Grupo 2", "Grupo 3"]
CLASS_SHORT = {"Key": "Key", "Grupo 1": "G1", "Grupo 2": "G2", "Grupo 3": "G3"}
CLASS_CODE = {c: i for i, c in enumerate(CLASS_ORDER)}


def req_labels(ids):
    """Rótulos textuais (R1, R2...), para o Plotly não tratar IDs como eixo contínuo."""
    return [str(x) if str(x).upper().startswith("R") else f"R{x}" for x in ids]


def paginate(n_items, per_page):
    """Divide 0..n_items-1 em páginas. Retorna lista de (inicio, fim_exclusivo)."""
    per_page = max(1, int(per_page))
    return [(s, min(s + per_page, n_items)) for s in range(0, n_items, per_page)]


def page_label(start, end):
    return f"Posições {start + 1} a {end}"


def class_codes(categories):
    return np.array([CLASS_CODE.get(str(c), np.nan) for c in np.asarray(categories).ravel()],
                    dtype=float).reshape(np.asarray(categories).shape)


# ---------------------------------------------------------------------------
# Diferenças especialista x grupo
# ---------------------------------------------------------------------------

def rank_class_differences(result):
    """Diferenças (especialista − grupo) de posição e de classe."""
    rank_exp = np.asarray(result["rank_experts"], dtype=float)
    rank_grp = np.asarray(result["rank_siprem"], dtype=float)
    delta_rank = rank_exp - rank_grp[:, None]

    code_exp = class_codes(result["categories_experts"])
    code_grp = class_codes(result["categories"])
    delta_class = code_exp - code_grp[:, None]

    return {
        "delta_rank": delta_rank,
        "abs_delta_rank": np.abs(delta_rank),
        "delta_class": delta_class,
        "abs_delta_class": np.abs(delta_class),
    }


def agreement_summary(result, near=3):
    """Tabela de aderência de cada especialista ao ranking/classificação do grupo."""
    diff = rank_class_differences(result)
    names = result["expert_names"]
    rows = []
    for e, name in enumerate(names):
        adr = diff["abs_delta_rank"][:, e]
        adc = diff["abs_delta_class"][:, e]
        rho = spearmanr(result["rank_experts"][:, e], result["rank_siprem"]).statistic
        rows.append({
            "Especialista": name,
            "Spearman com o grupo": rho,
            "Média |Δ posição|": adr.mean(),
            "Mediana |Δ posição|": float(np.median(adr)),
            "Máx |Δ posição|": adr.max(),
            f"% a até {near} posições": 100 * np.mean(adr <= near),
            "Média |Δ classe|": adc.mean(),
            "% mesma classe": 100 * np.mean(adc == 0),
        })
    return pd.DataFrame(rows).sort_values("Spearman com o grupo", ascending=False).reset_index(drop=True)


def agreement_matrices(result):
    """Spearman, Kappa (classes) e Footrule normalizada entre fontes (especialistas + SIPREM)."""
    from sklearn.metrics import cohen_kappa_score

    labels = list(result["expert_names"]) + ["SIPREM"]
    ranks = np.column_stack([result["rank_experts"], result["rank_siprem"]])
    cats = np.column_stack([result["categories_experts"], result["categories"]]).astype(str)

    n_req, m = ranks.shape
    spear = np.eye(m)
    kappa = np.eye(m)
    foot = np.zeros((m, m))
    max_foot = (n_req ** 2) // 2

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            spear[i, j] = spearmanr(ranks[:, i], ranks[:, j]).statistic
            foot[i, j] = np.abs(ranks[:, i] - ranks[:, j]).sum() / max_foot
            try:
                kappa[i, j] = cohen_kappa_score(cats[:, i], cats[:, j])
            except Exception:
                kappa[i, j] = np.nan

    mk = lambda M: pd.DataFrame(M, index=labels, columns=labels)
    return {"spearman": mk(spear), "kappa": mk(kappa), "footrule": mk(foot)}


# ---------------------------------------------------------------------------
# Explicação por requisito
# ---------------------------------------------------------------------------

def criterion_contributions(result, weights_crisp):
    """Contribuição aproximada de cada critério ao fluxo líquido.

    contribuição_k = peso defuzzificado do critério k × fluxo líquido unicritério k
    (média dos especialistas). É uma aproximação para interpretação: o Φ fuzzy do
    SiPREM usa pesos fuzzy por especialista e multiplicação de Geldermann.
    """
    w = np.asarray(weights_crisp, dtype=float).reshape(1, -1)
    contrib = result["unicriterion_net"] * w
    return pd.DataFrame(contrib, columns=CRITERIA)


# ---------------------------------------------------------------------------
# Incerteza da classificação
# ---------------------------------------------------------------------------

def class_boundaries(result):
    """Fronteiras entre Grupo 1|2|3 (pontos médios entre centroides do K-means 1D)."""
    c = np.sort(np.asarray(result["kmeans_centroids"], dtype=float))[::-1]
    if c.size < 2:
        return np.array([]), c
    return (c[:-1] + c[1:]) / 2, c


def class_uncertainty(result, margin_pct=0.10):
    """Margem de cada requisito até a fronteira de classe mais próxima.

    margem_relativa = distância / (maior centroide − menor centroide).
    'Classe_incerta' = margem_relativa < margin_pct (requisito perto da fronteira).
    Requisitos Key (definidos por maioria de impacto máximo) não têm margem.
    """
    n = len(result["phi_net"])
    phi = np.asarray(result["phi_net"], dtype=float)
    bounds, cents = class_boundaries(result)

    margin = np.full(n, np.nan)
    rel = np.full(n, np.nan)
    is_key = np.asarray(result["categories"]) == "Key"

    if bounds.size:
        span = float(cents[0] - cents[-1]) or 1.0
        dist = np.abs(phi[:, None] - bounds[None, :])
        margin = dist.min(axis=1)
        rel = margin / span

    margin[is_key] = np.nan
    rel[is_key] = np.nan
    uncertain = (rel < margin_pct) & ~is_key

    return pd.DataFrame({
        "Margem_fronteira": margin,
        "Margem_relativa": rel,
        "Classe_incerta": np.where(np.isnan(rel), False, uncertain),
    })


def divergence_text(result, ids):
    """Texto 'E1=7 | E2=3 | E3=5' com o impacto dado por cada especialista."""
    imp = result["impact_matrix"]
    names = result["expert_names"]
    return [" | ".join(f"{n}={int(v)}" for n, v in zip(names, row)) for row in imp]


# ---------------------------------------------------------------------------
# Tabelas prontas para a interface/relatório
# ---------------------------------------------------------------------------

def build_ranking_table(data, result, margin_pct=0.10):
    unc = class_uncertainty(result, margin_pct)
    df = pd.DataFrame({
        "Rank": result["rank_siprem"].astype(int),
        "Requisito": data["ids"],
        "Descricao": data["descriptions"],
        "Categoria": result["categories"],
        "Phi_Liquido": result["phi_net"],
        "Phi_Plus": result["phi_plus"],
        "Phi_Minus": result["phi_minus"],
        "Classe_incerta": unc["Classe_incerta"].values,
        "Divergencia_Impacto": np.asarray(result["divergence_alert"], dtype=bool),
        "Impacto_por_especialista": divergence_text(result, data["ids"]),
    })
    return df.sort_values(["Rank", "Requisito"], key=lambda s: s.astype(str) if s.name == "Requisito" else s).reset_index(drop=True)


def relations_dataframe(data, result):
    rel = promethee_i_relations(result["phi_plus"], result["phi_minus"])
    labels = req_labels(data["ids"])
    return pd.DataFrame(rel, index=labels, columns=labels)
