"""Figuras Plotly do SiPREM. Todas as figuras longas trabalham por janela (start, end)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.analysis import CLASS_ORDER, CLASS_SHORT, class_codes, rank_class_differences, req_labels
from core.data_loader import CRITERIA
from core.promethee import promethee_i_relations

ROW_PX = 26


def _height(n_rows, extra=190, minimum=420):
    return max(minimum, ROW_PX * n_rows + extra)


def _order(result):
    """Índices dos requisitos do melhor para o pior no ranking SIPREM (desempate pelo Φ)."""
    return np.lexsort((-np.asarray(result["phi_net"]), np.asarray(result["rank_siprem"])))


def plot_weights(fahp):
    names = fahp.get("sheets") or [f"E{i + 1}" for i in range(fahp["w_crisp_individual"].shape[1])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=CRITERIA, y=fahp["w_crisp_group"], name="Peso do grupo"))
    for e, name in enumerate(names):
        fig.add_trace(go.Scatter(x=CRITERIA, y=fahp["w_crisp_individual"][:, e],
                                 mode="markers", marker=dict(size=10), name=name))
    fig.update_layout(title="Pesos dos critérios (FAHP-Express)", xaxis_title="Critério",
                      yaxis_title="Peso", legend_title="Fonte", height=460)
    return fig


def plot_rank_heatmap(data, result, start, end):
    order = _order(result)[start:end]
    labels = np.array(req_labels(data["ids"]))[order]
    ranks = np.column_stack([result["rank_experts"], result["rank_siprem"]])[order]
    cols = list(result["expert_names"]) + ["SIPREM"]
    ylab = [f"{l} (#{int(r)})" for l, r in zip(labels, result["rank_siprem"][order])]
    n = len(result["rank_siprem"])

    fig = go.Figure(go.Heatmap(
        z=ranks, x=cols, y=ylab,
        text=np.vectorize(lambda v: f"{v:g}")(ranks), texttemplate="%{text}",
        colorscale="Blues_r", zmin=1, zmax=n, xgap=1, ygap=1,
        colorbar=dict(title="Posição"),
        hovertemplate="%{y}<br>%{x}: posição %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(
        title=f"Posições no ranking — especialistas vs SIPREM (posições {start + 1} a {end})",
        xaxis_title="Fonte", yaxis_title="Requisito (posição SIPREM)",
        height=_height(len(order)), margin=dict(l=90, r=40, t=70, b=60))
    return fig


def plot_class_heatmap(data, result, start, end):
    order = _order(result)[start:end]
    labels = np.array(req_labels(data["ids"]))[order]
    cats = np.column_stack([result["categories_experts"], result["categories"]])[order]
    cols = list(result["expert_names"]) + ["SIPREM"]
    z = class_codes(cats)
    text = np.vectorize(lambda c: CLASS_SHORT.get(str(c), str(c)))(cats)
    ylab = [f"{l} (#{int(r)})" for l, r in zip(labels, result["rank_siprem"][order])]

    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=ylab, text=text, texttemplate="%{text}",
        colorscale=[[0.00, "#3730a3"], [0.249, "#3730a3"], [0.250, "#2d9cdb"], [0.499, "#2d9cdb"],
                    [0.500, "#7ac943"], [0.749, "#7ac943"], [0.750, "#fff200"], [1.00, "#fff200"]],
        zmin=0, zmax=3, xgap=1, ygap=1,
        colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=["Key", "G1", "G2", "G3"], title="Classe"),
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(
        title=f"Classificações por fonte (posições {start + 1} a {end} do ranking SIPREM)",
        xaxis_title="Fonte", yaxis_title="Requisito (posição SIPREM)",
        height=_height(len(order)), margin=dict(l=90, r=40, t=70, b=60))
    return fig


def plot_relation_window(data, result, start, end):
    """Bloco da matriz do PROMETHEE I (requisitos start..end contra eles mesmos)."""
    order = _order(result)[start:end]
    rel_full = promethee_i_relations(result["phi_plus"], result["phi_minus"])
    rel = rel_full[np.ix_(order, order)]
    labels = np.array(req_labels(data["ids"]))[order]
    mapping = {"P-": 0, "I": 1, "P": 2, "R": 3}
    z = np.vectorize(mapping.get)(rel)

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=rel, texttemplate="%{text}",
        textfont=dict(size=9 if len(order) <= 40 else 7),
        colorscale=[[0.00, "#3333cc"], [0.249, "#3333cc"], [0.250, "#aaaaaa"], [0.499, "#aaaaaa"],
                    [0.500, "#22aa22"], [0.749, "#22aa22"], [0.750, "#e8cc28"], [1.00, "#e8cc28"]],
        zmin=0, zmax=3, xgap=1, ygap=1,
        colorbar=dict(tickvals=[0, 1, 2, 3],
                      ticktext=["P− (dominado)", "I (indiferente)", "P (domina)", "R (incomparável)"],
                      title="Relação"),
        hovertemplate="i=%{y}<br>j=%{x}<br>%{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", tickangle=90)
    fig.update_layout(
        title=f"Fuzzy PROMETHEE I — posições {start + 1} a {end} (linha i em relação à coluna j)",
        xaxis_title="Requisito j", yaxis_title="Requisito i",
        height=_height(len(order), extra=230, minimum=520), margin=dict(l=80, r=60, t=70, b=90))
    return fig


def relation_counts(result):
    rel = promethee_i_relations(result["phi_plus"], result["phi_minus"])
    iu = np.triu_indices(rel.shape[0], k=1)
    vals = rel[iu]
    n = max(len(vals), 1)
    return {
        "Comparáveis (um domina o outro)": 100 * np.mean((vals == "P") | (vals == "P-")),
        "Indiferentes": 100 * np.mean(vals == "I"),
        "Incomparáveis": 100 * np.mean(vals == "R"),
    }


def plot_gaia(data, result):
    X = np.asarray(result["unicriterion_net"], dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    total = np.sum(S ** 2)
    explained = (S[:2] ** 2) / total * 100 if total > 0 else np.zeros(2)
    load = Vt[:2, :].T

    span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1e-9)
    arrow = span * 0.22
    labels = req_labels(data["ids"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", text=labels,
        customdata=np.column_stack([np.asarray(result["phi_net"], dtype=float),
                                    np.asarray(result["rank_siprem"], dtype=float)]),
        marker=dict(size=8, color=result["phi_net"], colorscale="Turbo", showscale=True,
                    colorbar=dict(title="Φ"), line=dict(width=0.4, color="#444")),
        hovertemplate="<b>%{text}</b><br>Φ líquido: %{customdata[0]:.4f}<br>"
                      "Rank: %{customdata[1]:.0f}<extra></extra>",
        name="Requisitos"))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    short = {"CAPACIDADE NACIONAL DE IMPLEMENTAÇÃO": "Capacidade nacional", "RiscoVerif": "Risco de verificação"}
    for k, crit in enumerate(CRITERIA):
        vx, vy = float(load[k, 0] * arrow), float(load[k, 1] * arrow)
        name = short.get(crit, crit)
        fig.add_trace(go.Scatter(x=[0, vx], y=[0, vy], mode="lines", name=name,
                                 line=dict(color=colors[k % 5], width=3)))
        fig.add_annotation(x=vx, y=vy, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowwidth=2.6, arrowcolor=colors[k % 5], text="")

    fig.add_hline(y=0, line_width=0.5, line_color="#cccccc")
    fig.add_vline(x=0, line_width=0.5, line_color="#cccccc")
    fig.update_layout(
        title=f"GAIA — fluxos líquidos unicritério (informação preservada: {explained.sum():.1f}%)",
        xaxis_title=f"Eixo 1 ({explained[0]:.1f}%)", yaxis_title=f"Eixo 2 ({explained[1]:.1f}%)",
        height=680, hovermode="closest", legend_title="Critérios")
    return fig


def plot_delta_histograms(result):
    diff = rank_class_differences(result)
    names = result["expert_names"]
    rows_r, rows_c = [], []
    for e, name in enumerate(names):
        for v in diff["abs_delta_rank"][:, e]:
            rows_r.append({"Especialista": name, "|Δ posição|": v})
        for v in diff["abs_delta_class"][:, e]:
            rows_c.append({"Especialista": name, "|Δ classe|": str(int(v))})
    df_r, df_c = pd.DataFrame(rows_r), pd.DataFrame(rows_c)

    n = len(result["rank_siprem"])
    fig_r = px.histogram(df_r, x="|Δ posição|", facet_col="Especialista", facet_col_wrap=3,
                         nbins=min(40, max(10, n // 5)),
                         title="Diferença absoluta de posição: especialista vs grupo SIPREM")
    fig_r.update_layout(height=340 * int(np.ceil(len(names) / 3)), showlegend=False,
                        yaxis_title="Requisitos")

    fig_c = px.histogram(df_c, x="|Δ classe|", facet_col="Especialista", facet_col_wrap=3,
                         category_orders={"|Δ classe|": ["0", "1", "2", "3"]},
                         title="Diferença absoluta de classe (Key, G1, G2, G3): especialista vs grupo SIPREM")
    fig_c.update_layout(height=340 * int(np.ceil(len(names) / 3)), showlegend=False,
                        yaxis_title="Requisitos")
    return fig_r, fig_c


def plot_agreement_heatmap(matrix, title, zmin, zmax, colorscale="RdBu"):
    z = matrix.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(matrix.columns), y=list(matrix.index),
        text=np.vectorize(lambda v: f"{v:.2f}")(z), texttemplate="%{text}",
        zmin=zmin, zmax=zmax, colorscale=colorscale, xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title, height=430)
    return fig


def plot_contribution(contrib_row, title):
    vals = np.asarray(contrib_row, dtype=float)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=CRITERIA, orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title, xaxis_title="Contribuição aproximada ao Φ líquido",
                      height=330, margin=dict(l=40, r=20, t=60, b=40))
    return fig
