"""Exportações do SiPREM: Excel completo e relatório HTML (sem Streamlit/Plotly)."""

from __future__ import annotations

import io
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd

from .data_loader import CRITERIA, CRIT_DIR, PREF_TYPES
from .fuzzy_numbers import defuzz_coa
from .analysis import (
    agreement_matrices, agreement_summary, build_ranking_table,
    criterion_contributions, relations_dataframe, req_labels,
)

PREF_NAMES = {1: "Usual", 2: "U-shape", 3: "V-shape", 4: "Level", 5: "Linear", 6: "Gaussiana"}


def config_table(result):
    return pd.DataFrame({
        "Criterio": CRITERIA,
        "Direcao": ["Maximizar" if d == 1 else "Minimizar" for d in CRIT_DIR],
        "Tipo_Preferencia": [PREF_NAMES[int(t)] for t in PREF_TYPES],
        "q": result["q_vals"],
        "p": result["p_vals"],
    })


def weights_table(fahp, expert_names=None):
    wf = np.asarray(fahp["w_fuzzy_group"], dtype=float)
    df = pd.DataFrame({
        "Criterio": CRITERIA,
        "w_a": wf[:, 0], "w_b": wf[:, 1], "w_c": wf[:, 2], "w_d": wf[:, 3],
        "Peso_defuzzificado_grupo": np.asarray(fahp["w_crisp_group"], dtype=float),
    })
    names = fahp.get("sheets") or [f"E{i + 1}" for i in range(fahp["w_crisp_individual"].shape[1])]
    for e, name in enumerate(names):
        df[f"Peso_{name}"] = fahp["w_crisp_individual"][:, e]
    return df


def build_excel(data, result, fahp, margin_pct=0.10):
    ids = data["ids"]
    names = result["expert_names"]

    ranking = build_ranking_table(data, result, margin_pct)

    phi_exp = pd.DataFrame(result["phi_net_experts"], columns=[f"Phi_{n}" for n in names])
    rank_exp = pd.DataFrame(result["rank_experts"].astype(int), columns=[f"Rank_{n}" for n in names])
    por_esp = pd.concat([pd.DataFrame({"Requisito": ids}), phi_exp, rank_exp], axis=1)

    impact = pd.DataFrame(result["impact_matrix"].astype(int), columns=list(names))
    impact.insert(0, "Requisito", ids)

    classes = pd.DataFrame(result["categories_experts"], columns=list(names))
    classes.insert(0, "Requisito", ids)
    classes["SIPREM"] = result["categories"]

    contrib = criterion_contributions(result, fahp["w_crisp_group"])
    contrib.insert(0, "Requisito", ids)

    mats = agreement_matrices(result)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="Ranking", index=False)
        por_esp.to_excel(writer, sheet_name="Phi_Rank_por_Especialista", index=False)
        impact.to_excel(writer, sheet_name="Impacto", index=False)
        classes.to_excel(writer, sheet_name="Classes_por_Especialista", index=False)
        contrib.to_excel(writer, sheet_name="Contribuicao_Criterios", index=False)
        weights_table(fahp).to_excel(writer, sheet_name="Pesos_FAHP", index=False)
        agreement_summary(result).to_excel(writer, sheet_name="Aderencia_Especialistas", index=False)
        mats["kappa"].to_excel(writer, sheet_name="Kappa_Classes")
        mats["spearman"].to_excel(writer, sheet_name="Spearman_Ranking")
        relations_dataframe(data, result).to_excel(writer, sheet_name="Relacoes_PROMETHEE_I")
        config_table(result).to_excel(writer, sheet_name="Configuracao", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
    output.seek(0)
    return output.getvalue()


def build_html_report(data, result, fahp, margin_pct=0.10, top_n=20):
    ranking = build_ranking_table(data, result, margin_pct)
    cls_counts = ranking["Categoria"].value_counts().reindex(["Key", "Grupo 1", "Grupo 2", "Grupo 3"]).fillna(0).astype(int)

    fmt = lambda df: df.to_html(index=False, border=0, classes="t", float_format=lambda x: f"{x:.4f}")
    cols = ["Rank", "Requisito", "Descricao", "Categoria", "Phi_Liquido"]

    top = ranking.head(top_n)[cols]
    unc = ranking[ranking["Classe_incerta"]].sort_values("Rank").head(top_n)[cols]
    div = ranking[ranking["Divergencia_Impacto"]].head(top_n)[cols + ["Impacto_por_especialista"]]

    w = weights_table(fahp)[["Criterio", "w_a", "w_b", "w_c", "w_d", "Peso_defuzzificado_grupo"]]
    conf = config_table(result)
    adh = agreement_summary(result)

    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#222;max-width:1100px}"
           "h1{margin-bottom:0}h2{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:32px}"
           ".t{border-collapse:collapse;width:100%;font-size:13px}.t th{background:#f0f2f6;text-align:left}"
           ".t td,.t th{padding:5px 8px;border-bottom:1px solid #e6e6e6}.small{color:#666;font-size:12px}")

    parts = [
        f"<html><head><meta charset='utf-8'><title>Relatório SiPREM</title><style>{css}</style></head><body>",
        "<h1>Relatório SiPREM</h1>",
        f"<p class='small'>Gerado em {datetime.now():%d/%m/%Y %H:%M} — "
        f"{data['n_requirements']} requisitos, {data['n_experts']} especialistas "
        f"({escape(', '.join(data['sheets']))}).</p>",
        "<h2>Distribuição das classes</h2>",
        fmt(cls_counts.rename_axis("Classe").reset_index(name="Requisitos")),
        f"<h2>Top {top_n} do ranking</h2>", fmt(top),
        f"<h2>Requisitos com classe incerta (perto da fronteira, margem &lt; {margin_pct:.0%})</h2>",
        fmt(unc) if len(unc) else "<p>Nenhum.</p>",
        f"<h2>Divergência de impacto entre especialistas (diferença ≥ {result['alert_threshold']})</h2>",
        fmt(div) if len(div) else "<p>Nenhum.</p>",
        "<h2>Pesos dos critérios (FAHP-Express)</h2>", fmt(w),
        "<h2>Aderência dos especialistas ao grupo</h2>", fmt(adh),
        "<h2>Configuração do PROMETHEE</h2>", fmt(conf),
        "<p class='small'>SiPREM — FAHP-Express + Fuzzy PROMETHEE. Classes: Key por maioria de impacto máximo; "
        "demais por K-means (3 grupos) sobre o fluxo líquido.</p></body></html>",
    ]
    return "\n".join(parts)
