from pathlib import Path
import hashlib
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.data_loader import (
    CRITERIA, PREF_TYPES, InputValidationError,
    read_expert_workbook, read_fahp_workbook, align_fahp_to_data,
)
from core.siprem_core import run_siprem, default_thresholds
from core.analysis import (
    agreement_matrices, agreement_summary, build_ranking_table,
    criterion_contributions, paginate, page_label, req_labels,
)
from core.export import build_excel, build_html_report, config_table, weights_table
import plots

st.set_page_config(page_title="SiPREM — Priorização de Requisitos", page_icon="🎯", layout="wide")


def _stretch():
    """Compatível com versões novas (width='stretch') e antigas (use_container_width)."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        if (major, minor) >= (1, 50):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


STRETCH = _stretch()


# ---------------------------------------------------------------- cálculo (com cache)
@st.cache_data(show_spinner=False, max_entries=8)
def run_pipeline(req_bytes, fahp_bytes, q_vals, p_vals, alert_threshold, seed):
    with tempfile.TemporaryDirectory() as tmp:
        req_path = Path(tmp) / "avaliacoes.xlsx"
        fahp_path = Path(tmp) / "fahp.xlsx"
        req_path.write_bytes(req_bytes)
        fahp_path.write_bytes(fahp_bytes)
        data = read_expert_workbook(req_path)
        fahp = read_fahp_workbook(fahp_path)

    fahp, warnings = align_fahp_to_data(data, fahp)
    result = run_siprem(
        data["evaluations"], fahp["w_fuzzy_individual"], random_state=int(seed),
        q_vals=list(q_vals), p_vals=list(p_vals), alert_threshold=int(alert_threshold),
    )
    return data, fahp, result, warnings


@st.cache_data(show_spinner=False, max_entries=8)
def cached_excel(_data, _result, _fahp, margin_pct, key):
    return build_excel(_data, _result, _fahp, margin_pct)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_report(_data, _result, _fahp, margin_pct, key):
    return build_html_report(_data, _result, _fahp, margin_pct)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_agreement(_result, key):
    return agreement_matrices(_result), agreement_summary(_result)


# ---------------------------------------------------------------- barra lateral
st.title("SiPREM")
st.subheader("Sistema de Priorização de Requisitos de Equipamentos Militares")
st.caption("FAHP-Express + Fuzzy PROMETHEE — núcleo matemático portado do MATLAB para Python")

examples_dir = ROOT / "exemplos"
req_example = examples_dir / "Exemplo_Avaliacoes_Especialistas_SiPREM.xlsx"
fahp_example = examples_dir / "Exemplo_Pesos_FAHP_SiPREM.xlsx"
guide_path = ROOT / "Guia_de_Usuario_SiPREM.pdf"

d_q, d_p = default_thresholds()

with st.sidebar:
    st.header("1. Entradas")
    use_example = False
    if req_example.exists() and fahp_example.exists():
        use_example = st.checkbox("Usar planilhas de exemplo", value=False,
                                  help="Carrega um exemplo pequeno para conhecer o sistema.")
    if use_example:
        requisitos_file = fahp_file = None
        st.caption("Exemplo carregado. Clique em Executar.")
    else:
        requisitos_file = st.file_uploader("Planilha de avaliações dos especialistas", type=["xlsx"],
                                           help="Abas E1, E2, … (ou 'Especialista 1', …) com ID, descrição e 5 critérios.")
        fahp_file = st.file_uploader("Planilha FAHP", type=["xlsx"],
                                     help="Mesmas abas de especialistas; linha 4 (B4:F4) com os julgamentos de referência.")

    with st.expander("⚙️ Parâmetros avançados"):
        st.caption("Os padrões reproduzem o MATLAB. Altere só se souber o efeito.")
        q_list, p_list = list(d_q), list(d_p)
        for k, crit in enumerate(CRITERIA):
            if int(PREF_TYPES[k]) in (4, 5):
                c1, c2 = st.columns(2)
                q_list[k] = c1.number_input(f"q — {crit[:18]}", value=float(d_q[k]), step=0.1, min_value=0.0, key=f"q{k}")
                p_list[k] = c2.number_input(f"p — {crit[:18]}", value=float(d_p[k]), step=0.1, min_value=0.0, key=f"p{k}")
        alert_threshold = st.slider("Divergência de impacto (diferença mínima)", 1, 6, 3,
                                    help="Sinaliza requisitos em que especialistas diferem em pelo menos este valor no Impacto.")
        seed = st.number_input("Semente do K-means", value=0, step=1)

    st.header("2. Execução")
    executar = st.button("▶ Executar SiPREM", type="primary", **STRETCH)

    st.header("3. Exibição")
    per_page = st.select_slider("Requisitos por página (gráficos)", options=[20, 30, 40, 50, 60, 100], value=40)
    margin_pct = st.slider("Classe incerta: margem até a fronteira", 0.02, 0.30, 0.10, 0.01,
                           help="Requisito a menos desta fração da distância entre centroides de uma fronteira de classe.")

    st.divider()
    st.header("Guia e exemplos")
    if guide_path.exists():
        st.download_button("📘 Guia do usuário (PDF)", data=guide_path.read_bytes(), file_name=guide_path.name,
                           mime="application/pdf", **STRETCH)
    if req_example.exists():
        st.download_button("📊 Exemplo: avaliações", data=req_example.read_bytes(), file_name=req_example.name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **STRETCH)
    if fahp_example.exists():
        st.download_button("⚖️ Exemplo: pesos FAHP", data=fahp_example.read_bytes(), file_name=fahp_example.name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **STRETCH)

# ---------------------------------------------------------------- execução
current_params = (tuple(q_list), tuple(p_list), int(alert_threshold), int(seed))

if executar:
    if use_example:
        req_bytes, fahp_bytes = req_example.read_bytes(), fahp_example.read_bytes()
        source = "planilhas de exemplo"
    elif requisitos_file is None or fahp_file is None:
        st.error("Selecione as duas planilhas (ou marque “Usar planilhas de exemplo”) antes de executar.")
        st.stop()
    else:
        req_bytes, fahp_bytes = requisitos_file.getvalue(), fahp_file.getvalue()
        source = f"{requisitos_file.name} + {fahp_file.name}"
    st.session_state["inputs"] = dict(req=req_bytes, fahp=fahp_bytes, params=current_params, source=source)

inputs = st.session_state.get("inputs")

if inputs is None:
    st.info("Envie as duas planilhas na barra lateral (ou use o exemplo) e clique em **Executar SiPREM**.")
    st.stop()

try:
    with st.spinner("Executando FAHP + Fuzzy PROMETHEE + classificação..."):
        qv, pv, thr, sd = inputs["params"]
        data, fahp, result, warnings = run_pipeline(inputs["req"], inputs["fahp"], qv, pv, thr, sd)
except InputValidationError as exc:
    st.error("Encontrei problemas nas planilhas. Corrija e execute novamente:")
    st.markdown("\n".join(f"- {m}" for m in exc.messages))
    st.session_state.pop("inputs", None)
    st.stop()
except Exception as exc:
    st.error("Não foi possível executar o SiPREM com os arquivos enviados.")
    with st.expander("Detalhes técnicos"):
        st.exception(exc)
    st.session_state.pop("inputs", None)
    st.stop()

if inputs["params"] != current_params:
    st.warning("Os parâmetros avançados mudaram desde a última execução. Clique em **Executar SiPREM** para aplicá-los.")
for w in warnings:
    st.warning(w)
if data.get("used_sheet_fallback") or fahp.get("used_sheet_fallback"):
    st.warning("Nenhuma aba seguiu o padrão E1, E2… / Especialista N; usei todas as abas, exceto a última, "
               f"como especialistas ({', '.join(data['sheets'])}).")

# ---------------------------------------------------------------- resultado
ranking = build_ranking_table(data, result, margin_pct)
n_req = data["n_requirements"]
cache_key = (hashlib.md5(inputs["req"] + inputs["fahp"]).hexdigest(), inputs["params"])

st.divider()
st.header("Resultado")
st.caption(f"Entrada: {inputs['source']}")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Especialistas", data["n_experts"])
c2.metric("Requisitos", n_req)
c3.metric("Key", int((ranking["Categoria"] == "Key").sum()))
c4.metric("Grupo 1", int((ranking["Categoria"] == "Grupo 1").sum()))
c5.metric("Classe incerta", int(ranking["Classe_incerta"].sum()))
c6.metric("Divergência de impacto", int(ranking["Divergencia_Impacto"].sum()))

tab_rank, tab_graf, tab_exp, tab_in = st.tabs(["🏆 Ranking", "📈 Gráficos", "👥 Especialistas", "ℹ️ Entradas"])

# ================================================================ Ranking
with tab_rank:
    f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
    busca = f1.text_input("Buscar (ID ou texto da descrição)")
    cats_sel = f2.multiselect("Classe", ["Key", "Grupo 1", "Grupo 2", "Grupo 3"],
                              default=["Key", "Grupo 1", "Grupo 2", "Grupo 3"])
    so_incerta = f3.checkbox("Só classe incerta")
    so_div = f4.checkbox("Só com divergência")

    view = ranking[ranking["Categoria"].isin(cats_sel)]
    if busca.strip():
        b = busca.strip().lower()
        mask = (view["Requisito"].astype(str).str.lower().str.contains(b, regex=False)
                | view["Descricao"].astype(str).str.lower().str.contains(b, regex=False))
        view = view[mask]
    if so_incerta:
        view = view[view["Classe_incerta"]]
    if so_div:
        view = view[view["Divergencia_Impacto"]]

    st.caption(f"{len(view)} de {n_req} requisitos. Clique no cabeçalho de uma coluna para ordenar.")
    st.dataframe(view, hide_index=True, height=520, column_config={
        "Descricao": st.column_config.TextColumn("Descrição", width="large"),
        "Phi_Liquido": st.column_config.NumberColumn("Φ líquido", format="%.4f"),
        "Phi_Plus": st.column_config.NumberColumn("Φ+", format="%.4f"),
        "Phi_Minus": st.column_config.NumberColumn("Φ−", format="%.4f"),
        "Classe_incerta": st.column_config.CheckboxColumn("Classe incerta"),
        "Divergencia_Impacto": st.column_config.CheckboxColumn("Divergência de impacto"),
        "Impacto_por_especialista": st.column_config.TextColumn("Impacto por especialista"),
    }, **STRETCH)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Excel completo", data=cached_excel(data, result, fahp, margin_pct, cache_key),
                       file_name="Resultado_SiPREM.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **STRETCH)
    d2.download_button("⬇️ Relatório (HTML)", data=cached_report(data, result, fahp, margin_pct, cache_key),
                       file_name="Relatorio_SiPREM.html", mime="text/html", **STRETCH)
    d3.download_button("⬇️ Ranking filtrado (CSV)", data=view.to_csv(index=False).encode("utf-8-sig"),
                       file_name="Ranking_SiPREM.csv", mime="text/csv", **STRETCH)

    st.subheader("Por que este requisito ficou nesta posição?")
    if len(view) == 0:
        st.info("Nenhum requisito com os filtros atuais.")
    else:
        opts = view.index.tolist()
        fmt = lambda i: f"#{int(view.loc[i, 'Rank'])} — {req_labels([view.loc[i, 'Requisito']])[0]} — {str(view.loc[i, 'Descricao'])[:70]}"
        sel = st.selectbox("Escolha um requisito", opts, format_func=fmt)
        row = view.loc[sel]
        j = data["ids"].index(row["Requisito"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Posição", int(row["Rank"]))
        m2.metric("Classe", row["Categoria"])
        m3.metric("Φ líquido", f"{row['Phi_Liquido']:.4f}")
        m4.metric("Classe incerta", "Sim" if row["Classe_incerta"] else "Não")
        st.write(str(row["Descricao"]))

        contrib = criterion_contributions(result, fahp["w_crisp_group"])
        left, right = st.columns([3, 2])
        left.plotly_chart(plots.plot_contribution(
            contrib.iloc[j].values, "Contribuição de cada critério (peso × fluxo unicritério)"), **STRETCH)
        left.caption("Aproximação para interpretação: positivo favorece o requisito, negativo o penaliza.")
        by_exp = pd.DataFrame({
            "Especialista": result["expert_names"],
            "Impacto": result["impact_matrix"][j].astype(int),
            "Posição": result["rank_experts"][j].astype(int),
            "Classe": result["categories_experts"][j],
            "Φ líquido": result["phi_net_experts"][j],
        })
        right.markdown("**Visão de cada especialista**")
        right.dataframe(by_exp, hide_index=True, **STRETCH)

# ================================================================ Gráficos
with tab_graf:
    pages = paginate(n_req, per_page)
    st.subheader("Pesos dos critérios — FAHP-Express")
    st.plotly_chart(plots.plot_weights(fahp), **STRETCH)
    st.caption("Barras = peso do grupo; pontos = pesos individuais dos especialistas.")

    st.subheader("Mapas por faixa do ranking")
    pg = st.selectbox("Faixa de requisitos (ordem do ranking SIPREM)", range(len(pages)),
                      format_func=lambda i: page_label(*pages[i]), key="page_sel")
    s0, e0 = pages[pg]

    st.markdown("**Posições no ranking**")
    st.plotly_chart(plots.plot_rank_heatmap(data, result, s0, e0), **STRETCH)
    st.markdown("**Classificações por fonte**")
    st.plotly_chart(plots.plot_class_heatmap(data, result, s0, e0), **STRETCH)

    st.markdown("**Fuzzy PROMETHEE I**")
    rc = plots.relation_counts(result)
    cc = st.columns(len(rc))
    for col, (name, val) in zip(cc, rc.items()):
        col.metric(name, f"{val:.1f}% dos pares")
    st.plotly_chart(plots.plot_relation_window(data, result, s0, e0), **STRETCH)
    st.caption("P = domina; P− = dominado; I = indiferente; R = incomparável. A matriz completa está no Excel.")

    st.subheader("GAIA — fluxos líquidos unicritério")
    st.plotly_chart(plots.plot_gaia(data, result), **STRETCH)

# ================================================================ Especialistas
with tab_exp:
    mats, summary = cached_agreement(result, cache_key)

    st.subheader("Aderência de cada especialista ao grupo")
    st.dataframe(summary, hide_index=True, column_config={
        "Spearman com o grupo": st.column_config.NumberColumn(format="%.3f"),
        "Média |Δ posição|": st.column_config.NumberColumn(format="%.1f"),
        "Mediana |Δ posição|": st.column_config.NumberColumn(format="%.1f"),
        "Máx |Δ posição|": st.column_config.NumberColumn(format="%.0f"),
        "% mesma classe": st.column_config.NumberColumn(format="%.1f"),
    }, **STRETCH)

    st.subheader("Distribuição das diferenças em relação ao grupo")
    fig_r, fig_c = plots.plot_delta_histograms(result)
    st.plotly_chart(fig_r, **STRETCH)
    st.plotly_chart(fig_c, **STRETCH)

    st.subheader("Concordância entre fontes")
    k1, k2 = st.columns(2)
    k1.plotly_chart(plots.plot_agreement_heatmap(mats["kappa"], "Kappa de Cohen (classes)", -1, 1), **STRETCH)
    k2.plotly_chart(plots.plot_agreement_heatmap(mats["spearman"], "Spearman (posições)", -1, 1), **STRETCH)

    st.subheader("Dados por especialista")
    labels = req_labels(data["ids"])
    phi_exp = pd.DataFrame(result["phi_net_experts"], index=labels, columns=result["expert_names"])
    phi_exp.index.name = "Requisito"
    impact = pd.DataFrame(result["impact_matrix"].astype(int), index=labels, columns=result["expert_names"])
    impact.index.name = "Requisito"
    t1, t2 = st.columns(2)
    t1.markdown("**Φ líquido individual**")
    t1.dataframe(phi_exp, height=380, **STRETCH)
    t2.markdown("**Impacto informado**")
    t2.dataframe(impact, height=380, **STRETCH)

# ================================================================ Entradas
with tab_in:
    st.write("**Abas de especialistas (avaliações):**", ", ".join(data["sheets"]))
    if data.get("ignored_sheets"):
        st.write("**Abas ignoradas (avaliações):**", ", ".join(map(str, data["ignored_sheets"])))
    st.write("**Abas de especialistas (FAHP):**", ", ".join(fahp["sheets"]))
    if fahp.get("ignored_sheets"):
        st.write("**Abas ignoradas (FAHP):**", ", ".join(map(str, fahp["ignored_sheets"])))
    st.markdown("**Pesos dos critérios**")
    st.dataframe(weights_table(fahp), hide_index=True, **STRETCH)
    st.markdown("**Configuração do PROMETHEE**")
    st.dataframe(config_table(result), hide_index=True, **STRETCH)

st.divider()
st.caption("SiPREM — interface web. A lógica de cálculo fica no módulo core/.")
