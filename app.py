from pathlib import Path
import io
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.data_loader import read_expert_workbook, read_fahp_workbook, CRITERIA, CRIT_DIR, PREF_TYPES, SCALE_MAX
from core.siprem_core import run_siprem, build_fuzzy_matrix
from core.promethee import pref_trap
from core.fuzzy_numbers import trap_add, trap_sub, trap_mul_geldermann, trap_scalar_div, defuzz_coa

st.set_page_config(
    page_title="SiPREM — Priorização de Requisitos",
    page_icon="🎯",
    layout="wide",
)


def save_uploaded(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".xlsx"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return Path(tmp.name)


def build_result_dataframe(data, result):
    return pd.DataFrame({
        "Rank": result["rank_siprem"].astype(int),
        "Requisito": data["ids"],
        "Descricao": data["descriptions"],
        "Phi_Plus": result["phi_plus"],
        "Phi_Minus": result["phi_minus"],
        "Phi_Liquido": result["phi_net"],
        "Categoria": result["categories"],
        "Alerta_Divergencia_Impacto": (
            result["impact_matrix"].max(axis=1)
            - result["impact_matrix"].min(axis=1) >= 3
        ),
    }).sort_values(["Rank", "Requisito"]).reset_index(drop=True)


def build_excel(data, result):
    ranking = build_result_dataframe(data, result)
    phi_exp = pd.DataFrame(result["phi_net_experts"], index=data["ids"], columns=result["expert_names"])
    phi_exp.insert(0, "Requisito", phi_exp.index)
    impact = pd.DataFrame(result["impact_matrix"], index=data["ids"], columns=result["expert_names"])
    impact.insert(0, "Requisito", impact.index)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="Ranking", index=False)
        phi_exp.to_excel(writer, sheet_name="Phi_por_Especialista", index=False)
        impact.to_excel(writer, sheet_name="Impacto", index=False)
        pd.DataFrame({
            "Criterio": CRITERIA,
            "Direcao": ["Maximizar", "Minimizar", "Maximizar", "Minimizar", "Minimizar"],
            "Tipo_Preferencia": ["Usual", "Level", "Level", "Level", "Level"],
        }).to_excel(writer, sheet_name="Configuracao", index=False)
    output.seek(0)
    return output.getvalue()


def aggregate_weights(fahp):
    """Defuzzifica os pesos individuais e retorna média + distribuição por especialista."""
    wf = np.asarray(fahp["w_fuzzy_individual"], dtype=float)  # crit x 4 x exp
    crisp = np.array([[defuzz_coa(wf[k, :, e]) for e in range(wf.shape[2])] for k in range(wf.shape[0])])
    return crisp.mean(axis=1), crisp


def preference_relation_matrix(result, ids):
    """PROMETHEE I: P+ (domina), P- (dominado), I e R (incomparável)."""
    plus = np.asarray(result["phi_plus"], dtype=float)
    minus = np.asarray(result["phi_minus"], dtype=float)
    n = len(ids)
    rel = np.empty((n, n), dtype=object)
    eps = 1e-10
    for i in range(n):
        for j in range(n):
            if i == j:
                rel[i, j] = "I"
                continue
            pplus = plus[i] > plus[j] + eps
            pminus = minus[i] < minus[j] - eps
            dominated_plus = plus[i] < plus[j] - eps
            dominated_minus = minus[i] > minus[j] + eps
            if pplus and pminus:
                rel[i, j] = "P"
            elif dominated_plus and dominated_minus:
                rel[i, j] = "P-"
            elif abs(plus[i] - plus[j]) <= eps and abs(minus[i] - minus[j]) <= eps:
                rel[i, j] = "I"
            else:
                rel[i, j] = "R"
    return rel


def plot_weights(fahp):
    mean_w, crisp = aggregate_weights(fahp)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=CRITERIA, y=mean_w, name="Peso médio"))
    for e, name in enumerate(fahp.get("sheets", [f"E{i+1}" for i in range(crisp.shape[1])])):
        fig.add_trace(go.Scatter(x=CRITERIA, y=crisp[:, e], mode="markers", name=name))
    fig.update_layout(
        title="Pesos fuzzy dos critérios obtidos pelo FAHP-Express",
        xaxis_title="Critério", yaxis_title="Peso", legend_title="Fonte",
        height=500,
    )
    return fig


def _req_labels(ids):
    """IDs como categorias textuais para impedir que o Plotly trate R1/R2 ou 1/2 como eixo contínuo."""
    return [str(x) if str(x).startswith("R") else f"R{x}" for x in ids]


def plot_rank_heatmap(data, result):
    ranks = np.asarray(result["rank_experts"], dtype=float)
    labels = _req_labels(data["ids"])
    df = pd.DataFrame(ranks, index=labels, columns=result["expert_names"] )
    df["SIPREM"] = np.asarray(result["rank_siprem"], dtype=float)
    df = df.sort_values("SIPREM", ascending=True)

    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=list(df.columns),
        y=list(df.index),
        text=np.vectorize(lambda x: f"{x:g}")(df.values),
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale="Blues_r",
        zmin=1, zmax=len(labels),
        xgap=1, ygap=1,
        colorbar=dict(title="Posição"),
        hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Heatmap das posições no ranking: especialistas vs grupo SIPREM",
        xaxis_title="Fonte de avaliação",
        yaxis_title="Requisitos",
        height=max(700, len(labels) * 24),
        margin=dict(l=70, r=40, t=70, b=70),
    )
    return fig


def plot_classification_heatmap(data, result):
    cats = np.asarray(result["categories"], dtype=object)
    experts = result["expert_names"]
    labels = _req_labels(data["ids"])
    n_req = len(labels)
    class_mat = np.empty((n_req, len(experts)), dtype=object)
    for e in range(len(experts)):
        phi_e = result["phi_net_experts"][:, e]
        impact_e = result["impact_matrix"][:, e]
        key_e = np.where(impact_e == SCALE_MAX[0])[0]
        from core.classification import classify_requirements_kmeans
        ce, _, _, _ = classify_requirements_kmeans(phi_e, key_e, random_state=0)
        class_mat[:, e] = ce

    class_df = pd.DataFrame(class_mat, index=labels, columns=experts)
    class_df["SIPREM"] = cats
    order = np.argsort(result["rank_siprem"])
    class_df = class_df.iloc[order]

    mapping = {"Key": 0, "Grupo 1": 1, "Grupo 2": 2, "Grupo 3": 3}
    z = class_df.replace(mapping).astype(float).values
    text = class_df.astype(str).values
    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(class_df.columns),
        y=list(class_df.index),
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=[
            [0.00, "#3730a3"], [0.249, "#3730a3"],
            [0.250, "#2d9cdb"], [0.499, "#2d9cdb"],
            [0.500, "#7ac943"], [0.749, "#7ac943"],
            [0.750, "#fff200"], [1.00, "#fff200"],
        ],
        zmin=0, zmax=3, xgap=1, ygap=1,
        colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=["Key", "G1", "G2", "G3"], title="Classificação"),
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Heatmap das classificações por fonte",
        xaxis_title="Fonte",
        yaxis_title="Requisitos",
        height=max(700, n_req * 24),
        margin=dict(l=70, r=40, t=70, b=70),
    )
    return fig

def plot_boxplot(data, result):
    phi = np.asarray(result["phi_net_experts"], dtype=float)
    rows = []
    for j, rid in enumerate(data["ids"]):
        for e, name in enumerate(result["expert_names"]):
            rows.append({"Requisito": rid, "Fluxo líquido": phi[j, e], "Fonte": name})
    df = pd.DataFrame(rows)
    fig = px.box(df, x="Requisito", y="Fluxo líquido", points=False, title="Boxplot dos fluxos líquidos individuais por requisito (SiPREM)")
    fig.add_trace(go.Scatter(
        x=data["ids"], y=result["phi_net"], mode="markers", marker=dict(symbol="star", size=11),
        name="Fluxo agregado SIPREM"
    ))
    fig.update_layout(height=550, xaxis_title="Requisitos", yaxis_title="Φ líquido")
    return fig


def compute_unicriterion_net(data, result):
    """Calcula fluxos líquidos por requisito e critério, agregados entre especialistas."""
    n_req = data["n_requirements"]
    n_crit = data["n_criteria"]
    n_exp = data["n_experts"]
    q = result["q_vals"]
    p = result["p_vals"]
    uni = np.zeros((n_req, n_crit))

    for k in range(n_crit):
        exp_values = np.zeros((n_req, n_exp))
        for e, name in enumerate(result["expert_names"]):
            F = build_fuzzy_matrix(data["evaluations"][name])
            for i in range(n_req):
                sum_p = np.zeros(4)
                sum_m = np.zeros(4)
                for j in range(n_req):
                    if i == j:
                        continue
                    if CRIT_DIR[k] == 1:
                        D_ij = np.array([F[i, k, 0]-F[j, k, 3], F[i, k, 1]-F[j, k, 2], F[i, k, 2]-F[j, k, 1], F[i, k, 3]-F[j, k, 0]])
                    else:
                        D_ij = np.array([F[j, k, 0]-F[i, k, 3], F[j, k, 1]-F[i, k, 2], F[j, k, 2]-F[i, k, 1], F[j, k, 3]-F[i, k, 0]])
                    P = pref_trap(D_ij, int(PREF_TYPES[k]), q[k], p[k])
                    sum_p += P
                    # Pji is recomputed by reversed pair through the same criterion.
                    if CRIT_DIR[k] == 1:
                        D_ji = np.array([F[j, k, 0]-F[i, k, 3], F[j, k, 1]-F[i, k, 2], F[j, k, 2]-F[i, k, 1], F[j, k, 3]-F[i, k, 0]])
                    else:
                        D_ji = np.array([F[i, k, 0]-F[j, k, 3], F[i, k, 1]-F[j, k, 2], F[i, k, 2]-F[j, k, 1], F[i, k, 3]-F[j, k, 0]])
                    Pji = pref_trap(D_ji, int(PREF_TYPES[k]), q[k], p[k])
                    sum_m += Pji
                exp_values[i, e] = defuzz_coa(trap_scalar_div(sum_p, n_req - 1) - trap_scalar_div(sum_m, n_req - 1))
        uni[:, k] = exp_values.mean(axis=1)
    return uni


def plot_gaia(data, result):
    X = compute_unicriterion_net(data, result)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    total = np.sum(S ** 2)
    explained = (S[:2] ** 2) / total * 100 if total > 0 else np.array([0.0, 0.0])
    load = Vt[:2, :].T

    # Escala dos vetores. Os vetores representam a direção dos critérios no plano GAIA.
    max_coord = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1e-9)
    arrow_scale = max_coord * 0.22

    labels = _req_labels(data["ids"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers",
        text=labels,
        customdata=np.column_stack([labels, result["phi_net"], result["rank_siprem"]]),
        marker=dict(
            size=10,
            color=result["phi_net"],
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Φ"),
            line=dict(width=0.5, color="#444"),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Φ líquido: %{customdata[1]:.4f}<br>"
            "Rank SiPREM: %{customdata[2]:.0f}<extra></extra>"
        ),
        name="Requisitos",
    ))

    # Vetores dos critérios: cada um recebe uma cor própria, rótulo no extremo
    # e entrada na legenda. Isso evita a ambiguidade causada por várias setas pretas
    # convergindo para a mesma região da origem.
    criterion_colors = [
        "#1f77b4",  # azul
        "#d62728",  # vermelho
        "#2ca02c",  # verde
        "#9467bd",  # roxo
        "#ff7f0e",  # laranja
    ]
    short_names = {
        "Impacto": "Impacto",
        "Custo": "Custo",
        "CAPACIDADE NACIONAL DE IMPLEMENTAÇÃO": "Capacidade nacional",
        "Prazo": "Prazo",
        "RiscoVerif": "Risco de verificação",
    }

    # Pequenos deslocamentos dos rótulos, proporcionais ao quadrante do vetor.
    # Assim os nomes não ficam todos exatamente sobre a ponta das setas.
    label_offsets = [
        (0.018, 0.018),
        (0.018, -0.022),
        (0.018, 0.024),
        (-0.020, -0.022),
        (-0.020, 0.022),
    ]

    for k, crit in enumerate(CRITERIA):
        vx = float(load[k, 0] * arrow_scale)
        vy = float(load[k, 1] * arrow_scale)
        color = criterion_colors[k % len(criterion_colors)]
        name = short_names.get(crit, crit)

        # Linha do vetor na legenda.
        fig.add_trace(go.Scatter(
            x=[0, vx], y=[0, vy],
            mode="lines",
            line=dict(color=color, width=3),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Vetor do critério<extra></extra>",
            showlegend=True,
        ))

        # Cabeça da seta.
        fig.add_annotation(
            x=vx, y=vy, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.8,
            arrowcolor=color, text="",
        )

        # Nome do critério próximo à ponta, com fundo branco para leitura.
        ox, oy = label_offsets[k % len(label_offsets)]
        fig.add_annotation(
            x=vx + ox * max_coord,
            y=vy + oy * max_coord,
            xref="x", yref="y",
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=11, color=color),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )

    # Identificação explícita da origem dos vetores.
    fig.add_annotation(
        x=0, y=0,
        xref="x", yref="y",
        text="<b>Origem</b>",
        showarrow=False,
        yshift=-18,
        font=dict(size=10, color="#555"),
        bgcolor="rgba(255,255,255,0.75)",
    )

    fig.add_hline(y=0, line_width=0.5, line_color="#cccccc")
    fig.add_vline(x=0, line_width=0.5, line_color="#cccccc")
    fig.update_layout(
        title=f"GAIA — plano baseado em fluxos líquidos unicritério (informação preservada: {explained[:2].sum():.2f}%)",
        xaxis_title=f"GAIA axis 1 ({explained[0]:.1f}%)",
        yaxis_title=f"GAIA axis 2 ({explained[1]:.1f}%)",
        height=700,
        margin=dict(l=70, r=60, t=80, b=70),
        hovermode="closest",
        legend=dict(
            title="Vetores dos critérios",
            orientation="v",
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
    )
    return fig

def plot_promethee_relation(data, result):
    rel = preference_relation_matrix(result, data["ids"])
    labels = _req_labels(data["ids"])
    mapping = {"P-": 0, "I": 1, "P": 2, "R": 3}
    z = np.vectorize(mapping.get)(rel)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=rel,
        texttemplate="%{text}",
        textfont=dict(size=7),
        colorscale=[
            [0.00, "#3333cc"], [0.249, "#3333cc"],
            [0.250, "#aaaaaa"], [0.499, "#aaaaaa"],
            [0.500, "#22aa22"], [0.749, "#22aa22"],
            [0.750, "#e8cc28"], [1.00, "#e8cc28"],
        ],
        zmin=0, zmax=3,
        xgap=1, ygap=1,
        colorbar=dict(
            tickvals=[0, 1, 2, 3],
            ticktext=["P− (Dominado)", "I (Indiferente)", "P (Domina)", "R (Incomparável)"],
            title="Relação",
        ),
        hovertemplate="i=%{y}<br>j=%{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Relações do Fuzzy PROMETHEE I no SiPREM",
        xaxis_title="Requisito j",
        yaxis_title="Requisito i",
        height=850,
        margin=dict(l=80, r=80, t=70, b=80),
    )
    return fig


# ---------------- UI ----------------
st.title("SiPREM")
st.subheader("Sistema de Priorização de Requisitos de Equipamentos Militares")
st.caption("Versão web experimental — núcleo matemático portado do MATLAB para Python")

with st.sidebar:
    st.header("1. Entradas")
    requisitos_file = st.file_uploader("Planilha de avaliações dos especialistas", type=["xlsx"], help="Use a mesma planilha atualmente utilizada pelo SiPREM/MATLAB.")
    fahp_file = st.file_uploader("Planilha FAHP", type=["xlsx"], help="Use a mesma planilha FAHP atualmente utilizada pelo SiPREM/MATLAB.")
    st.divider()
    st.header("2. Execução")
    executar = st.button("▶ Executar SiPREM", type="primary", use_container_width=True)
    st.divider()
    st.caption("Critérios configurados no núcleo atual:")
    for c in CRITERIA:
        st.write(f"• {c}")

if "result" not in st.session_state:
    st.info("Envie as duas planilhas na barra lateral e clique em **Executar SiPREM**.")

if executar:
    if requisitos_file is None or fahp_file is None:
        st.error("É necessário selecionar as duas planilhas antes de executar.")
        st.stop()
    req_path = save_uploaded(requisitos_file)
    fahp_path = save_uploaded(fahp_file)
    try:
        with st.spinner("Executando FAHP + Fuzzy PROMETHEE + classificação..."):
            data = read_expert_workbook(req_path)
            fahp = read_fahp_workbook(fahp_path)
            result = run_siprem(data["evaluations"], fahp["w_fuzzy_individual"], random_state=0)
            ranking = build_result_dataframe(data, result)
            excel_bytes = build_excel(data, result)
        st.session_state.update(data=data, fahp=fahp, result=result, ranking=ranking, excel_bytes=excel_bytes)
        st.success("SiPREM executado com sucesso.")
    except Exception as exc:
        st.error("Não foi possível executar o SiPREM com os arquivos enviados.")
        st.exception(exc)
        st.stop()

if "result" in st.session_state:
    data = st.session_state["data"]
    fahp = st.session_state["fahp"]
    result = st.session_state["result"]
    ranking = st.session_state["ranking"]

    st.divider()
    st.header("Resultado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Especialistas", data["n_experts"])
    c2.metric("Requisitos", data["n_requirements"])
    c3.metric("Critérios", data["n_criteria"])
    c4.metric("Requisitos KEY", int(np.sum(ranking["Categoria"] == "Key")))

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking", "📈 Gráficos", "👥 Especialistas", "ℹ️ Entradas"])

    with tab1:
        st.dataframe(ranking, use_container_width=True, hide_index=True, column_config={
            "Phi_Plus": st.column_config.NumberColumn("Phi+", format="%.6f"),
            "Phi_Minus": st.column_config.NumberColumn("Phi−", format="%.6f"),
            "Phi_Liquido": st.column_config.NumberColumn("Phi líquido", format="%.6f"),
            "Alerta_Divergencia_Impacto": st.column_config.CheckboxColumn("Divergência de impacto"),
        })
        st.download_button("⬇️ Baixar resultado em Excel", data=st.session_state["excel_bytes"], file_name="Resultado_SiPREM_Web.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    with tab2:
        st.subheader("1. Pesos dos critérios — FAHP-Express")
        st.plotly_chart(plot_weights(fahp), use_container_width=True)
        st.caption("Barras = média dos pesos defuzzificados; pontos = pesos individuais dos especialistas.")

        st.subheader("2. Relações do Fuzzy PROMETHEE I")
        st.plotly_chart(plot_promethee_relation(data, result), use_container_width=True)
        st.caption("P = domina; P− = dominado; I = indiferente; R = incomparável.")

        st.subheader("3. Heatmap das posições no ranking")
        st.plotly_chart(plot_rank_heatmap(data, result), use_container_width=True)

        st.subheader("4. Heatmap das classificações por fonte")
        st.plotly_chart(plot_classification_heatmap(data, result), use_container_width=True)

        st.subheader("5. GAIA — fluxos líquidos unicritério")
        with st.spinner("Calculando o plano GAIA..."):
            st.plotly_chart(plot_gaia(data, result), use_container_width=True)

        st.subheader("6. Boxplot dos fluxos líquidos individuais")
        st.plotly_chart(plot_boxplot(data, result), use_container_width=True)

    with tab3:
        phi_exp = pd.DataFrame(result["phi_net_experts"], index=data["ids"], columns=result["expert_names"])
        phi_exp.index.name = "Requisito"
        st.dataframe(phi_exp, use_container_width=True)
        st.markdown("**Impacto de missão informado pelos especialistas**")
        impact = pd.DataFrame(result["impact_matrix"], index=data["ids"], columns=result["expert_names"])
        impact.index.name = "Requisito"
        st.dataframe(impact, use_container_width=True)

    with tab4:
        st.write("**Abas de especialistas encontradas:**", ", ".join(data["sheets"]))
        st.write("**Planilha FAHP:**", ", ".join(fahp["sheets"]))
        st.table(pd.DataFrame({
            "Critério": CRITERIA,
            "Direção": ["Maximizar", "Minimizar", "Maximizar", "Minimizar", "Minimizar"],
            "Tipo de preferência": ["Usual", "Level", "Level", "Level", "Level"],
        }))

st.divider()
st.caption("SiPREM — interface web. A lógica de cálculo permanece no módulo core/.")
