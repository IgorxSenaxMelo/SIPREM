"""Leitura e validação das planilhas do SiPREM."""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

from .fahp import group_weights_from_reference_rows, saaty_linguistic2num


CRITERIA = [
    "Impacto",
    "Custo",
    "CAPACIDADE NACIONAL DE IMPLEMENTAÇÃO",
    "Prazo",
    "RiscoVerif",
]

CRIT_DIR = np.array([1, -1, 1, -1, -1], dtype=int)
PREF_TYPES = np.array([1, 4, 4, 4, 4], dtype=int)
SCALE_MAX = np.array([7, 7, 7, 7, 7], dtype=int)
SCALE_MIN = np.array([1, 1, 1, 1, 1], dtype=int)


# Escala fuzzy trapezoidal de avaliação de 7 níveis (Ghorabaee).
# Cada linha corresponde aos níveis linguísticos de 1 a 7.
TFN_EVAL_7 = np.array([
    [0, 0, 1, 2],
    [1, 2, 2, 3],
    [2, 3, 4, 5],
    [4, 5, 5, 6],
    [5, 6, 7, 8],
    [7, 8, 8, 9],
    [8, 9, 10, 10],
], dtype=float)



class InputValidationError(ValueError):
    """Erro de planilha com uma lista de mensagens legíveis para o usuário."""

    def __init__(self, messages):
        self.messages = list(messages)
        super().__init__("\n".join(self.messages))


MAX_REPORTED_ERRORS = 40

_EXPERT_SHEET_RE = re.compile(
    r"^(e|esp|especialista|expert|expert[oa])\s*[-_.]?\s*\d+$", re.IGNORECASE
)


def select_expert_sheets(sheets):
    """Escolhe as abas de especialistas.

    Aceita E1, E2..., Especialista 1, Especialista 2... e ignora abas como
    'tab', 'DADOS' ou 'LEIA-ME'. Se nenhum nome seguir o padrão, usa o
    comportamento antigo (todas menos a última) e avisa.

    Retorna (abas_de_especialistas, abas_ignoradas, usou_fallback).
    """
    sheets = list(sheets)
    experts = [s for s in sheets if _EXPERT_SHEET_RE.match(str(s).strip())]
    if experts:
        return experts, [s for s in sheets if s not in experts], False
    if len(sheets) >= 2:
        return sheets[:-1], sheets[-1:], True
    return sheets, [], True


def _raise_if_errors(errors):
    if errors:
        shown = errors[:MAX_REPORTED_ERRORS]
        if len(errors) > MAX_REPORTED_ERRORS:
            shown.append(f"... e mais {len(errors) - MAX_REPORTED_ERRORS} problema(s).")
        raise InputValidationError(shown)


def linguistic2num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or (not isinstance(x, str) and pd.isna(x)):
        raise ValueError("célula vazia")

    if isinstance(x, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(x)):
            raise ValueError("valor NaN/inf")
        v = float(x)
        if v < SCALE_MIN[0] or v > SCALE_MAX[0]:
            raise ValueError(f"valor numérico {v:g} fora da escala 1–7")
        return v

    s = str(x).strip().lower()
    mapping = {
        "muito alto": 7,
        "alto": 6,
        "moderadamente alto": 5,
        "médio": 4,
        "medio": 4,
        "moderadamente baixo": 3,
        "baixo": 2,
        "muito baixo": 1,
    }
    if s not in mapping:
        raise ValueError(f"valor linguístico desconhecido: {x!r}")
    return mapping[s]


def fuzzify_by_scale(val, scale_max):
    if int(scale_max) != 7:
        raise ValueError("Apenas a escala de 7 níveis está disponível nesta versão.")
    v = int(np.rint(float(val)))
    v = max(1, min(7, v))
    return TFN_EVAL_7[v - 1].copy()


def read_expert_workbook(path):
    path = Path(path)
    xls = pd.ExcelFile(path)
    expert_sheets, ignored, fallback = select_expert_sheets(xls.sheet_names)

    if not expert_sheets:
        raise InputValidationError(["Nenhuma aba de especialista foi encontrada."])

    errors = []
    evaluations = {}
    ids = None
    descriptions = None
    criteria_headers = None

    for sheet in expert_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=0)

        if df.shape[1] < 2 + len(CRITERIA):
            errors.append(
                f"Aba '{sheet}': são necessárias pelo menos {2 + len(CRITERIA)} colunas "
                f"(ID, descrição e {len(CRITERIA)} critérios); encontradas {df.shape[1]}."
            )
            continue

        df = df.iloc[:, :2 + len(CRITERIA)]
        df = df.dropna(how="all").reset_index(drop=True)

        if df.shape[0] < 2:
            errors.append(f"Aba '{sheet}': são necessários pelo menos 2 requisitos.")
            continue

        current_ids = df.iloc[:, 0].tolist()
        current_desc = df.iloc[:, 1].tolist()

        for i, rid in enumerate(current_ids):
            if pd.isna(rid):
                errors.append(f"Aba '{sheet}', linha {i + 2}: ID do requisito vazio.")

        if len(set(map(str, current_ids))) != len(current_ids):
            dup = pd.Series(current_ids).astype(str)
            dup = sorted(set(dup[dup.duplicated()]))[:5]
            errors.append(f"Aba '{sheet}': IDs duplicados (ex.: {', '.join(dup)}).")

        if ids is None:
            ids = current_ids
            descriptions = current_desc
            criteria_headers = [str(c) for c in df.columns[2:2 + len(CRITERIA)]]
        elif current_ids != ids:
            if len(current_ids) != len(ids):
                errors.append(
                    f"Aba '{sheet}': {len(current_ids)} requisitos, mas a primeira aba "
                    f"tem {len(ids)}."
                )
            else:
                k = next(i for i, (a, b) in enumerate(zip(current_ids, ids)) if a != b)
                errors.append(
                    f"Aba '{sheet}', linha {k + 2}: ID {current_ids[k]!r} difere da "
                    f"primeira aba ({ids[k]!r}). Todas as abas devem ter os mesmos IDs na mesma ordem."
                )

        numeric = np.full((df.shape[0], len(CRITERIA)), np.nan, dtype=float)
        for i in range(df.shape[0]):
            for k in range(len(CRITERIA)):
                try:
                    numeric[i, k] = linguistic2num(df.iat[i, 2 + k])
                except ValueError as exc:
                    errors.append(
                        f"Aba '{sheet}', linha {i + 2} (requisito {current_ids[i]}), "
                        f"coluna '{CRITERIA[k]}': {exc}."
                    )
        evaluations[sheet] = numeric

    _raise_if_errors(errors)

    return {
        "sheets": expert_sheets,
        "ignored_sheets": ignored,
        "used_sheet_fallback": fallback,
        "ids": ids,
        "descriptions": descriptions,
        "evaluations": evaluations,
        "n_experts": len(expert_sheets),
        "n_requirements": len(ids),
        "n_criteria": len(CRITERIA),
    }


def read_fahp_workbook(path):
    path = Path(path)
    xls = pd.ExcelFile(path)
    expert_sheets, ignored, fallback = select_expert_sheets(xls.sheet_names)

    if not expert_sheets:
        raise InputValidationError(["Nenhuma aba de especialista foi encontrada na planilha FAHP."])

    errors = []
    rows = []
    for sheet in expert_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        if df.shape[0] < 4 or df.shape[1] < 1 + len(CRITERIA):
            errors.append(
                f"Aba FAHP '{sheet}': a linha 4 (B4:F4) com os {len(CRITERIA)} julgamentos "
                f"de referência não foi encontrada."
            )
            continue
        row = df.iloc[3, 1:1 + len(CRITERIA)].tolist()  # B4:F4
        for k, cell in enumerate(row):
            try:
                v = saaty_linguistic2num(cell)
                if not (1 <= v <= 9):
                    raise ValueError(f"valor {v:g} fora da escala de Saaty 1–9")
            except ValueError as exc:
                errors.append(f"Aba FAHP '{sheet}', célula {chr(ord('B') + k)}4: {exc}.")
        rows.append(row)

    _raise_if_errors(errors)

    result = group_weights_from_reference_rows(rows)
    result["sheets"] = expert_sheets
    result["ignored_sheets"] = ignored
    result["used_sheet_fallback"] = fallback
    return result


def _norm_name(x):
    return re.sub(r"\s+", "", str(x)).lower()


def align_fahp_to_data(data, fahp):
    """Garante que os pesos FAHP e as avaliações usem os mesmos especialistas.

    Retorna (fahp_alinhado, avisos). Levanta InputValidationError se o número de
    especialistas for diferente nas duas planilhas.
    """
    n_data = data["n_experts"]
    n_fahp = len(fahp["sheets"])

    if n_data != n_fahp:
        raise InputValidationError([
            f"A planilha de avaliações tem {n_data} especialista(s) "
            f"({', '.join(data['sheets'])}), mas a planilha FAHP tem {n_fahp} "
            f"({', '.join(fahp['sheets'])}). Os dois arquivos precisam ter os mesmos especialistas."
        ])

    warnings = []
    names_data = [_norm_name(s) for s in data["sheets"]]
    names_fahp = [_norm_name(s) for s in fahp["sheets"]]

    if names_data == names_fahp:
        return fahp, warnings

    if sorted(names_data) == sorted(names_fahp):
        order = [names_fahp.index(n) for n in names_data]
        aligned = dict(fahp)
        aligned["w_fuzzy_individual"] = fahp["w_fuzzy_individual"][:, :, order]
        aligned["w_crisp_individual"] = fahp["w_crisp_individual"][:, order]
        aligned["sheets"] = [fahp["sheets"][i] for i in order]
        warnings.append("As abas dos especialistas estavam em ordens diferentes nas duas "
                        "planilhas; os pesos FAHP foram reordenados pelo nome da aba.")
        return aligned, warnings

    warnings.append(
        "Os nomes das abas de especialistas diferem entre as planilhas "
        f"({', '.join(data['sheets'])} × {', '.join(fahp['sheets'])}). "
        "Os pesos FAHP foram associados às avaliações pela POSIÇÃO das abas."
    )
    return fahp, warnings
