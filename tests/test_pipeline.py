import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import make_fahp, make_requirements
from core.data_loader import (
    CRIT_DIR, PREF_TYPES, InputValidationError, align_fahp_to_data,
    read_expert_workbook, read_fahp_workbook, select_expert_sheets,
)
from core.siprem_core import build_fuzzy_matrix, default_thresholds, run_siprem
from core.promethee import (
    promethee_i_relations, run_fuzzy_promethee_trap, run_fuzzy_promethee_trap_reference,
)
from core.analysis import (
    agreement_matrices, agreement_summary, build_ranking_table, class_uncertainty,
    criterion_contributions, paginate,
)
from core.export import build_excel, build_html_report


@pytest.fixture()
def files(tmp_path):
    return make_requirements(tmp_path / "req.xlsx"), make_fahp(tmp_path / "fahp.xlsx")


@pytest.fixture()
def loaded(files):
    data = read_expert_workbook(files[0])
    fahp = read_fahp_workbook(files[1])
    fahp, _ = align_fahp_to_data(data, fahp)
    return data, fahp, run_siprem(data["evaluations"], fahp["w_fuzzy_individual"])


def test_vectorized_matches_reference(loaded):
    data, fahp, _ = loaded
    q, p = default_thresholds()
    F = build_fuzzy_matrix(data["evaluations"]["E1"])
    w = fahp["w_fuzzy_individual"][:, :, 0]
    ref = run_fuzzy_promethee_trap_reference(F, w, CRIT_DIR, PREF_TYPES, q, p)
    vec = run_fuzzy_promethee_trap(F, w, CRIT_DIR, PREF_TYPES, q, p)
    for a, b in zip(ref, vec):
        assert np.allclose(a, b, atol=1e-12)


def test_select_expert_sheets():
    assert select_expert_sheets(["E1", "E2", "tab"])[0] == ["E1", "E2"]
    assert select_expert_sheets(["Especialista 1", "Especialista 2", "LEIA-ME"])[0] == ["Especialista 1", "Especialista 2"]
    exp, ign, fallback = select_expert_sheets(["A", "B", "C"])
    assert exp == ["A", "B"] and fallback


def test_pipeline_outputs(loaded):
    data, fahp, r = loaded
    n = data["n_requirements"]
    assert r["phi_net"].shape == (n,)
    assert r["categories_experts"].shape == (n, 3)
    assert r["unicriterion_net"].shape == (n, 5)
    assert np.isclose(fahp["w_crisp_group"].sum(), 1.0)
    assert sorted(set(r["rank_siprem"])) == sorted(set(r["rank_siprem"]))  # sem NaN
    assert not np.isnan(r["phi_net"]).any()


def test_invalid_value_reports_location(tmp_path):
    p = make_requirements(tmp_path / "bad.xlsx", bad=("E2", 5, 4, "Altto"))
    with pytest.raises(InputValidationError) as exc:
        read_expert_workbook(p)
    msg = " ".join(exc.value.messages)
    assert "E2" in msg and "linha 5" in msg and "Altto" in msg


def test_expert_count_mismatch(tmp_path):
    data = read_expert_workbook(make_requirements(tmp_path / "r.xlsx"))
    fahp = read_fahp_workbook(make_fahp(tmp_path / "f.xlsx", sheets=("E1", "E2")))
    with pytest.raises(InputValidationError):
        align_fahp_to_data(data, fahp)


def test_fahp_reordered_by_name(tmp_path):
    data = read_expert_workbook(make_requirements(tmp_path / "r.xlsx"))
    fahp = read_fahp_workbook(make_fahp(tmp_path / "f.xlsx", sheets=("E3", "E1", "E2")))
    aligned, warn = align_fahp_to_data(data, fahp)
    assert aligned["sheets"] == ["E1", "E2", "E3"] and warn


def test_promethee_i_relations():
    rel = promethee_i_relations([0.8, 0.5, 0.5], [0.1, 0.4, 0.4])
    assert rel[0, 1] == "P" and rel[1, 0] == "P-" and rel[1, 2] == "I" and rel[0, 0] == "I"


def test_threshold_validation(files):
    data = read_expert_workbook(files[0])
    fahp = read_fahp_workbook(files[1])
    with pytest.raises(ValueError):
        run_siprem(data["evaluations"], fahp["w_fuzzy_individual"], q_vals=[0, 2, 2, 2, 2], p_vals=[0, 1, 1, 1, 1])


def test_analysis_and_exports(loaded):
    data, fahp, r = loaded
    assert paginate(300, 40)[-1] == (280, 300)
    table = build_ranking_table(data, r)
    assert len(table) == data["n_requirements"] and table["Rank"].iloc[0] == 1
    assert len(agreement_summary(r)) == 3
    assert agreement_matrices(r)["kappa"].shape == (4, 4)
    assert criterion_contributions(r, fahp["w_crisp_group"]).shape == (data["n_requirements"], 5)
    assert class_uncertainty(r)["Classe_incerta"].dtype == bool
    assert len(build_excel(data, r, fahp)) > 1000
    assert "Relatório SiPREM" in build_html_report(data, r, fahp)
