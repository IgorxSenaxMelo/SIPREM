"""Comparador genérico MATLAB × Python.

Uso:
    python tests/validate_matlab_vs_python.py

O script procura um Excel de referência exportado pelo MATLAB.
Ele não inventa valores MATLAB: se o arquivo de referência não existir,
apenas informa o que falta.

Formato esperado do arquivo MATLAB:
- aba 'Consolidado' com Rank, Requisito, Phi_Plus, Phi_Minus,
  Phi_Liquido e Categoria.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_loader import read_expert_workbook, read_fahp_workbook
from core.siprem_core import run_siprem


def compare_vectors(name, matlab, python, atol=1e-10):
    matlab = np.asarray(matlab, dtype=float)
    python = np.asarray(python, dtype=float)
    if matlab.shape != python.shape:
        return {
            "Resultado": name,
            "MATLAB": str(matlab.shape),
            "Python": str(python.shape),
            "Diferença": "dimensão diferente",
            "Status": "ERRO",
        }

    diff = np.max(np.abs(matlab-python)) if matlab.size else 0.0
    return {
        "Resultado": name,
        "MATLAB": float(np.max(matlab)) if matlab.size else 0.0,
        "Python": float(np.max(python)) if python.size else 0.0,
        "Diferença": float(diff),
        "Status": "OK" if diff <= atol else "DIVERGENTE",
    }


def main():
    data_path = ROOT / "data" / "dados_requisitos_lista.xlsx"
    fahp_path = ROOT / "data" / "F_AHP_A.xlsx"
    matlab_path = ROOT / "data" / "Resultado_SIPREM_FAHP_FuzzyPromethee_Geldermann.xlsx"

    if not data_path.exists() or not fahp_path.exists():
        print("Coloque os dois arquivos Excel de entrada em C:\\SIPREM\\data.")
        return

    data = read_expert_workbook(data_path)
    fahp = read_fahp_workbook(fahp_path)
    result = run_siprem(data["evaluations"], fahp["w_fuzzy_individual"])

    print("=== VALIDAÇÃO ESTRUTURAL ===")
    print(f"Especialistas: {data['n_experts']}")
    print(f"Requisitos:    {data['n_requirements']}")
    print(f"Critérios:     {data['n_criteria']}")
    print()
    print("Pesos fuzzy individuais:")
    print(fahp["w_fuzzy_individual"])

    if not matlab_path.exists():
        print("\nArquivo MATLAB de referência ainda não encontrado:")
        print(matlab_path)
        print("Exportar o resultado do MATLAB para esse arquivo é necessário para")
        print("a comparação numérica automática.")
        return

    mat = pd.read_excel(matlab_path, sheet_name="Consolidado")
    if "Phi_Liquido" not in mat.columns:
        raise ValueError("A aba Consolidado não possui a coluna Phi_Liquido.")

    rows = []
    rows.append(compare_vectors(
        "Phi(net) SIPREM",
        mat["Phi_Liquido"].to_numpy(),
        result["phi_net"],
    ))

    rows.append(compare_vectors(
        "Rank SIPREM",
        mat["Rank"].to_numpy(),
        result["rank_siprem"],
        atol=0,
    ))

    out = pd.DataFrame(rows)
    print("\n=== COMPARAÇÃO MATLAB × PYTHON ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
