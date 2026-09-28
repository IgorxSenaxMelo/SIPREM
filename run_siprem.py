from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.data_loader import read_expert_workbook, read_fahp_workbook
from core.siprem_core import run_siprem

DATA = ROOT / "data" / "dados_requisitos_lista.xlsx"
FAHP = ROOT / "data" / "F_AHP_A.xlsx"

data = read_expert_workbook(DATA)
fahp = read_fahp_workbook(FAHP)
result = run_siprem(data["evaluations"], fahp["w_fuzzy_individual"], random_state=0)

df = pd.DataFrame({
    "Rank": result["rank_siprem"].astype(int),
    "Requisito": data["ids"],
    "Phi_Plus": result["phi_plus"],
    "Phi_Minus": result["phi_minus"],
    "Phi_Liquido": result["phi_net"],
    "Categoria": result["categories"],
    "Alerta_Divergencia_Impacto": (
        result["impact_matrix"].max(axis=1) -
        result["impact_matrix"].min(axis=1) >= 3
    ),
}).sort_values("Rank")

out = ROOT / "output"
out.mkdir(exist_ok=True)
df.to_excel(out / "Resultado_SIPREM_Python.xlsx", index=False)

print("=== SiPREM Python ===")
print(f"Especialistas: {data['n_experts']}")
print(f"Requisitos:    {data['n_requirements']}")
print(f"Critérios:     {data['n_criteria']}")
print()
print(df.head(10).to_string(index=False))
print()
print("Arquivo gerado:")
print(out / "Resultado_SIPREM_Python.xlsx")
