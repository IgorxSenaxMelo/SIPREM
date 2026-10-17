"""Executa o SiPREM pela linha de comando (sem interface web).

Uso:
    python run_siprem.py avaliacoes.xlsx pesos_fahp.xlsx [saida.xlsx]

Sem argumentos, usa as planilhas de exemplo da pasta exemplos/.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.data_loader import (
    InputValidationError, align_fahp_to_data, read_expert_workbook, read_fahp_workbook,
)
from core.siprem_core import run_siprem
from core.analysis import build_ranking_table
from core.export import build_excel


def main(argv):
    ex = ROOT / "exemplos"
    req = Path(argv[1]) if len(argv) > 1 else ex / "Exemplo_Avaliacoes_Especialistas_SiPREM.xlsx"
    fahp_path = Path(argv[2]) if len(argv) > 2 else ex / "Exemplo_Pesos_FAHP_SiPREM.xlsx"
    out = Path(argv[3]) if len(argv) > 3 else ROOT / "output" / "Resultado_SiPREM.xlsx"

    try:
        data = read_expert_workbook(req)
        fahp, warnings = align_fahp_to_data(data, read_fahp_workbook(fahp_path))
    except InputValidationError as exc:
        print("Problemas encontrados nas planilhas:")
        for m in exc.messages:
            print(" -", m)
        return 1

    for w in warnings:
        print("AVISO:", w)

    result = run_siprem(data["evaluations"], fahp["w_fuzzy_individual"], random_state=0)
    table = build_ranking_table(data, result)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(build_excel(data, result, fahp))

    print("=== SiPREM Python ===")
    print(f"Especialistas: {data['n_experts']} ({', '.join(data['sheets'])})")
    print(f"Requisitos:    {data['n_requirements']}")
    print()
    print(table.head(10)[["Rank", "Requisito", "Categoria", "Phi_Liquido", "Classe_incerta"]].to_string(index=False))
    print(f"\nArquivo gerado: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
