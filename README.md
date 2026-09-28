# SiPREM — conversão MATLAB → Python

Esta é a **Fase C** do projeto: conversão do núcleo matemático.
A interface Streamlit ainda não é o foco desta etapa.

## Estrutura

```text
C:\SIPREM
├── core\
│   ├── data_loader.py
│   ├── fahp.py
│   ├── fuzzy_numbers.py
│   ├── promethee.py
│   └── siprem_core.py
├── tests\
│   ├── test_core.py
│   └── validate_matlab_vs_python.py
├── data\
│   ├── dados_requisitos_lista.xlsx
│   └── F_AHP_A.xlsx
├── requirements.txt
└── README.md
```

## Instalação

No PowerShell, dentro de `C:\SIPREM`:

```bash
python -m venv .venv
```

Ative:

```bash
.venv\Scripts\Activate.ps1
```

Instale:

```bash
pip install -r requirements.txt
```

## Teste do núcleo

```bash
pytest
```

## Próximo teste

Copie os dois Excel reais para `C:\SIPREM\data` usando os nomes:

```text
dados_requisitos_lista.xlsx
F_AHP_A.xlsx
```

Depois:

```bash
python tests\validate_matlab_vs_python.py
```

Para validar numericamente contra o MATLAB, também será necessário colocar em `data` o Excel de referência exportado pelo MATLAB:

```text
Resultado_SIPREM_FAHP_FuzzyPromethee_Geldermann.xlsx
```

com a aba `Consolidado`.

## Observação científica

Nenhuma simplificação metodológica foi introduzida deliberadamente nesta fase. As operações fuzzy e as regras principais foram portadas conforme as funções MATLAB analisadas.
