# SiPREM — Sistema de Priorização de Requisitos de Equipamentos Militares

FAHP-Express + Fuzzy PROMETHEE, portado do MATLAB para Python, com interface web (Streamlit).

## O que o sistema entrega

- **Ranking** dos requisitos com classe (Key, Grupo 1, 2, 3), Φ líquido, **classe incerta** (requisito perto da fronteira entre classes) e **divergência de impacto** entre especialistas.
- **Busca e filtros** por texto, classe, incerteza e divergência.
- **Explicação por requisito**: contribuição de cada critério e visão de cada especialista.
- **Gráficos paginados** (padrão: 40 requisitos por página, em ordem de ranking): posições, classificações e matriz do PROMETHEE I; mais pesos FAHP e GAIA.
- **Aderência dos especialistas** ao grupo: Spearman, histogramas de diferença de posição/classe, Kappa.
- **Exportações**: Excel completo (11 abas), relatório HTML e CSV do ranking filtrado.
- **Validação das planilhas** com mensagens que indicam aba, linha e coluna.

## Como usar

1. Instale o Python 3.11 ou 3.12 (marque "Add Python to PATH" no Windows).
2. Execute `executar_siprem_web.bat` (Windows) ou `./executar_siprem_web.sh` (Linux/macOS). Na primeira vez ele cria o ambiente e instala as dependências.
3. Na barra lateral, envie as duas planilhas (ou marque **Usar planilhas de exemplo**) e clique em **Executar SiPREM**.

Alternativa por linha de comando: `python run_siprem.py avaliacoes.xlsx pesos_fahp.xlsx saida.xlsx`.

Com Docker: `docker build -t siprem .` e `docker run -p 8501:8501 siprem`.

## Formato das planilhas

- **Avaliações**: uma aba por especialista, nomeada `E1, E2, …` ou `Especialista 1, 2, …`. Colunas: ID, descrição e os 5 critérios (Impacto, Custo, Capacidade nacional, Prazo, Risco de verificação) em níveis linguísticos de "Muito baixo" a "Muito alto". Todas as abas devem ter os mesmos IDs, na mesma ordem. Outras abas (`DADOS`, `LEIA-ME`) são ignoradas.
- **FAHP**: mesmas abas de especialistas; a linha 4 (B4:F4) traz os julgamentos de referência na escala de Saaty (texto). Os dois arquivos precisam ter os mesmos especialistas; se os nomes diferirem, o sistema avisa e associa por posição.

Veja `exemplos/` e o `Guia_de_Usuario_SiPREM.pdf`.

## Parâmetros avançados

Limiares q e p dos critérios com função Level, limite de divergência de impacto e semente do K-means. Os padrões reproduzem o MATLAB.

## Estrutura

```text
app.py              interface Streamlit
plots.py            figuras Plotly (paginadas)
run_siprem.py       execução por linha de comando
core/
  data_loader.py    leitura e validação das planilhas
  fahp.py           FAHP trapezoidal + Buckley
  promethee.py      Fuzzy PROMETHEE (vetorizado; versão de referência mantida para testes)
  siprem_core.py    pipeline principal
  classification.py Key por maioria + K-means
  analysis.py       aderência, contribuição por critério, incerteza de classe, paginação
  export.py         Excel completo e relatório HTML
tests/              pytest
```

## Testes

```bash
pytest
```

Os testes geram planilhas sintéticas e comparam a versão vetorizada do PROMETHEE com a implementação original.

## Notas de interpretação

- **Contribuição por critério** = peso defuzzificado × fluxo líquido unicritério. É uma aproximação para leitura; o Φ do SiPREM usa pesos fuzzy por especialista.
- **Classe incerta**: distância do Φ líquido até a fronteira entre grupos (ponto médio entre centroides do K-means) menor que a fração escolhida da distância entre os centroides. Requisitos Key não têm margem, pois são definidos por maioria de impacto máximo.
- O fluxo unicritério do GAIA agora usa a subtração fuzzy trapezoidal, como no MATLAB.
