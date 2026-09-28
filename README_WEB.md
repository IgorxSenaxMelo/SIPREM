# SiPREM — versão Web (iniciante)

Esta versão transforma o núcleo Python do SiPREM em uma interface web local usando **Streamlit**.

A lógica matemática permanece no diretório `core/`. A interface apenas:

1. recebe a planilha de avaliações dos especialistas;
2. recebe a planilha FAHP;
3. executa o núcleo SiPREM;
4. mostra o ranking na tela;
5. mostra o gráfico do Phi líquido;
6. permite baixar o resultado em Excel.

## 1. O que você precisa instalar

No Windows, instale o **Python 3.11 ou 3.12** caso ainda não tenha Python.

Durante a instalação do Python, marque:

> Add Python to PATH

## 2. Coloque a pasta do projeto

Extraia a pasta `SIPREM` em um local simples, por exemplo:

```text
C:\SIPREM
```

A pasta deve conter:

```text
C:\SIPREM
├── app.py
├── requirements.txt
├── core\
├── data\
├── tests\
└── output\
```

## 3. Abra o PowerShell

Entre na pasta:

```powershell
cd C:\SIPREM
```

## 4. Crie o ambiente virtual

Execute uma vez:

```powershell
python -m venv .venv
```

Depois ative:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o Windows bloquear a ativação, execute no PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois tente novamente:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 5. Instale as bibliotecas

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Isso instala, entre outras, as bibliotecas usadas pelo SiPREM:

- pandas
- numpy
- scipy
- scikit-learn
- openpyxl
- plotly
- streamlit

## 6. Teste o programa

Ainda no PowerShell:

```powershell
streamlit run app.py
```

O navegador deverá abrir automaticamente em um endereço parecido com:

```text
http://localhost:8501
```

Se não abrir, copie esse endereço para o navegador.

## 7. Como usar

Na barra lateral da página:

### Planilha 1 — avaliações

Selecione a mesma planilha Excel usada atualmente pelo MATLAB, contendo as abas dos especialistas.

### Planilha 2 — FAHP

Selecione a planilha Excel usada atualmente para os pesos FAHP.

Depois clique em:

> ▶ Executar SiPREM

A aplicação executará:

```text
Planilhas
   ↓
Leitura dos especialistas
   ↓
FAHP / pesos fuzzy
   ↓
Fuzzy PROMETHEE
   ↓
Phi+ / Phi− / Phi líquido
   ↓
Identificação de requisitos KEY
   ↓
K-means dos demais requisitos
   ↓
Ranking SiPREM
```

## 8. Resultado

A página mostra:

- número de especialistas;
- número de requisitos;
- número de critérios;
- quantidade de requisitos KEY;
- ranking completo;
- Phi+;
- Phi−;
- Phi líquido;
- categoria de cada requisito;
- alerta de divergência de impacto;
- gráfico do Phi líquido;
- resultados individuais dos especialistas.

Também existe o botão:

> ⬇️ Baixar resultado em Excel

## 9. Como fechar o programa

Volte ao PowerShell onde o Streamlit está executando e pressione:

```text
Ctrl + C
```

## 10. Como abrir novamente depois

Na próxima vez, não precisa recriar o ambiente virtual.

Abra o PowerShell e faça:

```powershell
cd C:\SIPREM
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## 11. Atalho opcional

O arquivo `executar_siprem_web.bat` pode ser usado para facilitar a inicialização.

Basta dar dois cliques nele.

## Observação importante

Esta é a **primeira camada web** do SiPREM. Ela foi feita para preservar o núcleo de cálculo já convertido para Python.

A validação numérica MATLAB × Python foi deliberadamente deixada para uma etapa posterior, conforme solicitado. Portanto, esta versão deve ser tratada como **protótipo funcional da interface web**, não como versão final validada para uso operacional.
