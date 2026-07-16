# Projeto Fórmula 1 — Previsão Probabilística de Rankings

Projeto que investiga a previsão probabilística de resultados de corridas de Fórmula 1 por meio de modelos estatísticos de ranking.

---

## Problema

O resultado de uma corrida de Fórmula 1 é naturalmente um **ranking** de competidores. Tratá-lo como classificação binária (vence ou não vence) desperdiça a maior parte da informação disponível. Este projeto trata a previsão como um problema ordinal e probabilístico: em vez de prever apenas o vencedor, o modelo estima uma **distribuição de probabilidades sobre posições** para cada piloto.

---

## Abordagem

O projeto combina dois modelos probabilísticos de ranking:

**Modelo de Mallows** — agrupa corridas históricas com padrões de resultado similares, identificando contextos competitivos distintos (ex: eras de dominância de diferentes equipes).

**Modelo de Plackett–Luce** — estima parâmetros de habilidade relativa para cada piloto, globalmente e dentro de cada contexto identificado pelo Mallows.

O sistema opera de forma **incremental**: a previsão de cada corrida é feita antes de seu resultado ser incorporado ao modelo, respeitando a ordem temporal e evitando vazamento de informação.

A qualidade do modelo é avaliada tanto por métricas de concordância ordinal (Top-N accuracy, Kendall tau) quanto por métricas probabilísticas (Ranked Probability Score).

---

## Dados

Regra de fonte oficial por período (reestruturação):

| Período | Fonte |
|---|---|
| até 2022 | dataset histórico em `data/Season<ano>/` |
| 2023 em diante | OpenF1, sincronizada e processada em `data/openf1/` |

A sincronização é centralizada (`python -m src.data_openf1.sync`) — **nenhum
pipeline chama a API diretamente**. O carregador compartilhado
(`src/data/repository.py`) aplica a regra acima automaticamente. Detalhes,
comandos e manifests: [`data/openf1/README.md`](data/openf1/README.md).
A equivalência entre as fontes na transição 2023–2025 está documentada em
[`docs/relatorio_validacao_dados.md`](docs/relatorio_validacao_dados.md).

---

## Organização do Repositório

```
Projeto-Formula1/
├── artifacts/        # Execuções imutáveis dos pipelines (<pipeline>/<run_id>/)
├── artigos/          # Base bibliográfica do projeto
├── data/             # Dados históricos (CSVs) e camada OpenF1 (raw/processed/manifests)
├── docs/             # Plano de reestruturação, decisões e relatórios de validação
├── notebooks/        # Narrativa progressiva do projeto (00 → 07)
├── src/              # Módulos Python
│   ├── data/             # Loader histórico + repositório compartilhado (regra de fontes)
│   ├── data_openf1/      # Cliente, sincronizador central e features da OpenF1
│   ├── engine/           # Treinamento e predição (compartilhado)
│   ├── evaluation/       # Métricas de avaliação (compartilhado)
│   ├── models/           # Mallows, Plackett-Luce, pesos (compartilhado)
│   └── pipeline_*/       # Um diretório por pipeline (configs/ + README próprio)
├── tests/            # Suíte pytest (unit, data_quality, integration, e2e, regression, smoke)
├── pytest.ini
├── requirements.txt
└── README.md
```

Os módulos em `src/data/`, `src/data_openf1/`, `src/engine/`, `src/evaluation/` e `src/models/` são **compartilhados entre todos os pipelines**. As pastas `pipeline_*/` apenas orquestram esses módulos para cada experimento específico.

---

## Pipelines

Este README funciona como índice; cada pipeline tem README próprio com
objetivo, configuração, resultados esperados e saídas.

| Pipeline | Split (treino · val · teste) | Avaliação | Documentação |
|---|---|---|---|
| Mallows + Plackett–Luce | 2019–22 · 2023 · 2024 | Top-3/Top-5/Kendall τ | [`src/pipeline_mallows_plackett_luce/README.md`](src/pipeline_mallows_plackett_luce/README.md) |
| Score Rules (Monte Carlo + RPS) | 2019–22 · 2023 · 2024 | + RPS vs baseline uniforme | [`src/pipeline_score_rules/README.md`](src/pipeline_score_rules/README.md) |
| OpenF1 (contexto de corrida) | 2019–23 · 2024 · 2025 | + contexto SC/VSC/grid/DNF | [`src/pipeline_openf1/README.md`](src/pipeline_openf1/README.md) |
| Científico comparativo | — congelado; retomada na Etapa 9 | — | `src/pipeline_cientifico_comparativo/` |

Cada execução grava uma pasta imutável em `artifacts/<pipeline>/<run_id>/`
(config resolvida, manifest, métricas, tabelas Parquet/CSV, plots, log e
`nb_data*.pkl` auxiliar), com o ponteiro `latest_run.json` indicando a mais
recente — é dele que os notebooks 05–07 leem.

---

## Como Executar

```bash
# 1. Clone e configure o ambiente
git clone https://github.com/Guili-Lopes/Projeto-Formula1.git
cd Projeto-Formula1
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 2. Sincronize os dados da OpenF1 (2023+) quando necessário
python -m src.data_openf1.sync --start-year 2023 --end-year 2026 --profile core

# 3. Execute os pipelines individualmente
python -m src.pipeline_mallows_plackett_luce.run_experiment
python -m src.pipeline_score_rules.run_pipeline_score_rules
python -m src.pipeline_openf1.run_pipeline_openf1

# 4. Explore os notebooks em ordem
jupyter notebook notebooks/

# 5. Rode a suíte de testes
pytest            # rápida (unit, dados, integração, e2e sintético, smoke)
pytest -m slow    # + reexecução completa do P1 contra o baseline
```

Os notebooks de análise (05, 06, 07) leem os artefatos da última execução de
cada pipeline via `artifacts/<pipeline>/latest_run.json`.

---

## Referências

As principais referências teóricas estão disponíveis em `artigos/` e cobrem:

- Plackett–Luce e modelos de ranking probabilístico
- Modelo de Mallows para rankings parciais
- Inferência bayesiana e algoritmos MM
- Score rules e Ranked Probability Score
- Previsão probabilística e calibração

---

## Autor

**Guilherme Lopes** — [github.com/Guili-Lopes](https://github.com/Guili-Lopes)
