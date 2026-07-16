# Pipeline 3 — Mallows + Plackett–Luce + contexto OpenF1

## Objetivo do pipeline

Pipeline probabilístico (Monte Carlo + RPS, como o Pipeline Score Rules) com
split deslocado uma temporada à frente e enriquecido com **features
contextuais da OpenF1** por corrida: grid de largada (`grid_<SIGLA>`), DNFs
observados (`dnf_<SIGLA>`) e eventos de prova (safety car, virtual safety
car, bandeiras vermelhas e amarelas). O contexto é pós-corrida e **não entra
na previsão** — alimenta a análise do notebook 07.

## Como executar

```bash
python -m src.pipeline_openf1.run_pipeline_openf1
python -m src.pipeline_openf1.run_pipeline_openf1 --config caminho/config.yaml
```

## Configuração utilizada

`configs/default.yaml` — split histórico deste pipeline: treino 2019–2023,
validação 2024, teste 2025; `monte_carlo` idêntico ao Score Rules
(10.000 simulações, seed `null` preserva o comportamento histórico);
bloco `context` com `first_year: 2023` e `allow_partial: true`.

## Dados utilizados

- **Rankings**: repositório compartilhado (`src/data/repository.py`) —
  legado até 2022, OpenF1 2023+.
- **Contexto**: `data/openf1/processed/` (tabela `race_context_2023_2025`,
  regenerável com `python -m src.data_openf1.sync ... --rebuild-context`).
- O pipeline roda com `OPENF1_OFFLINE=1`: **nenhuma chamada à API** — este
  era o único pipeline que antes podia disparar chamadas via cache; agora
  consome exclusivamente dados sincronizados em disco.

## Resultados esperados

| Modo | Fase | N | Top-3 | Top-5 | Kendall τ | RPS modelo | Ganho |
|---|---|---:|---:|---:|---:|---:|---:|
| legacy_only (baseline) | val 2024 | 24 | 0,347 | 0,575 | 0,400 | ~0,1271 | ~0,0491 |
| legacy_only (baseline) | teste 2025 | **15** | 0,533 | 0,667 | 0,405 | ~0,1422 | ~0,0347 |
| prefer_openf1 (oficial) | val 2024 | 24 | 0,375 | 0,567 | 0,387 | ~0,1227 | ~0,0535 |
| prefer_openf1 (oficial) | teste 2025 | **24** | 0,514 | 0,658 | 0,443 | ~0,1351 | ~0,0414 |

**Atenção à diferença de completude**: o CSV legado de 2025 cobria apenas 15
corridas (até Zandvoort); a fonte OpenF1 cobre as **24 corridas** da
temporada. As métricas de teste dos dois modos, portanto, **não são
comparáveis diretamente** — a diferença vem dos dados, não do código
(a migração foi validada em `legacy_only` com métricas idênticas ao
baseline). As pequenas variações na validação 2024 (≤0,03) vêm da ordenação
da cauda de DNFs na fonte OpenF1 propagada pelo treino incremental
(`docs/relatorio_validacao_dados.md`).

## Saídas geradas

`artifacts/pipeline_openf1/<run_id>/` com: `config_resolved.yaml`,
`manifest.json`, `metrics_summary.json`, `race_metrics.parquet`,
`rps_metrics.parquet` (inclui colunas de contexto por corrida),
`predictions.parquet`, `position_probabilities.parquet`,
`race_context.parquet`, `grid_vs_finish.parquet`,
`openf1_coverage_report.parquet`, `dnf_analysis.json`, `nb_data_p3.pkl`
(auxiliar), `run.log`, `runtime.json`, `plots/viz*.png` e ponteiro
`latest_run.json` na raiz do pipeline. Falhas geram `error.json`.

## Testes

- `tests/pipelines/openf1/` — configuração e execução reduzida
- `tests/regression/` — comparação com baselines (RPS com tolerância)

## Notebook relacionado

`notebooks/07_resultados_analise_pipeline_openf1.ipynb`
(lê os artefatos da última execução via `latest_run.json`).
