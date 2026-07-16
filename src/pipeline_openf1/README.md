# Pipeline 3 — Mallows + Plackett–Luce + contexto OpenF1

## Objetivo do pipeline

Pipeline probabilístico (Monte Carlo + RPS, como o Score Rules) com split
mais recente e **features contextuais da OpenF1** por corrida: grid de
largada (`grid_<SIGLA>`), DNFs observados (`dnf_<SIGLA>`) e eventos de prova
(safety car, virtual safety car, bandeiras vermelhas e amarelas). O contexto
é pós-corrida e **não** entra na previsão — alimenta a análise do notebook 07.

## Como executar

```bash
python -m src.pipeline_openf1.run_pipeline_openf1
python -m src.pipeline_openf1.run_pipeline_openf1 --config caminho/config.yaml
```

## Configuração utilizada

`configs/default.yaml` — split histórico deste pipeline: treino 2019–2023,
validação 2024, teste 2025; `monte_carlo` idêntico ao Score Rules
(10.000 simulações, seed null preserva o comportamento histórico);
bloco `context` com `first_year: 2023` e `allow_partial: true`.

## Dados utilizados

Rankings pelo repositório compartilhado (legado até 2022, OpenF1 2023+) e
contexto pelas tabelas processadas de `data/openf1/` — o pipeline roda com
`OPENF1_OFFLINE=1` e **nunca chama a API**; dados ausentes são obtidos com
`python -m src.data_openf1.sync`.

> **Nota da reestruturação (teste 2025):** o CSV legado de 2025 era parcial
> (15 corridas, até Zandvoort). Com a OpenF1 como fonte oficial, o teste
> passa a cobrir as **24 corridas** da temporada. As métricas de teste dos
> dois modos abaixo não são comparáveis entre si por cobrirem conjuntos
> diferentes de corridas — a diferença é de completude de dados, não de
> modelo. Ver `docs/relatorio_validacao_dados.md`.

## Resultados esperados

| Modo | Fase | Corridas | Top-3 | Top-5 | Kendall τ | RPS modelo | Ganho |
|---|---|---:|---:|---:|---:|---:|---:|
| legacy_only (baseline) | val 2024 | 24 | 0,347 | 0,575 | 0,400 | ~0,1271 | ~0,0491 |
| legacy_only (baseline) | teste 2025 | 15 | 0,533 | 0,667 | 0,405 | ~0,1422 | ~0,0347 |
| prefer_openf1 (oficial) | val 2024 | 24 | 0,375 | 0,567 | 0,387 | ~0,1227 | ~0,0535 |
| prefer_openf1 (oficial) | teste 2025 | **24** | 0,514 | 0,658 | 0,443 | ~0,1351 | ~0,0414 |

Em `legacy_only` o pipeline reproduz o baseline pré-reestruturação
(determinísticas idênticas; RPS a ±0,0001). Em `prefer_openf1`, além do
teste ampliado, a validação 2024 muda levemente porque o treino passa a
incluir 2023 da OpenF1 (ordem da cauda de DNFs propagada pelo estado
incremental). RPS oscila ±0,005 entre execuções (Monte Carlo sem seed).

## Saídas geradas

`artifacts/pipeline_openf1/<run_id>/` com: `config_resolved.yaml`,
`manifest.json`, `metrics_summary.json`, `race_metrics.parquet`,
`rps_metrics.parquet` (inclui colunas de contexto por corrida),
`predictions.parquet`, `position_probabilities.parquet`,
`race_context.parquet`, `grid_vs_finish.parquet`,
`openf1_coverage_report.parquet`, `dnf_analysis.json`, `nb_data_p3.pkl`
(auxiliar), `run.log`, `runtime.json`, `plots/viz*.png` e ponteiro
`latest_run.json`. Falhas geram `error.json` na pasta da execução.

## Testes

- `tests/pipelines/openf1/` — configuração e execução reduzida
- `tests/regression/` — comparação com baselines (RPS com tolerância)

## Notebook relacionado

`notebooks/07_resultados_analise_pipeline_openf1.ipynb`
(lê os artefatos da última execução via `latest_run.json`).
