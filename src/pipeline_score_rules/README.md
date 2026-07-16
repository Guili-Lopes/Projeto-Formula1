# Pipeline Score Rules — Monte Carlo + RPS

## Objetivo do pipeline

Estender o Pipeline 1 com previsão probabilística: para cada corrida, gera o
vetor de probabilidades de cada piloto em cada posição via Monte Carlo
(10.000 simulações sobre os skill scores do Plackett–Luce do cluster
previsto) e avalia com o Ranked Probability Score (RPS) contra um baseline
uniforme.

## Como executar

```bash
python -m src.pipeline_score_rules.run_pipeline_score_rules
python -m src.pipeline_score_rules.run_pipeline_score_rules --config caminho/config.yaml
```

## Configuração utilizada

`configs/default.yaml` — split histórico deste pipeline: treino 2019–2022,
validação 2023, teste 2024; `n_simulations` 10000, `n_positions` 20,
`monte_carlo.seed: null` (preserva o comportamento histórico não
determinístico — comparações de RPS usam tolerância; defina um inteiro para
ativar seeds derivadas por corrida e RPS reprodutível).

## Dados utilizados

Idênticos ao Pipeline 1: repositório compartilhado com regra de fontes
(legado até 2022, OpenF1 2023+), `OPENF1_OFFLINE=1` por padrão.

## Resultados esperados

| Modo | Fase | Top-3 | Kendall τ | RPS modelo | RPS baseline | Ganho |
|---|---|---:|---:|---:|---:|---:|
| legacy_only (baseline) | val 2023 | 0,515 | 0,422 | ~0,1353 | 0,1899 | ~0,0545 |
| legacy_only (baseline) | teste 2024 | 0,500 | 0,408 | ~0,1383 | 0,1898 | ~0,0515 |
| prefer_openf1 (oficial) | val 2023 | 0,515 | 0,426 | ~0,1348 | 0,1899 | ~0,0551 |
| prefer_openf1 (oficial) | teste 2024 | 0,500 | 0,408 | ~0,1431 | 0,1898 | ~0,0467 |

Valores de RPS oscilam ±0,005 entre execuções por conta do Monte Carlo sem
seed (comportamento histórico preservado). As métricas determinísticas em
`legacy_only` reproduzem o baseline exatamente.

## Saídas geradas

`artifacts/pipeline_score_rules/<run_id>/` com: `config_resolved.yaml`,
`manifest.json`, `metrics_summary.json`, `race_metrics.parquet`,
`rps_metrics.parquet`, `predictions.parquet`,
`position_probabilities.parquet` (formato longo: fase × corrida × piloto ×
posição × probabilidade), `nb_data_p2.pkl` (auxiliar), `run.log`,
`runtime.json`, `plots/viz*.png` e ponteiro `latest_run.json` na raiz do
pipeline. Falhas geram `error.json` na pasta da execução.

## Testes

- `tests/pipelines/score_rules/` — configuração e execução reduzida
- `tests/regression/` — RPS comparado com tolerância

## Notebook relacionado

`notebooks/06_resultados_analise_pipeline_score_rules.ipynb`
(lê os artefatos da última execução via `latest_run.json`).
