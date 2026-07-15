# Pipeline 1 — Mallows + Plackett–Luce

## Objetivo do pipeline

Modelar cada corrida de F1 como um ranking parcial: clusterização de corridas
via modelo de Mallows (distância de Kendall) e habilidade dos pilotos via
Plackett–Luce ponderado (pesos temporais e regulatórios), com atualização
incremental corrida a corrida e avaliação determinística (Top-3, Top-5,
Kendall τ).

## Como executar

A partir da raiz do projeto:

```bash
python -m src.pipeline_mallows_plackett_luce.run_experiment
# ou com configuração alternativa:
python -m src.pipeline_mallows_plackett_luce.run_experiment --config caminho/config.yaml
```

## Configuração utilizada

`configs/default.yaml` — preserva o **split histórico** deste pipeline
(decisão complementar nº 3):

| Bloco | Valores |
|---|---|
| splits | treino 2019–2022 · validação 2023 · teste 2024 |
| model | n_clusters 2 · n_iter 150 · alpha 0.5 |
| data | top_k 10 · source_policy `prefer_openf1` |
| seed | 42 |

## Dados utilizados

Carregados pelo repositório compartilhado (`src/data/repository.py`):
até 2022 pelo dataset histórico (`data/Season<ano>/`), 2023+ pelas tabelas
processadas da OpenF1 (`data/openf1/processed/race_results/`). O pipeline
roda com `OPENF1_OFFLINE=1`: nenhuma chamada à API é feita — dados ausentes
devem ser sincronizados com `python -m src.data_openf1.sync`.

Com `source_policy: legacy_only` o comportamento pré-reestruturação é
reproduzido exatamente (validado na migração: métricas idênticas ao baseline).

## Resultados esperados

| Modo | Fase | Top-3 | Top-5 | Kendall τ |
|---|---|---:|---:|---:|
| legacy_only (baseline) | validação 2023 | 0,515 | 0,618 | 0,422 |
| legacy_only (baseline) | teste 2024 | 0,500 | 0,583 | 0,408 |
| prefer_openf1 (oficial) | validação 2023 | 0,515 | 0,618 | 0,426 |
| prefer_openf1 (oficial) | teste 2024 | 0,500 | 0,575 | 0,408 |

As pequenas diferenças (≤0,008) vêm da ordenação da cauda de DNFs na fonte
OpenF1 (por voltas completadas), documentada em
`docs/relatorio_validacao_dados.md`.

## Saídas geradas

Cada execução cria uma pasta imutável em
`artifacts/pipeline_mallows_plackett_luce/<run_id>/` com:
`config_resolved.yaml`, `manifest.json`, `metrics_summary.json`,
`race_metrics.parquet` (+ `.csv`), `predictions.parquet`,
`skill_history.parquet`, `regulatory_weights.parquet`, `nb_data.pkl`
(formato auxiliar para o notebook), `run.log`, `runtime.json` e
`plots/viz*.png`. O ponteiro `latest_run.json` na raiz do pipeline aponta
para a última execução bem-sucedida; falhas geram `error.json` na pasta da
execução sem mover o ponteiro.

## Testes

- `tests/pipelines/mallows_plackett_luce/` — configuração e execução reduzida
- `tests/regression/` — comparação com os baselines da migração

## Notebook relacionado

`notebooks/05_resultados_analise_pipeline1_Mallows_PlackettLuce.ipynb`
(lê os artefatos da última execução via `latest_run.json`).
