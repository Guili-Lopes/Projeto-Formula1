# Inventário — Etapa 1 da Reestruturação

**Objetivo:** mapear arquivos, pipelines, notebooks, dependências e dados do
repositório antes de qualquer migração, conforme a Etapa 1 do plano
(`reestruturacao_projeto_formula1.md`).

**Base do inventário:** branch `main`, commit `659ae5c`
("Merge pull request #1 from Guili-Lopes/feature/fase-0-pipeline-cientifico").

**Regra desta etapa:** nenhum arquivo é excluído.

---

## 1. Módulos compartilhados em `src/`

| Módulo | Arquivos | Observações |
|---|---|---|
| `src/data/` | `data_pipeline.py` | Carregamento dos CSVs históricos e construção de `RaceRecord` |
| `src/data_openf1/` | `client.py`, `cache.py`, `schema.py`, `race_mapping.py`, `feature_builder.py`, `coverage_report.py`, `__init__.py` | Única origem do import de `requests` (`client.py:24`) |
| `src/models/` | `models_mallows.py`, `models_plackett_luce.py`, `models_weights.py` | Mallows, Plackett–Luce e pesos |
| `src/engine/` | `engine_trainer.py`, `engine_predictor.py` | Treino incremental e previsão |
| `src/evaluation/` | `evaluation_metrics.py`, `additional_metrics.py` | Métricas determinísticas e adicionais |
| `src/experiments/` | `artifact_store.py`, `manifest.py`, `reproducibility.py`, `console_capture.py`, `experiment_result.py`, `__init__.py` | Infraestrutura de artefatos e reprodutibilidade |
| `src/Teste MCMC/` | `MCMC_ClusteringPartialRankings.py` | Experimento histórico (notebook 04); pasta com espaço no nome |

## 2. Pipelines e pontos de entrada

| Pipeline | Ponto de entrada | Outros módulos | Estado na reorganização |
|---|---|---|---|
| `pipeline_mallows_plackett_luce` | `run_experiment.py` | `visualization_plots.py` | 1º a migrar (Etapa 4) |
| `pipeline_score_rules` | `run_pipeline_score_rules.py` | `monte_carlo.py`, `scoring_rules.py`, `visualization_plots_p2.py` | 2º a migrar (Etapa 5) |
| `pipeline_openf1` | `run_pipeline_openf1.py` | `visualization_plots_p3.py`, `__init__.py` | 3º a migrar (Etapa 6) |
| `pipeline_cientifico_comparativo` | `run_phase.py`, `run_all.py` | `config_loader.py`, `config_schema.py`, `registry.py`, `adapters/` (5 arquivos), `phases/phase_00_baseline.py`, `aggregation/collect_results.py`, `configs/` (2 YAMLs) | **Congelado** — fora da migração inicial; será refeito na Etapa 9 |

Configurações YAML existentes (apenas no pipeline científico):
`src/pipeline_cientifico_comparativo/configs/common.yaml` e
`configs/m0_baseline.yaml`. Os três pipelines históricos **não possuem YAML**
(parâmetros fixos no código) — alvo das Etapas 4–6.

## 3. Referências a `outputs/` e `.pkl` no código

Arquivos que referenciam `outputs/` (destino atual dos resultados; serão
migrados para `artifacts/`):

- `src/experiments/artifact_store.py`
- `src/pipeline_mallows_plackett_luce/run_experiment.py`
- `src/pipeline_openf1/run_pipeline_openf1.py`
- `src/pipeline_score_rules/run_pipeline_score_rules.py`

Arquivos que geram/consomem `.pkl`:

- `src/pipeline_mallows_plackett_luce/run_experiment.py` (`nb_data.pkl`)
- `src/pipeline_score_rules/run_pipeline_score_rules.py`
- `src/pipeline_openf1/run_pipeline_openf1.py` (`nb_data_p3.pkl`)
- `src/pipeline_cientifico_comparativo/phases/phase_00_baseline.py`

## 4. Notebooks

| Notebook | Depende de `.pkl`? |
|---|---|
| `00_projeto_overview.ipynb` | Não |
| `01_F1_regras_basicas.ipynb` | Não |
| `02_data_sources.ipynb` | Não |
| `03_modelo_algoritmo.ipynb` | Não |
| `04_MCMC_ClusteringPartialRankings.ipynb` | Não |
| `05_resultados_analise_pipeline1_Mallows_PlackettLuce.ipynb` | **Sim** |
| `06_resultados_analise_pipeline_score_rules.ipynb` | **Sim** |
| `07_resultado_analise_pipeline3_openf1.ipynb` | **Sim** |

Nota: o notebook 07 **existe** com o nome acima, resolvendo a incerteza
registrada na documentação principal (seção 17).

Os notebooks 05–07 são o alvo da Etapa 7 (leitura de Parquet/JSON/YAML em vez
de dependência exclusiva de PKL).

## 5. Testes atuais

Cinco arquivos em `tests/`, todos ligados à infraestrutura do pipeline
científico comparativo:

- `test_config_loader.py`
- `test_artifact_store.py`
- `test_reproducibility.py`
- `test_additional_metrics.py`
- `test_pipeline_cientifico_phase0.py`

Conforme decisão registrada (`decisoes_complementares_reestruturacao.md`, §4),
a pasta `tests/` será refeita do zero para módulos compartilhados e pipelines
históricos; os casos do pipeline científico retornarão na Etapa 9. Nada é
excluído nesta etapa.

## 6. Dependências — declaradas × efetivamente usadas

Auditoria de imports em `src/` e `notebooks/`:

| Dependência declarada | Importada em `src/`? | Importada em notebooks? | Situação |
|---|---|---|---|
| numpy | Sim (18×) | Sim | OK |
| pandas | Sim (11×) | Sim (nb 07) | OK |
| matplotlib | Sim (9×) | Sim (nb 05–07) | OK |
| PyYAML (`yaml`) | Sim (3×) | Não | OK |
| pyarrow | Não (uso implícito) | Não | OK — backend Parquet do pandas |
| jupyter / notebook / ipykernel | — | — | Ferramentas de ambiente (IPython usado no nb 07) |
| **requests** | **Sim (`client.py`)** | Não | **Não estava declarada — corrigido neste PR** |
| scipy | Não | Não | Candidata a avaliação futura |
| seaborn | Não | Não | Candidata a avaliação futura |
| scikit-learn | Não | Não | Candidata a avaliação futura |
| torch | Não | Não | Candidata a avaliação futura |
| torchvision | Não | Não | Candidata a avaliação futura |
| tqdm | Não | Não | Candidata a avaliação futura |

Conforme decisão §5(b), nenhuma dependência é removida agora; a coluna
"candidata a avaliação futura" implementa a Prioridade 3 da documentação
principal ("Avaliar remoção futura de dependências sem uso efetivo").

## 7. Dados históricos (`data/Season*`)

| Temporada | Arquivos | Observações |
|---|---|---|
| 2019 | 3 | `drivers`, `raceResults`, `tracks` — sem `calendar`; prefixo `formula1_` (minúsculo) |
| 2020 | 3 | `calendar`, `drivers`, `raceResults`; prefixo minúsculo |
| 2021 | 5 | + `sprintQualifyingResults`, `teams`; prefixo minúsculo |
| 2022 | 7 | + `qualifyingResults`, `sprintRaceResults`, `driverOfTheDayVotes`; prefixo `Formula1_` (maiúsculo) |
| 2023 | 8 | + `sprintResults`, `sprintShootoutResults` |
| 2024 | 8 | `sprintQualifyingResults` + `sprintResults` |
| 2025 | 4 | Apenas `RaceResults`, `QualifyingResults`, `SprintResults`, `SprintQualifyingResults` — **sem `drivers`, `calendar` e `teams`**; padrão `2025Season` (S maiúsculo) |

Pontos de atenção para o carregador compartilhado (Etapa 2) e para a
validação (Etapa 3):

1. inconsistência de caixa nos prefixos (`formula1_` × `Formula1_`) e no
   padrão de 2025 (`2025Season`);
2. conjuntos de arquivos diferentes por temporada (2019 sem calendário;
   2025 sem cadastro de pilotos/equipes);
3. nomenclatura dos arquivos de sprint muda ao longo dos anos
   (`sprintQualifying` → `sprintRace` → `sprint` + `sprintShootout`).

## 8. Dados OpenF1 (`data/openf1/`)

- `raw/`: **357 arquivos CSV** de cache, por sessão, cobrindo 6 tipos de
  consulta: `drivers`, `meetings`, `race_control`, `session_result`,
  `sessions_meeting`, `starting_grid`. Estrutura plana (sem partição por ano)
  — será reorganizada na Etapa 2 (`raw/<ano>/`).
- `processed/`: 2 arquivos — `openf1_coverage_report_2024_2025.csv` e
  `race_context_2023_2025.csv`.
- Não há `manifests/` nem `README.md` — serão criados na Etapa 2.

## 9. Resultados atuais registrados como referência

Referência de equivalência do M0 (validação 2024), conforme documentação
principal e artefatos da Fase 0:

| Métrica | Valor |
|---|---:|
| Corridas | 24 |
| Top-3 médio | 0,347 |
| Top-5 médio | 0,575 |
| Kendall tau médio | 0,400 |
| RPS médio do modelo | 0,1270 |
| RPS médio do baseline | 0,1762 |
| Ganho médio absoluto | 0,0492 |

Os *baselines* de regressão dos pipelines históricos (Pipeline 1, Score Rules
e OpenF1) serão capturados em suas respectivas etapas de migração
(Etapas 4–6), executando cada pipeline **antes** da migração e registrando as
saídas como referência de comparação.

## 10. Infraestrutura do repositório

- `.gitignore` já ignora `artifacts/` (correto: execuções não devem ser
  versionadas). `outputs/` não estava ignorado — corrigido neste PR como
  medida de proteção, evitando o versionamento acidental de resultados
  legados durante a migração.
- Não existe `pytest.ini` — os testes atuais usam `unittest`. A adoção de
  `pytest` acompanhará a reconstrução de `tests/` (seção 9 do plano).
- Não existia a pasta `docs/` — criada neste PR para abrigar o plano de
  reestruturação, as decisões complementares e este inventário.

## 11. Itens protegidos nesta etapa

Permanecem intocados, conforme seção 12 do plano:

`data/Season2023–2025`, caches atuais de `data/openf1/raw/`, arquivos PKL
gerados pelos pipelines, notebooks que leem PKL, todo o código do
`pipeline_cientifico_comparativo` e os testes atuais em `tests/`.

---

**Próximo passo (Etapa 2):** centralização da OpenF1 — sincronizador oficial,
perfis `core`/`full`, partição por ano, manifests, tabelas processadas com
campos de status, carregador compartilhado e bloqueio de chamadas diretas à
API pelos pipelines.
