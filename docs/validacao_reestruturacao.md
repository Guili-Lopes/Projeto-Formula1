# Validação da Reestruturação — Etapas 2 a 8

Relatório consolidado da execução das Etapas 2–8 do plano
(`docs/reestruturacao_projeto_formula1.md`), com a comparação antes/depois
dos três pipelines históricos. A Etapa 9 (retomada do pipeline científico)
fica de fora por decisão do próprio plano e da decisão complementar nº 2.

## Método de validação

Cada pipeline migrado foi validado em **dois eixos independentes**:

1. **Migração de código** — execução do pipeline migrado com
   `source_policy: legacy_only` (mesmos CSVs de antes). Critério: métricas
   determinísticas idênticas ao baseline capturado da versão original;
   RPS dentro da tolerância de ruído Monte Carlo (±0,005 — o Monte Carlo
   histórico não usa seed, comportamento preservado).
2. **Troca de fonte de dados** — execução com `source_policy: prefer_openf1`
   (regra oficial: legado até 2022, OpenF1 2023+). Diferenças quantificadas
   e explicadas.

Os baselines foram capturados executando os pipelines **originais** neste
mesmo ambiente antes de qualquer modificação, e estão versionados em
`tests/regression/baselines/*.json`.

## Validação dos dados (Etapa 3)

Comparação legado × OpenF1 nas temporadas de transição
(detalhes em `docs/relatorio_validacao_dados.md`):

| Verificação | Resultado |
|---|---|
| Corridas comparadas (2023–2025) | 61 |
| Ordem dos classificados idêntica | **61/61** |
| Conjunto de DNFs idêntico | 60/61 |
| Ranking completo (incl. ordem da cauda de DNFs) idêntico | 51/61 |

As três diferenças estruturais encontradas, todas documentadas:

- **DNS omitidos pela OpenF1** — a API não lista pilotos que não largaram
  (ex.: STR em Singapura 2023), explicando a única divergência de conjunto
  de DNFs e as corridas com 19 pilotos (Singapura 2023, Austrália 2024,
  Espanha 2025).
- **Ordem da cauda de DNFs** — o loader histórico usa a ordem das linhas do
  CSV; a fonte OpenF1 ordena por voltas completadas (critério da
  classificação oficial). Afeta apenas a cauda de 10 corridas.
- **Legado 2025 parcial** — o CSV histórico de 2025 cobre 15 corridas (até
  Zandvoort); a OpenF1 cobre as **24** da temporada.

## Pipeline 1 — Mallows + Plackett–Luce

| Modo | Fase | Top-3 | Top-5 | Kendall τ |
|---|---|---:|---:|---:|
| Original (baseline) | val 2023 | 0,515 | 0,618 | 0,422 |
| Migrado `legacy_only` | val 2023 | **0,515** | **0,618** | **0,422** |
| Migrado `prefer_openf1` | val 2023 | 0,515 | 0,618 | 0,426 |
| Original (baseline) | teste 2024 | 0,500 | 0,583 | 0,408 |
| Migrado `legacy_only` | teste 2024 | **0,500** | **0,583** | **0,408** |
| Migrado `prefer_openf1` | teste 2024 | 0,500 | 0,575 | 0,408 |

Migração de código: **reprodução exata**. Troca de fonte: |Δ| ≤ 0,008,
consequência direta da ordem da cauda de DNFs propagada pelas atualizações
incrementais do estado.

## Pipeline Score Rules — Monte Carlo + RPS

Métricas determinísticas idênticas às do Pipeline 1 (mesmo split e seed).

| Modo | Fase | RPS modelo | RPS baseline | Ganho |
|---|---|---:|---:|---:|
| Original (baseline) | val 2023 | 0,1353 | 0,1899 | 0,0545 |
| Migrado `legacy_only` | val 2023 | 0,1356 | 0,1899 | 0,0543 |
| Migrado `prefer_openf1` | val 2023 | 0,1348 | 0,1899 | 0,0551 |
| Original (baseline) | teste 2024 | 0,1383 | 0,1898 | 0,0515 |
| Migrado `legacy_only` | teste 2024 | 0,1383 | 0,1898 | 0,0515 |
| Migrado `prefer_openf1` | teste 2024 | 0,1431 | 0,1898 | 0,0467 |

Migração de código: determinísticas exatas, RPS a ±0,0003 do baseline
(muito abaixo do ruído Monte Carlo). A configuração
`monte_carlo.seed: null` preserva o comportamento histórico; um inteiro
ativa seeds derivadas por corrida para RPS reprodutível.

## Pipeline 3 — OpenF1

| Modo | Fase | Corridas | Top-3 | Top-5 | Kendall τ | RPS | Ganho |
|---|---|---:|---:|---:|---:|---:|---:|
| Original (baseline) | val 2024 | 24 | 0,347 | 0,575 | 0,400 | 0,1271 | 0,0491 |
| Migrado `legacy_only` | val 2024 | 24 | **0,347** | **0,575** | **0,400** | 0,1271 | 0,0491 |
| Migrado `prefer_openf1` | val 2024 | 24 | 0,375 | 0,567 | 0,387 | 0,1227 | 0,0535 |
| Original (baseline) | teste 2025 | 15 | 0,533 | 0,667 | 0,405 | 0,1422 | 0,0347 |
| Migrado `legacy_only` | teste 2025 | 15 | **0,533** | **0,667** | **0,405** | 0,1422 | 0,0346 |
| Migrado `prefer_openf1` | teste 2025 | **24** | 0,514 | 0,658 | 0,443 | 0,1351 | 0,0414 |

Migração de código: reprodução exata (RPS a ±0,0001). No modo oficial, o
teste de 2025 passa de 15 para **24 corridas** — diferença de completude de
dados, não de modelo; os números de teste dos dois modos não são
comparáveis entre si. A validação 2024 muda levemente porque o treino passa
a incluir 2023 da OpenF1.

## Infraestrutura criada

- **Sincronizador central** (`src/data_openf1/sync.py`): única camada com
  acesso à API; perfis `core`/`full`; modos `--force-refresh`,
  `--incremental` e `--from-local-cache`; manifests por ano com status
  `completo`/`parcial`; layout `data/openf1/raw/<ano>/` +
  `processed/<tabela>/`.
- **Bloqueio de API nos pipelines**: `OPENF1_OFFLINE=1` no cliente; os três
  entry points ativam por padrão.
- **Carregador compartilhado** (`src/data/repository.py`): regra de fontes
  por período com políticas `prefer_openf1`/`strict_openf1`/`legacy_only`,
  proveniência por ano registrada nos manifests das execuções e rótulos
  históricos de corrida preservados (casamento por circuito do preditor).
- **Execuções imutáveis** (`src/experiments/run_store.py`):
  `artifacts/<pipeline>/<run_id>/` com config resolvida, manifest (commit,
  dependências), métricas, tabelas Parquet + espelho CSV, plots, `run.log`,
  `runtime.json`, `error.json` em falha e ponteiro `latest_run.json`.
- **Configuração por YAML** (`configs/default.yaml` por pipeline),
  preservando o split histórico de cada um (decisão complementar nº 3).
- **Notebooks 05–07** lendo da última execução via `latest_run.json`
  (Parquet/JSON preferenciais; PKL auxiliar); notebooks 00–07 executados de
  ponta a ponta.
- **Suíte de testes nova** (72 testes, pytest): `unit/`, `data_quality/`
  (verificações da Etapa 3 sobre os dados versionados), `integration/`,
  `e2e/` (pipelines completos sobre dados sintéticos, sem tocar em `data/`
  nem na API), `regression/` (baselines com tolerância) e `smoke/`. O teste
  `slow` reexecuta o P1 completo e confirma a reprodução exata do baseline.
  Os testes do pipeline científico foram movidos para
  `src/pipeline_cientifico_comparativo/tests/` (serão retomados na Etapa 9).

## Critérios de conclusão (seção 13 do plano)

| Critério | Status |
|---|---|
| Sincronizador central único com perfis e manifests | ✅ |
| Nenhum pipeline chama a API diretamente | ✅ (`OPENF1_OFFLINE=1`) |
| Regra de fontes (legado ≤2022, OpenF1 2023+) | ✅ |
| Dados 2023–2025 armazenados e processados | ✅ |
| Dados 2026 sincronizados | ⏳ requer rede — comando pronto (abaixo) |
| `artifacts/` imutável + `latest_run.json` + `error.json` | ✅ |
| YAML por pipeline com split histórico preservado | ✅ |
| README por pipeline + README índice | ✅ |
| Notebooks lendo de artifacts (Parquet/JSON) | ✅ |
| `tests/` nova cobrindo módulos e pipelines | ✅ (72 testes) |
| Resultados reproduzem os originais | ✅ (validação em dois eixos) |

## Pendências para o autor do projeto

1. **Fazer merge do PR da Etapa 1 (#3)** antes deste PR (este inclui os
   commits daquele até o merge).
2. **Sincronização online** (o ambiente desta execução não tinha acesso à
   API — tudo foi populado do cache local):
   ```bash
   python -m src.data_openf1.sync --start-year 2026 --end-year 2026 --incremental
   python -m src.data_openf1.sync --start-year 2023 --end-year 2025 --incremental
   ```
3. **Revogar o token do GitHub** usado nesta sessão
   (Settings → Developer settings → Personal access tokens).
4. **Convivência do legado** (plano, seção 12): `data/Season2023–2025/` e o
   cache plano `data/openf1/raw/*.csv` seguem no repositório como referência
   temporária; remover em um PR futuro após o período de validação.
5. **Etapa 9**: retomada do pipeline científico comparativo (M0/M1/M2 serão
   redefinidos), agora sobre a base reestruturada.
