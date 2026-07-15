# Camada de dados OpenF1

Esta pasta é a **única origem de dados OpenF1** do projeto. Nenhum pipeline
chama a API diretamente; a sincronização é centralizada em
`src/data_openf1/sync.py` e o consumo é feito pelo carregador compartilhado
`src/data/repository.py`.

```
OpenF1 API ──► sincronizador central ──► raw/<ano>/ ──► processed/ ──► pipelines
                (src.data_openf1.sync)                     ▲
                                                           └── manifests/
```

## Regra de fonte oficial por período

| Período | Fonte |
|---|---|
| até 2022 | dataset histórico (`data/Season<ano>/`) |
| 2023 em diante | dados OpenF1 desta pasta |

Os CSVs históricos de 2023–2025 foram preservados em
`data/legacy/reference_2023_2025/` como referência da transição
(ver `docs/relatorio_validacao_dados.md`).

## Estrutura

```
data/openf1/
├── raw/
│   ├── <ano>/<endpoint>/<endpoint>_<chave>.csv   # layout oficial (por ano)
│   └── *.csv                                     # cache plano legado (temporário)
├── processed/
│   └── <tabela>/<tabela>_<ano>.parquet (+ .csv)  # tabelas padronizadas
├── manifests/
│   └── sync_<ano>.json                           # relatório de cada sincronização
└── README.md
```

Tabelas processadas atuais: `meetings`, `sessions`, `drivers`,
`race_results`, `sprint_results`, `qualifying_results`, `starting_grid`,
`race_control`, `race_control_summary` (SC/VSC/bandeiras) e, quando o raw
existir, `weather`, `pit`, `stints`, `laps`, `position`, `intervals`,
`championship_drivers`, `championship_teams`.

`race_results` captura desde já os campos de status da API (posição oficial,
`classified`, `dnf`, `dns`, `dsq`, `number_of_laps`, `gap_to_leader`,
`points`, piloto por sigla e equipe), mesmo que o modelo atual ainda não use
todos — decisão complementar 5c da reestruturação.

## Como sincronizar

```bash
# Sincronização completa de um intervalo (rede necessária)
python -m src.data_openf1.sync --start-year 2023 --end-year 2026 --profile core

# Perfil full (adiciona laps, position, intervals — alto volume)
python -m src.data_openf1.sync --start-year 2024 --end-year 2024 --profile full

# Atualização incremental (baixa somente o que falta; ideal para 2026)
python -m src.data_openf1.sync --start-year 2026 --end-year 2026 --incremental

# Forçar atualização de um ano específico
python -m src.data_openf1.sync --start-year 2025 --end-year 2025 --force-refresh

# Importar o cache local legado sem rede (usado na reestruturação)
python -m src.data_openf1.sync --start-year 2023 --end-year 2025 --from-local-cache

# Regenerar a tabela de contexto de corrida (grid, DNF, SC/VSC/bandeiras)
python -m src.data_openf1.sync --start-year 2023 --end-year 2025 --rebuild-context
```

Perfis: `core` = drivers, session_result, starting_grid, race_control,
weather, pit, stints e classificações de campeonato; `full` = core + laps,
position, intervals. Endpoints de altíssimo volume (car_data, location,
team_radio) ainda não são expostos pelo cliente e estão registrados nos
manifests como `unsupported_endpoints`.

## Manifests

Cada execução grava `manifests/sync_<ano>.json` com: data/hora, modo
(online/offline), perfil, endpoints consultados, registros por endpoint,
corridas encontradas × com resultado, arquivos criados/atualizados/reutilizados,
respostas vazias, erros e o **status da temporada** (`completo`/`parcial`).
Temporadas em andamento (ex.: 2026) permanecem `parcial` e podem ser
atualizadas com `--incremental`.

## Modo offline

Com `OPENF1_OFFLINE=1`, qualquer chamada à API é bloqueada no cliente
(`src/data_openf1/client.py`) e registrada em log. Os pipelines migrados
executam com esse modo ativado por padrão: se um dado não estiver em disco,
a resposta correta é rodar o sincronizador, nunca chamar a API do pipeline.
