# Reestruturação Geral do Projeto Fórmula 1

## 1. Objetivo

Este documento define a reorganização da arquitetura, dos dados, dos pipelines, dos artefatos, dos notebooks e dos testes do projeto Fórmula 1.

O objetivo é preparar uma base mais organizada, reproduzível e confiável antes de continuar o desenvolvimento do novo pipeline científico comparativo.

A prioridade será primeiro organizar e estabilizar o projeto existente. Somente depois dessa etapa o desenvolvimento do novo pipeline será retomado.

---

## 2. Decisão sobre o novo pipeline

As mudanças já implementadas para o novo pipeline serão mantidas por enquanto.

Não será feita uma remoção imediata do código já criado. Quando o desenvolvimento do novo pipeline for retomado, cada componente será revisado para decidir:

- o que será reaproveitado;
- o que será adaptado;
- o que será renomeado;
- o que será descartado;
- o que deverá ser reconstruído.

As nomeações provisórias como `M0`, `M1` e `M2` não serão consideradas definitivas. A nomenclatura futura deverá utilizar nomes descritivos, baseados no comportamento de cada modelo.

Durante a reorganização atual, a pasta geral `tests/` não deverá conter testes relacionados ao `pipeline_cientifico_comparativo`.

---

# 3. Princípios da reestruturação

A reorganização seguirá os seguintes princípios:

1. Uma única camada centralizada para leitura e armazenamento dos dados da OpenF1.
2. Nenhum pipeline deverá depender exclusivamente de arquivos `.pkl`.
3. Todos os pipelines deverão salvar resultados em `artifacts/`.
4. Todos os pipelines deverão possuir configuração em YAML.
5. Todos os pipelines deverão possuir um `README.md`.
6. A pasta `tests/` deverá testar módulos compartilhados e pipelines existentes.
7. Os notebooks deverão consumir artefatos estruturados.
8. O código compartilhado deverá ser reutilizado entre pipelines.
9. Mudanças serão feitas gradualmente, sem apagar fluxos antigos antes da validação.
10. O novo pipeline somente será retomado após a estabilização do projeto.

---

# 4. Nova estratégia de dados

## 4.1 Fonte oficial por período

A regra planejada será:

```text
Até 2022:
usar o dataset histórico atualmente existente no projeto.

A partir de 2023:
usar somente os dados obtidos da OpenF1.
```

Isso significa que os pipelines antigos e futuros deverão utilizar a mesma camada de acesso aos dados.

Os pipelines não deverão decidir diretamente qual arquivo carregar. Essa responsabilidade ficará em um módulo compartilhado de dados.

Exemplo conceitual:

```python
from src.data.repository import load_race_results

records = load_race_results(year=2025)
```

Internamente, o carregador decidirá:

```python
def load_race_results(year):
    if year <= 2022:
        return load_legacy_results(year)

    return load_openf1_results(year)
```

---

## 4.2 Pasta central da OpenF1

Os dados obtidos da API serão armazenados em:

```text
data/openf1/
```

O nome em letras minúsculas deve ser mantido para evitar diferenças entre sistemas operacionais, Git, Linux, Docker e automações.

Estrutura proposta:

```text
data/
├── Season2019/
├── Season2020/
├── Season2021/
├── Season2022/
│
├── legacy/
│   └── reference_2023_2025/
│
└── openf1/
    ├── raw/
    │   ├── 2023/
    │   ├── 2024/
    │   ├── 2025/
    │   └── 2026/
    │
    ├── processed/
    │   ├── meetings/
    │   ├── sessions/
    │   ├── drivers/
    │   ├── race_results/
    │   ├── qualifying_results/
    │   ├── starting_grid/
    │   ├── laps/
    │   ├── stints/
    │   ├── pit/
    │   ├── intervals/
    │   ├── positions/
    │   ├── race_control/
    │   ├── weather/
    │   ├── overtakes/
    │   ├── championship_drivers/
    │   ├── championship_teams/
    │   ├── car_data/
    │   ├── location/
    │   └── team_radio/
    │
    ├── manifests/
    │   ├── sync_2023.json
    │   ├── sync_2024.json
    │   ├── sync_2025.json
    │   └── sync_2026.json
    │
    └── README.md
```

---

## 4.3 Script central de sincronização

Será criado um script oficial para buscar e armazenar os dados da OpenF1.

Exemplo de execução completa:

```bash
python -m src.data_openf1.sync \
  --start-year 2023 \
  --end-year 2026 \
  --profile full
```

Exemplo para atualizar apenas 2025:

```bash
python -m src.data_openf1.sync \
  --start-year 2025 \
  --end-year 2025 \
  --force-refresh
```

Exemplo para atualização incremental de 2026:

```bash
python -m src.data_openf1.sync \
  --start-year 2026 \
  --end-year 2026 \
  --incremental
```

A temporada de 2026 deverá ser marcada como parcial enquanto ainda estiver em andamento.

---

## 4.4 Perfis de sincronização

Como alguns endpoints podem gerar grandes volumes de dados, serão definidos dois perfis:

```text
core:
baixa os dados essenciais para os pipelines atuais.

full:
baixa todos os dados disponíveis e necessários para a base completa.
```

O perfil `core` poderá incluir:

- meetings;
- sessions;
- drivers;
- session results;
- starting grid;
- race control;
- weather;
- pit stops;
- stints;
- championship standings.

O perfil `full` poderá incluir também:

- laps;
- positions;
- intervals;
- car data;
- location;
- team radio;
- demais endpoints suportados.

Dados maiores poderão ser particionados por temporada, sessão e piloto.

Exemplo:

```text
data/openf1/raw/2025/car_data/session_9839/driver_44.parquet
data/openf1/raw/2025/location/session_9839/driver_44.parquet
```

---

## 4.5 Regra de acesso à API

Após essa mudança:

> Nenhum pipeline deverá chamar a API OpenF1 diretamente.

Somente a camada de sincronização poderá acessar a API.

Os pipelines deverão ler exclusivamente os dados já armazenados em `data/openf1/`.

Fluxo esperado:

```text
OpenF1 API
    ↓
sincronizador central
    ↓
data/openf1/raw/
    ↓
processamento e padronização
    ↓
data/openf1/processed/
    ↓
carregador compartilhado
    ↓
pipelines
```

---

## 4.6 Manifests de sincronização

Cada sincronização deverá gerar um arquivo de manifesto contendo:

- anos processados;
- data e hora da execução;
- endpoints consultados;
- quantidade de registros;
- sessões encontradas;
- corridas encontradas;
- arquivos criados;
- arquivos atualizados;
- respostas vazias;
- erros;
- cobertura da temporada;
- status parcial ou completo.

Exemplo:

```text
data/openf1/manifests/sync_2025.json
```

---

# 5. Padronização dos pipelines

Os pipelines existentes serão migrados gradualmente.

Pipelines inicialmente considerados:

```text
pipeline_mallows_plackett_luce
pipeline_score_rules
pipeline_openf1
```

O novo pipeline científico comparativo ficará fora dessa migração inicial.

---

## 5.1 Estrutura padrão de pipeline

Cada pipeline deverá possuir uma estrutura semelhante a:

```text
src/
└── pipeline_nome/
    ├── README.md
    ├── configs/
    │   └── default.yaml
    ├── run_pipeline.py
    ├── visualization_plots.py
    └── outros módulos específicos
```

Os nomes exatos dos arquivos poderão variar, mas todos os pipelines deverão ter:

- documentação;
- configuração;
- ponto de entrada;
- geração de resultados estruturados;
- integração com artefatos;
- testes próprios.

---

## 5.2 Configuração em YAML

Parâmetros atualmente fixos no código deverão ser transferidos para arquivos YAML.

Exemplo:

```yaml
pipeline:
  name: mallows_plackett_luce

data:
  train_years:
    - 2019
    - 2020
    - 2021
    - 2022
    - 2023
  validation_years:
    - 2024
  test_years:
    - 2025

model:
  n_clusters: 2
  mallows_iterations: 150
  plackett_luce_iterations: 200
  alpha: 0.5

evaluation:
  top_k: 10
  simulations: 10000
  positions: 20

reproducibility:
  seed: 42
```

O YAML utilizado deverá ser copiado para a pasta da execução como `config_resolved.yaml`.

---

# 6. Centralização dos resultados em `artifacts/`

## 6.1 Responsabilidade da pasta

A pasta:

```text
artifacts/
```

será responsável por armazenar tudo o que for gerado por uma execução.

Ela não armazenará código-fonte nem dados brutos.

Separação conceitual:

```text
src/        → código
data/       → dados de entrada
configs/    → parâmetros
artifacts/  → resultados das execuções
notebooks/  → análise e apresentação
tests/      → validação automática
```

---

## 6.2 Estrutura dos artefatos

Exemplo:

```text
artifacts/
└── pipeline_score_rules/
    └── <run_id>/
        ├── config_resolved.yaml
        ├── manifest.json
        ├── metrics_summary.json
        ├── race_metrics.parquet
        ├── predictions.parquet
        ├── position_probabilities.parquet
        ├── parameter_history.parquet
        ├── runtime.json
        ├── warnings.json
        ├── run.log
        ├── model.pkl
        ├── notebook_bundle.pkl
        └── plots/
            ├── metrics_by_race.png
            ├── rps_model_vs_baseline.png
            └── demais gráficos
```

---

## 6.3 Formatos utilizados

A estratégia será:

```text
Parquet/CSV:
fonte principal dos resultados numéricos.

JSON:
metadados, métricas resumidas, manifests e avisos.

YAML:
configuração da execução.

PNG:
gráficos gerados.

PKL:
objeto auxiliar para recarregar modelos e estruturas Python.
```

O `.pkl` continuará permitido, mas não será mais a única fonte de dados para os notebooks.

---

## 6.4 Execuções imutáveis

Cada execução deverá gerar uma nova pasta.

Uma execução não deverá sobrescrever outra.

Exemplo:

```text
artifacts/pipeline_openf1/20260715_101500/
artifacts/pipeline_openf1/20260715_113200/
```

Também poderá existir:

```text
latest_run.json
```

Esse arquivo indicará qual foi a execução mais recente.

---

## 6.5 Registro de falhas

Mesmo uma execução com erro deverá tentar salvar:

```text
config_resolved.yaml
manifest.json
error.json
warnings.json
run.log
```

Isso permitirá investigar falhas sem depender exclusivamente do terminal.

---

# 7. Atualização dos notebooks

Os notebooks atuais dependem fortemente de arquivos `.pkl`.

Essa dependência será removida gradualmente.

Os notebooks deverão passar a ler:

```text
metrics_summary.json
race_metrics.parquet
predictions.parquet
position_probabilities.parquet
parameter_history.parquet
manifest.json
```

O `.pkl` poderá continuar sendo usado para análises específicas, mas não será obrigatório para carregar os resultados básicos.

---

## 7.1 Formas de localizar uma execução

Os notebooks poderão localizar os artefatos por:

1. caminho informado manualmente;
2. `run_id`;
3. `latest_run.json`;
4. configuração no início do notebook.

Exemplo:

```python
from pathlib import Path
import json
import pandas as pd

pipeline_dir = Path("artifacts/pipeline_score_rules")

with open(pipeline_dir / "latest_run.json", encoding="utf-8") as file:
    latest = json.load(file)

run_dir = Path(latest["run_path"])

metrics = pd.read_parquet(run_dir / "race_metrics.parquet")
predictions = pd.read_parquet(run_dir / "predictions.parquet")
```

---

## 7.2 Migração segura dos notebooks

Para cada pipeline:

1. executar a versão atual;
2. registrar os resultados existentes;
3. implementar a nova saída em `artifacts/`;
4. comparar resultados antigos e novos;
5. atualizar o notebook;
6. executar todas as células;
7. verificar gráficos e tabelas;
8. remover a dependência exclusiva do `.pkl`;
9. manter o fluxo antigo até a validação estar concluída.

---

# 8. README de cada pipeline

Cada pipeline deverá possuir seu próprio `README.md`.

O documento deverá conter:

```text
objetivo
modelo utilizado
fontes de dados
divisão temporal
configuração YAML
comando de execução
arquivos de saída
gráficos gerados
notebook relacionado
limitações conhecidas
estrutura interna
```

Exemplo:

```text
src/pipeline_openf1/README.md
```

O README principal do repositório funcionará como índice para os pipelines.

---

# 9. Estrutura geral de testes

A pasta:

```text
tests/
```

será criada para testar módulos compartilhados e pipelines existentes.

Nesta etapa, ela não conterá testes relacionados ao novo pipeline científico comparativo.

Estrutura proposta:

```text
tests/
├── README.md
├── conftest.py
│
├── fixtures/
│   ├── datasets/
│   ├── api_responses/
│   └── expected/
│
├── helpers/
│   ├── assertions.py
│   ├── factories.py
│   ├── fake_data.py
│   └── pipeline_runner.py
│
├── unit/
│   ├── data/
│   ├── data_openf1/
│   ├── models/
│   ├── engine/
│   ├── evaluation/
│   └── visualization/
│
├── data_quality/
│   ├── test_season_integrity.py
│   ├── test_ranking_integrity.py
│   ├── test_driver_mapping.py
│   ├── test_race_names.py
│   └── test_openf1_coverage.py
│
├── integration/
│   ├── test_openf1_sync_flow.py
│   ├── test_data_to_engine.py
│   ├── test_engine_to_evaluation.py
│   ├── test_artifact_generation.py
│   └── test_visualization_flow.py
│
├── pipelines/
│   ├── mallows_plackett_luce/
│   ├── score_rules/
│   └── openf1/
│
├── regression/
│   ├── baselines/
│   ├── test_mallows_pl_regression.py
│   ├── test_score_rules_regression.py
│   └── test_openf1_regression.py
│
├── smoke/
│   ├── test_imports.py
│   ├── test_cli_help.py
│   └── test_existing_pipeline_commands.py
│
└── e2e/
    ├── test_pipeline_mallows_plackett_luce.py
    ├── test_pipeline_score_rules.py
    └── test_pipeline_openf1.py
```

---

## 9.1 Responsabilidade de cada grupo

### `fixtures/`

Dados pequenos e controlados usados pelos testes.

### `helpers/`

Funções auxiliares reutilizadas por diferentes testes.

### `unit/`

Testes rápidos de funções e classes isoladas.

### `data_quality/`

Valida os dados reais armazenados no projeto.

### `integration/`

Verifica a comunicação entre diferentes módulos.

### `pipelines/`

Testes específicos dos pipelines existentes.

### `regression/`

Compara resultados novos com referências validadas.

### `smoke/`

Verifica rapidamente se imports e comandos básicos continuam funcionando.

### `e2e/`

Executa um fluxo completo, normalmente com dados sintéticos pequenos.

---

## 9.2 Regras obrigatórias dos testes

1. Não modificar arquivos reais de `data/`.
2. Não escrever no diretório real de `artifacts/`.
3. Usar diretórios temporários.
4. Não chamar a OpenF1 real nos testes padrão.
5. Utilizar mocks e fixtures para respostas da API.
6. Fixar a seed.
7. Usar datasets sintéticos pequenos nos testes rápidos.
8. Não depender da ordem de execução.
9. Comparar resultados probabilísticos com tolerância.
10. Criar um teste para cada bug corrigido.
11. Não incluir o `pipeline_cientifico_comparativo` durante a reorganização inicial.

---

# 10. Ordem de migração dos pipelines

A ordem planejada será:

```text
1. pipeline_mallows_plackett_luce
2. pipeline_score_rules
3. pipeline_openf1
```

Para cada pipeline:

```text
dados centralizados
    ↓
configuração YAML
    ↓
salvamento em artifacts/
    ↓
README próprio
    ↓
testes
    ↓
notebook atualizado
    ↓
comparação com resultados antigos
```

---

# 11. Roadmap da reestruturação

## Etapa 1 — Inventário e proteção

- congelar temporariamente o desenvolvimento do novo pipeline;
- preservar as mudanças já realizadas;
- criar uma branch específica para a reorganização;
- mapear arquivos, pipelines e notebooks;
- identificar dependências;
- registrar resultados atuais como referência;
- não excluir arquivos nesta etapa.

---

## Etapa 2 — Centralização da OpenF1

- revisar o cliente OpenF1 atual;
- criar o sincronizador central;
- implementar perfis `core` e `full`;
- armazenar dados de 2023, 2024, 2025 e 2026;
- criar manifests;
- validar cobertura;
- construir tabelas processadas;
- implementar carregador compartilhado;
- impedir chamadas diretas da API dentro dos pipelines.

---

## Etapa 3 — Validação dos dados

- conferir quantidade de corridas;
- verificar temporadas parciais;
- validar pilotos e equipes;
- detectar corridas duplicadas;
- validar posições;
- conferir DNF, DNS e DSQ;
- comparar dados históricos e OpenF1 durante a transição;
- documentar diferenças encontradas.

---

## Etapa 4 — Migração do Pipeline Mallows + Plackett–Luce

- adicionar YAML;
- mover resultados para `artifacts/`;
- manter PKL como auxiliar;
- adicionar README;
- criar testes;
- atualizar notebook;
- comparar métricas com a versão antiga.

---

## Etapa 5 — Migração do Pipeline Score Rules

- adicionar YAML;
- mover resultados para `artifacts/`;
- salvar probabilidades e RPS em formatos estruturados;
- adicionar README;
- criar testes;
- atualizar notebook;
- comparar métricas com a versão antiga.

---

## Etapa 6 — Migração do Pipeline OpenF1

- remover chamadas diretas à API;
- consumir exclusivamente `data/openf1/`;
- adicionar YAML;
- mover resultados para `artifacts/`;
- adicionar README;
- criar testes;
- atualizar notebook;
- validar cobertura e ausência de vazamento temporal.

---

## Etapa 7 — Atualização geral dos notebooks

- remover dependência exclusiva de PKL;
- ler Parquet, JSON e YAML;
- utilizar `latest_run.json` ou `run_id`;
- revisar gráficos;
- executar todos os notebooks;
- corrigir referências antigas a `outputs/`.

---

## Etapa 8 — Validação geral

Executar:

```text
testes unitários
testes de qualidade dos dados
testes de integração
testes dos pipelines
smoke tests
testes end-to-end
pipelines completos
notebooks completos
```

Também será feita uma comparação entre os resultados anteriores e os novos resultados.

---

## Etapa 9 — Retorno ao novo pipeline

Somente após a reorganização e validação geral será retomado o novo pipeline científico.

Nesse momento serão revisados:

- nomenclatura dos modelos;
- estrutura das fases;
- componentes da Fase 0;
- ArtifactStore;
- YAMLs;
- gráficos;
- protocolo temporal;
- testes experimentais;
- integração com os módulos reorganizados.

---

# 12. Itens que não serão removidos imediatamente

Durante a migração, os seguintes elementos poderão continuar existindo temporariamente:

```text
data/Season2023/
data/Season2024/
data/Season2025/
outputs/ dos pipelines antigos
arquivos PKL antigos
notebooks que ainda leem PKL
branch do pipeline científico comparativo
```

Eles somente serão removidos ou arquivados quando:

- os novos dados forem validados;
- os pipelines forem migrados;
- os notebooks forem atualizados;
- os resultados forem comparados;
- os testes estiverem passando;
- nenhuma parte do projeto depender mais deles.

---

# 13. Critérios para considerar a reestruturação concluída

A reorganização será considerada concluída quando:

1. Os dados OpenF1 de 2023 em diante estiverem centralizados.
2. Houver um script oficial de sincronização.
3. Nenhum pipeline acessar a API diretamente.
4. Os pipelines utilizarem o carregador compartilhado.
5. Todos os pipelines existentes possuírem YAML.
6. Todos os pipelines existentes salvarem resultados em `artifacts/`.
7. Nenhum notebook depender exclusivamente de PKL.
8. Cada pipeline possuir um README.
9. A pasta `tests/` estiver organizada e funcional.
10. Os pipelines antigos executarem corretamente.
11. Os notebooks executarem corretamente.
12. Os resultados novos forem comparados com os anteriores.
13. A documentação principal estiver atualizada.
14. O projeto estiver pronto para retomar o novo pipeline científico.

---

# 14. Resultado esperado

Ao final da reorganização, o projeto deverá possuir uma arquitetura semelhante a:

```text
Projeto-Formula1/
├── artifacts/
│   ├── pipeline_mallows_plackett_luce/
│   ├── pipeline_score_rules/
│   └── pipeline_openf1/
│
├── data/
│   ├── Season2019/
│   ├── Season2020/
│   ├── Season2021/
│   ├── Season2022/
│   ├── legacy/
│   └── openf1/
│       ├── raw/
│       ├── processed/
│       ├── manifests/
│       └── README.md
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── data_openf1/
│   ├── engine/
│   ├── evaluation/
│   ├── models/
│   ├── visualization/
│   ├── pipeline_mallows_plackett_luce/
│   ├── pipeline_score_rules/
│   ├── pipeline_openf1/
│   └── pipeline_cientifico_comparativo/
│
├── tests/
│   ├── fixtures/
│   ├── helpers/
│   ├── unit/
│   ├── data_quality/
│   ├── integration/
│   ├── pipelines/
│   ├── regression/
│   ├── smoke/
│   └── e2e/
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# 15. Resumo executivo

A reestruturação será realizada antes da continuidade do novo pipeline.

As principais mudanças serão:

```text
OpenF1 centralizada de 2023 em diante
dados locais como fonte para todos os pipelines
YAML em todos os pipelines
artifacts/ como destino único dos resultados
PKL mantido apenas como auxiliar
notebooks lendo arquivos estruturados
README em cada pipeline
tests/ para módulos e pipelines existentes
novo pipeline temporariamente congelado
```

A reorganização será gradual e validada etapa por etapa, evitando quebrar os resultados, notebooks e pipelines já existentes.
