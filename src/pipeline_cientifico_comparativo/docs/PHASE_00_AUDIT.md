# Fase 0 — Auditoria e congelamento do baseline M0

## 1. Escopo

A Fase 0 não altera a formulação estatística do projeto. Ela cria uma execução
configurável, reproduzível e auditável do baseline Mallows + Plackett–Luce, com
Monte Carlo e RPS, usando treino em 2019–2023 e validação em 2024.

O teste de 2025 permanece bloqueado durante o desenvolvimento.

## 2. Fluxo localizado

```text
src/data/data_pipeline.py
        ↓
src/models/models_weights.py
        ↓
src/models/models_mallows.py
        ↓
src/models/models_plackett_luce.py
        ↓
src/engine/engine_trainer.py + engine_predictor.py
        ↓
src/pipeline_score_rules/monte_carlo.py
        ↓
src/evaluation/evaluation_metrics.py + scoring_rules.py
```

### Carregamento

`src/data/data_pipeline.py` lê os arquivos `raceResults.csv`, padroniza pilotos,
constrói rankings parciais e retorna objetos `RaceRecord` em ordem cronológica.

### Pesos

`src/models/models_weights.py` calcula os pesos temporais e regulatórios usados
no treinamento.

### Mallows

`src/models/models_mallows.py` agrupa as corridas com base em distância de
Kendall e atualiza consensos por Borda ponderado dentro do esquema iterativo de
atribuição.

### Plackett–Luce

`src/models/models_plackett_luce.py` estima scores globais e por cluster por
meio de um algoritmo MM ponderado.

### Atualização incremental

`src/engine/engine_trainer.py` treina o estado inicial e incorpora uma corrida
somente depois da previsão e avaliação correspondentes.

### Previsão

`src/engine/engine_predictor.py` escolhe o cluster com base no histórico do
circuito e combina scores globais e do cluster para formar o ranking pontual.

### Probabilidades

`src/pipeline_score_rules/monte_carlo.py` simula rankings Plackett–Luce e gera
a distribuição piloto × posição. O RPS é calculado em
`src/pipeline_score_rules/scoring_rules.py`.

## 3. Parâmetros históricos congelados

| Parâmetro | Valor |
|---|---:|
| Seed | 42 |
| Clusters Mallows | 2 |
| Iterações Mallows no treino | 150 |
| Iterações Mallows no refit | 150 |
| Alpha Mallows | 0.5 |
| Iterações Plackett–Luce | 200 |
| Top-k do ranking | 10 |
| Peso do cluster na previsão pontual | 0.7 |
| Tamanho mínimo do cluster | 5 |
| Simulações por corrida | 10.000 |
| Posições probabilísticas | 20 |
| Fonte de score no Monte Carlo | cluster |

Esses valores agora são definidos por YAML, mas preservam os defaults usados
pelos pipelines anteriores.

## 4. Reprodutibilidade implementada

A Fase 0 introduz:

- configuração resolvida em YAML;
- validação de schema e separação temporal;
- seed global de Python e NumPy;
- seed Monte Carlo estável e específica para cada corrida;
- manifesto com commit, status do Git, ambiente e versões de dependências;
- inventário e SHA-256 de cada arquivo de corrida usado;
- diretórios de execução imutáveis;
- logs, warnings e tempos de execução;
- tabelas oficiais em Parquet, com espelhos CSV selecionados;
- PKL apenas como conveniência e preservação de objetos.

## 5. Fragilidades identificadas no baseline

1. Os pipelines anteriores guardam diversos parâmetros como constantes em
   scripts de entrada. A nova camada YAML evita editar Python para cada cenário.

2. O Monte Carlo anterior podia ser executado sem seed por corrida.
   `np.random.seed(42)` não controla uma chamada independente a
   `np.random.default_rng(None)`. A Fase 0 deriva e registra uma seed estável
   para cada corrida.

3. As iterações do Plackett–Luce estavam fixadas em 200 dentro de
   `engine_trainer.py`, e o refit incremental do Mallows usava 150 de forma
   interna. Elas agora são parâmetros explícitos com os mesmos defaults.

4. Os pipelines anteriores gravam sempre na mesma pasta e podem sobrescrever
   resultados. O novo armazenamento cria um `run_id` único e não sobrescreve
   diretórios existentes.

5. O PKL era a interface principal dos notebooks. A nova convenção considera
   Parquet, JSON e YAML como fonte oficial e mantém PKL como conveniência.

6. O Pipeline 3 histórico construía `all_drivers` depois de carregar também
   2025. Isso incluía pilotos do teste final na validação de 2024 e alterava a
   normalização e o baseline uniforme. O M0 científico de validação carrega
   somente 2019–2024.

7. O teste de 2025 precisa permanecer intocado durante todas as decisões. O CLI
   não carrega 2025 em `validation` e bloqueia `final_test` sem autorização
   explícita.

8. Monte Carlo e scoring rules ainda residem em uma pasta de pipeline, embora
   sejam componentes reutilizáveis. A migração para módulos compartilhados
   deverá ocorrer em etapa posterior, sem duplicar lógica durante o
   congelamento do M0.

9. O loader atual converte qualquer valor não numérico em `Position` para DNF.
   A regra é preservada no M0 para equivalência e deverá ser revisada antes dos
   modelos D1, D2 e D3.

10. A previsão pontual combina scores globais e do cluster, enquanto o Monte
    Carlo histórico usa apenas scores do cluster. Essa decisão foi explicitada
    em `simulation.score_source: cluster`.

11. O universo de pilotos é estático dentro de uma execução e inclui todos os
    pilotos observados no período carregado, e não somente os inscritos em cada
    corrida. Isso afeta a normalização e torna o baseline uniforme
    relativamente fraco. O comportamento é preservado no M0 e deverá ser
    tratado nas fases com grid e features.

12. O consenso dos clusters é atualizado por Borda ponderado dentro do esquema
    iterativo de atribuição. A implementação atual não representa a posteriori
    completa de um modelo Mallows Bayesiano.

13. Mallows e Plackett–Luce usam número fixo de iterações, sem critério explícito
    de convergência. A Fase 0 registra esses valores para comparação futura.

14. Para circuitos sem histórico, o preditor usa o cluster mais frequente. Para
    circuitos conhecidos, usa o cluster modal do circuito no histórico
    observado.

15. O resultado observado é um ranking parcial formado por classificados
    Top-k e DNFs. Métricas de posição devem ser interpretadas dentro dessa
    representação, e não como avaliação integral das 20 posições oficiais.

## 6. Critério de reprodução

As métricas determinísticas devem permanecer próximas da referência registrada
para a validação de 2024. O RPS aceita tolerância maior porque a execução
histórica:

- não fixava seed Monte Carlo por corrida;
- usava um universo de pilotos que incluía 2025.

Depois da Fase 0, duas execuções M0 com a mesma configuração e os mesmos arquivos
de entrada devem produzir resultados idênticos.

O arquivo `baseline_equivalence.json` registra a comparação automática.

## 7. Saídas obrigatórias

Cada execução salva:

- configuração resolvida;
- manifesto e ambiente;
- inventário e hashes dos arquivos de entrada;
- métricas por corrida e agregadas;
- rankings previstos e observados;
- probabilidades por piloto e posição;
- histórico dos scores pré-previsão;
- registros processados;
- atribuições, tamanhos e consensos dos clusters;
- estado completo do modelo;
- logs, warnings e tempo de execução;
- comparação automática com a referência M0;
- pacote de conveniência para o notebook futuro.

## 8. Critério para encerrar a Fase 0

A Fase 0 será considerada validada quando:

1. o `dry-run` localizar todos os arquivos de 2019–2024;
2. a execução de validação terminar com 24 corridas;
3. `test_years_loaded` for `false`;
4. a comparação com a referência não indicar divergência inesperada;
5. duas execuções com a mesma seed forem equivalentes;
6. todos os artefatos obrigatórios forem gerados;
7. os pipelines existentes continuarem executáveis com seus valores padrão.
