# Pipeline Científico Comparativo

Este pacote coordena o estudo incremental e comparável dos modelos do Projeto
Fórmula 1. O código científico reutilizável continua em `src/models`,
`src/engine`, `src/evaluation`, `src/data` e módulos compartilhados. Esta pasta
contém configurações, registro de modelos, fases e orquestração.

## Fase 0 — M0

A Fase 0 congela o baseline atual:

- Mallows;
- Plackett–Luce ponderado;
- atualização incremental;
- Monte Carlo;
- Top-3, Top-5, acerto do vencedor, Kendall tau e erro de posição;
- RPS contra baseline uniforme.

A divisão temporal oficial é:

```text
Treino: 2019–2023
Validação: 2024
Teste final: 2025 (bloqueado durante o desenvolvimento)
```

## Dependências

```bash
pip install -r requirements.txt
```

A infraestrutura usa `PyYAML` para configurações e `pyarrow` para os artefatos
Parquet.

## Validar configuração e dados sem treinar

```bash
python -m src.pipeline_cientifico_comparativo.run_phase \
  --phase 0 \
  --mode validation \
  --dry-run
```

O `dry-run` resolve os YAMLs, verifica os schemas, localiza os CSVs necessários
e confirma que 2025 não é carregado na validação.

## Executar M0 na validação

```bash
python -m src.pipeline_cientifico_comparativo.run_phase \
  --phase 0 \
  --mode validation
```

É possível sobrescrever a semente e a raiz de artefatos sem editar Python:

```bash
python -m src.pipeline_cientifico_comparativo.run_phase \
  --phase 0 \
  --mode validation \
  --seed 123 \
  --artifact-root artifacts/experimento_seed123
```

## Executar todos os modelos registrados

Na Fase 0, somente M0 está registrado. O mesmo comando passará a executar M1,
M2 e as demais fases quando forem adicionadas.

```bash
python -m src.pipeline_cientifico_comparativo.run_all --mode validation
```

Ao final, o orquestrador cria uma tabela consolidada em
`artifacts/pipeline_cientifico_comparativo/comparison/validation/`.

## Teste final

O teste de 2025 não é carregado em `validation`. Quando todas as decisões de
modelagem e hiperparâmetros estiverem congeladas, a execução final exige uma
confirmação explícita:

```bash
python -m src.pipeline_cientifico_comparativo.run_all \
  --mode final_test \
  --unlock-final-test
```

No modo `final_test`, 2024 é incorporado sequencialmente como período de
warm-up antes da avaliação de 2025.

## Artefatos

Cada execução cria um diretório imutável:

```text
artifacts/pipeline_cientifico_comparativo/
└── validation/
    └── M0/
        └── seed_42/
            ├── latest_run.json
            └── M0_validation_seed42_<timestamp>/
                ├── config_resolved.yaml
                ├── environment.json
                ├── phase_00_audit.md
                ├── manifest.json
                ├── metrics_summary.json
                ├── baseline_equivalence.json
                ├── race_metrics.parquet
                ├── race_metrics.csv
                ├── predictions.parquet
                ├── predictions.csv
                ├── position_probabilities.parquet
                ├── parameter_history.parquet
                ├── records.parquet
                ├── records.csv
                ├── cluster_assignments.parquet
                ├── cluster_assignments.csv
                ├── cluster_consensus.json
                ├── runtime.json
                ├── warnings.json
                ├── run.log
                ├── model.pkl
                └── notebook_bundle.pkl
```

Parquet, JSON e YAML são a fonte oficial. O PKL é uma camada de conveniência
para notebooks e preservação de objetos Python.

## Testes

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Os testes cobrem configuração, trava temporal, reprodutibilidade sintética,
compatibilidade dos valores padrão, métricas adicionais e escrita não
destrutiva dos artefatos.
