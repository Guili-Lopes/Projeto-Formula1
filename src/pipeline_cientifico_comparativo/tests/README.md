# Testes congelados do pipeline científico

Suíte `unittest` original do pipeline científico comparativo, preservada
junto ao pipeline (congelado até a Etapa 9 da reestruturação). A nova suíte
do repositório (`tests/`, em pytest) não inclui casos do pipeline
científico — eles serão reincorporados quando o pipeline for retomado.

Execução isolada:

    python -m unittest discover -s src/pipeline_cientifico_comparativo/tests
