# Projeto Fórmula 1 — Previsão Probabilística de Rankings

Projeto que investiga a previsão probabilística de resultados de corridas de Fórmula 1 por meio de modelos estatísticos de ranking.

---

## Problema

O resultado de uma corrida de Fórmula 1 é naturalmente um **ranking** de competidores. Tratá-lo como classificação binária (vence ou não vence) desperdiça a maior parte da informação disponível. Este projeto trata a previsão como um problema ordinal e probabilístico: em vez de prever apenas o vencedor, o modelo estima uma **distribuição de probabilidades sobre posições** para cada piloto.

---

## Abordagem

O projeto combina dois modelos probabilísticos de ranking:

**Modelo de Mallows** — agrupa corridas históricas com padrões de resultado similares, identificando contextos competitivos distintos (ex: eras de dominância de diferentes equipes).

**Modelo de Plackett–Luce** — estima parâmetros de habilidade relativa para cada piloto, globalmente e dentro de cada contexto identificado pelo Mallows.

O sistema opera de forma **incremental**: a previsão de cada corrida é feita antes de seu resultado ser incorporado ao modelo, respeitando a ordem temporal e evitando vazamento de informação.

A qualidade do modelo é avaliada tanto por métricas de concordância ordinal (Top-N accuracy, Kendall tau) quanto por métricas probabilísticas (Ranked Probability Score).

---

## Dados

Duas fontes complementares:

**CSVs históricos** — resultados de corridas, qualificação, sprint e informações de pilotos para as temporadas de 2019 a 2025, organizados em `data/Season<ano>/`.

**API OpenF1** — dados contextuais de sessão (eventos de corrida, grade de largada, status de abandono) a partir de 2023, armazenados em cache local em `data/openf1/`.

---

## Organização do Repositório

```
Projeto-Formula1/
├── artigos/          # Base bibliográfica do projeto
├── data/             # Dados históricos (CSVs) e cache da OpenF1
├── notebooks/        # Narrativa progressiva do projeto (00 → 07)
├── src/              # Módulos Python
│   ├── data/             # Carregamento e transformação dos CSVs
│   ├── data_openf1/      # Acesso, cache e features da API OpenF1
│   ├── engine/           # Treinamento e predição (compartilhado)
│   ├── evaluation/       # Métricas de avaliação (compartilhado)
│   ├── models/           # Mallows, Plackett-Luce, pesos (compartilhado)
│   └── pipeline_*/
├── requirements.txt
└── README.md
```

Os módulos em `src/data/`, `src/data_openf1/`, `src/engine/`, `src/evaluation/` e `src/models/` são **compartilhados entre todos os pipelines**. As pastas `pipeline_*/` apenas orquestram esses módulos para cada experimento específico.

---

## Como Executar

```bash
# 1. Clone e configure o ambiente
git clone https://github.com/Guili-Lopes/Projeto-Formula1.git
cd Projeto-Formula1
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 2. Explore os notebooks em ordem
jupyter notebook notebooks/

# 3. Execute os pipelines individualmente
python -m src.pipeline_mallows_plackett_luce.run_experiment
python -m src.pipeline_score_rules.run_pipeline_score_rules
python -m src.pipeline_openf1.run_pipeline_openf1
```

Os notebooks de análise (05, 06, 07) dependem dos arquivos `.pkl` gerados pelos scripts acima.

---

## Referências

As principais referências teóricas estão disponíveis em `artigos/` e cobrem:

- Plackett–Luce e modelos de ranking probabilístico
- Modelo de Mallows para rankings parciais
- Inferência bayesiana e algoritmos MM
- Score rules e Ranked Probability Score
- Previsão probabilística e calibração

---

## Autor

**Guilherme Lopes** — [github.com/Guili-Lopes](https://github.com/Guili-Lopes)
