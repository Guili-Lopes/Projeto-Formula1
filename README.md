# 🏎️ Projeto Fórmula 1 — Previsão Probabilística de Resultados

> Estimativa Bayesiana dinâmica das probabilidades de título no Campeonato de Pilotos e de Construtores da Fórmula 1.

---

## 📌 Sobre o Projeto

Este projeto desenvolve um sistema que, a cada corrida da temporada de F1, estima:

- 🏆 A **probabilidade de cada piloto** vencer o Campeonato de Pilotos
- 🏗️ A **probabilidade de cada equipe** vencer o Campeonato de Construtores

As estimativas são **atualizadas corrida a corrida**, combinando modelos estatísticos Bayesianos com abordagens de aprendizado de máquina.

---

## 🔬 Metodologia

### Modelo Principal — Plackett–Luce Bayesiano Hierárquico

O modelo **Plackett–Luce** associa a cada piloto/equipe um parâmetro de *habilidade* $w_i > 0$. A probabilidade de observar um ranking é proporcional às habilidades relativas dos competidores:

$$P(\sigma \mid \mathbf{w}) = \prod_{k=1}^{n} \frac{w_{\sigma_k}}{\sum_{j=k}^{n} w_{\sigma_j}}$$

Na versão **Bayesiana Hierárquica**, os parâmetros de habilidade são variáveis aleatórias com distribuições a priori (e.g., Gama, Log-Normal), permitindo quantificar incerteza e realizar atualização sequencial.

### Inferência

| Método | Descrição |
|--------|-----------|
| **MCMC** | Amostragem exata (assintótica) via Metropolis–Hastings |
| **Power EP** | Expectation Propagation com α-divergências — alternativa escalável ao MCMC |

### Clustering de Corridas — Modelo de Mallows

Agrupa corridas por perfil de pista (rápidas vs. técnicas) usando o Modelo de Mallows com distância de Kendall e o Algoritmo 4 (MCMC para Clustering de Rankings Parciais).

### Comparação com Rede Neural

Um modelo de deep learning supervisionado é treinado para prever probabilidades de vitória/pódio/posição, permitindo comparação direta com a abordagem estatística.

---

## 📁 Estrutura do Repositório

```
Projeto-Formula1/
├── artigos/        # Artigos científicos de referência (PDF)
│
├── data/           # Datasets por temporada
│
├── notebooks/      # Notebooks Jupyter
│
├── src/            # Scripts Python
│        
├── README.md
└── requirements.txt
```

---

## 🗂️ Dados

Os dados cobrem as temporadas de **2019 a 2025** e incluem:

| Arquivo | Conteúdo |
|---------|----------|
| `raceResults.csv` | Resultados de cada corrida (posição, pontos, DNF) |
| `qualifyingResults.csv` | Resultados da qualificação |
| `SprintResults.csv` | Resultados das corridas Sprint (a partir de 2021) |
| `SprintQualifyingResults.csv` | Resultados do Sprint Qualifying |
| `drivers.csv` | Informações dos pilotos |
| `calendar.csv` | Calendário da temporada |

**Fontes:**
- [toUpperCase78/formula1-datasets](https://github.com/toUpperCase78/formula1-datasets) — dados 2019–2025
- [Kaggle — F1 World Championship 1950–2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — histórico completo

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/Guili-Lopes/Projeto-Formula1.git
cd Projeto-Formula1
```

### 2. Crie e ative um ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# ou
venv\Scripts\activate           # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute os notebooks

```bash
jupyter notebook notebooks/
```

Siga a ordem numérica dos notebooks para reproduzir o projeto do início.

### 5. Execute os scripts (Exemplo)

```bash
python src/Teste MCMC/MCMC_ClusteringPartialRankings.py
```

---

## 📚 Referências

1. Caron, F. & Doucet, A. — *Bayesian Inference for Plackett-Luce Ranking Models*
2. Minka, T. — *Power EP*
3. Gelman, A. et al. — *Bayesian Modelling and Analysis of Data*
4. Meila, M. & Chen, H. — *Modelo Mallows Concêntrico Top-k*
5. Vitelli, V. et al. — *Probabilistic Preference Learning with the Mallows Rank Model*
6. Chierichetti, F. et al. — *Mallows Models for Top-k Lists*
7. Guiver, J. & Snelson, E. — *Bayesian Inference for Plackett-Luce Ranking Models* (variante)
8. Meilă, M. — *Algorithms for Mallows Models (BT/PL)*
9. Chang, S. et al. — *Neural RankGAM: Learning to Rank with Neural Additive Models*

---

## 🧑‍💻 Autor

**Guilherme Lopes**
[GitHub](https://github.com/Guili-Lopes)
