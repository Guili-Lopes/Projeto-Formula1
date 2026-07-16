# Decisões Complementares — Reestruturação

Este documento registra decisões tomadas durante a revisão do plano de
reestruturação (`reestruturacao_projeto_formula1.md`). Ele complementa o plano
e deve ser lido junto com ele.

---

## 1. Fonte de dados 2023+ e comparabilidade

**Decisão:** a mudança de fonte para a OpenF1 a partir de 2023 não é tratada
como quebra de comparabilidade, pois os dados fundamentais (resultados,
corridas, posições) são os mesmos — a OpenF1 apenas adiciona mais informação.

**Salvaguarda mantida:** a Etapa 3 do plano (validação dos dados) fará a
comparação explícita entre os dados históricos e os dados da OpenF1 durante a
transição, documentando qualquer diferença encontrada. A pasta
`data/legacy/reference_2023_2025/` preservará os CSVs antigos como referência.

---

## 2. Pipeline científico comparativo

**Decisão:** tudo o que foi implementado para o `pipeline_cientifico_comparativo`
(Fase 0 / M0) será **considerado e refeito no futuro**, na Etapa 9 do roadmap.

Consequências práticas:

- o código atual permanece no repositório, congelado, sem alterações durante a
  reorganização;
- as nomenclaturas `M0`, `M1`, `M2` etc. são provisórias — a nomenclatura
  definitiva usará nomes descritivos baseados no comportamento de cada modelo;
- as métricas de referência do M0 (validação 2024) permanecem registradas como
  referência histórica, mas o M0 será re-congelado sobre a nova camada de dados
  quando o pipeline for retomado.

---

## 3. Configuração YAML dos pipelines históricos

**Decisão:** cada pipeline histórico terá o **seu próprio YAML, preservando o
split temporal histórico dele**, para que a migração reproduza os resultados
antigos.

Exemplo: o `pipeline_mallows_plackett_luce` manterá em seu `default.yaml` o
split treino 2019–2022 / validação 2023 / teste 2024. A eventual padronização
para o split oficial do pipeline científico (2019–2023 / 2024 / 2025) é uma
decisão separada e posterior, não parte da migração.

---

## 4. Pasta `tests/`

**Decisão:** a pasta `tests/` será **refeita do zero** seguindo a estrutura da
seção 9 do plano (fixtures, helpers, unit, data_quality, integration,
pipelines, regression, smoke, e2e).

- Os testes atuais (todos relacionados à infraestrutura do pipeline científico)
  não são excluídos na Etapa 1 — nada é excluído nesta etapa.
- Os casos de teste referentes ao pipeline científico serão adicionados de
  volta quando o desenvolvimento dele for retomado (Etapa 9).

---

## 5. Dependências e schema de dados processados

**Decisão (a):** `requests` é adicionado imediatamente ao `requirements.txt`
(é importado por `src/data_openf1/client.py`, mas não estava declarado, o que
quebraria uma instalação limpa).

**Decisão (b):** as dependências declaradas mas não utilizadas no código
(ver inventário da Etapa 1) **não são removidas agora** — ficam documentadas
para avaliação futura, conforme a Prioridade 3 da documentação principal.

**Decisão (c):** ao construir as tabelas `data/openf1/processed/` na Etapa 2,
os campos de status disponíveis na API (causa do abandono, voltas completadas,
status de classificação etc.) devem ser **capturados desde já**, mesmo que o
modelo ainda não os utilize. Isso evita uma re-sincronização futura e prepara
o terreno para a revisão do schema de corrida (novo `RaceRecord`), que ocorrerá
quando o pipeline científico for retomado.

---

*Registrado em 14 de julho de 2026, durante a preparação da Etapa 1.*
