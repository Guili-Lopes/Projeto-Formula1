# Relatório de Validação de Dados — Etapa 3

Comparação entre o dataset histórico e os dados OpenF1 processados,
conforme a Etapa 3 do plano de reestruturação.

## Corridas por temporada e fonte

| Ano | Legado | OpenF1 | Status OpenF1 | Pilotos/corrida | DNF | DNS | DSQ |
|---|---:|---:|---|---|---:|---:|---:|
| 2019 | 21 | — | — |  | — | — | — |
| 2020 | 17 | — | — |  | — | — | — |
| 2021 | 22 | — | — |  | — | — | — |
| 2022 | 22 | — | — |  | — | — | — |
| 2023 | 22 | 22 | parcial | 19–20 | 60 | 3 | 0 |
| 2024 | 24 | 24 | completo | 19–20 | 49 | 3 | 2 |
| 2025 | 15 | 24 | completo | 19–20 | 50 | 4 | 6 |

## Comparação legado × OpenF1 (2023–2025)

- Corridas comparadas: **61**
- Ordem dos classificados idêntica: **61/61**
- Conjunto de DNFs idêntico: **60/61**
- Ranking completo (incl. ordem da cauda de DNFs) idêntico: **51/61**

### Diferenças encontradas

- 2023 singapore: conjunto de DNFs diverge (legado=['BOT', 'OCO', 'STR', 'TSU'], openf1=['BOT', 'OCO', 'TSU'])
- 2025: apenas na OpenF1 → ['austin', 'baku', 'interlagos', 'las vegas', 'lusail', 'mexico city', 'monza', 'singapore', 'yas marina circuit']

## Ocorrências de qualidade

- 2023: corridas com menos de 20 pilotos na OpenF1: ['Singapore Grand Prix']
- 2024: corridas com menos de 20 pilotos na OpenF1: ['Australian Grand Prix']
- 2025: corridas com menos de 20 pilotos na OpenF1: ['Spanish Grand Prix']

## Conclusão

⚠️ Há diferenças documentadas acima; avaliar antes de remover o legado.
