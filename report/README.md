# AIT-ADS report (auto-generated)

## Что сделано
1) Каталог IDS правил:
- `report/rule_catalog_top.csv` — частоты rule_id + типичное описание/уровень + вклад Wazuh/AMiner

2) Таблица по стадиям атак из labels.csv:
- `report/stage_screen_all_scenarios.csv` — для каждого scenario и каждой стадии:
  - длительность
  - количество событий IDS в окне
  - сколько от Wazuh / AMiner
  - топ rule_id / description / rule_level

3) Pivot-таблица:
- `report/stage_pivot_events_total.csv` — stage × scenario → events_total

## Как связаны labels и логи
- `labels.csv` содержит ground truth: (scenario, attack stage, start/end time).
- Логи IDS (Wazuh/AMiner) не содержат названий стадий.
- События сопоставляются со стадией по времени: event.timestamp ∈ [start, end].

## Зачем это
Показывает, что:
- IDS уже классифицирует отдельные события (rule_id/description),
- многошаговая атака проявляется как последовательность стадий,
- у каждой стадии есть характерный паттерн событий.
Это база для эвристик и/или ML модели для распознавания стадий и multi-step атак.
