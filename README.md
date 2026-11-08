# Assignment 3: EDA and Metric Analysis

Эта папка содержит EDA, анализ метрики, стратегию валидации, adversarial validation и короткий NER-пилот на BiGRU. Полное обучение и ансамбли пока не выполнены; обученный пилот использует общий recurrent-энкодер и NER-голову из `recurrent_model.py`.

## Готовые результаты

- [Отчёт на английском](reports/EDA_AND_METRIC_ANALYSIS.md): выводы, таблицы, ограничения и рекомендации.
- [Ноутбук с результатами](eda_and_metrics.ipynb): таблицы, график и воспроизводимые примеры метрики.
- [Сводные показатели](reports/eda_summary.json): полный аудит, пересечения выборок и SHA-256 исходных файлов.
- [Краткая таблица данных](reports/dataset_overview.csv).
- [Граничные случаи метрики](reports/metric_edge_cases.csv).
- [График EDA](figures/eda_overview.png).
- [Стратегия валидации](reports/VALIDATION_STRATEGY.md): фиксированные части, защита от утечки, причины расхождений с Public LB.
- [Ноутбук валидации](validation.ipynb): реальные результаты трёхfold adversarial validation.
- [Результаты adversarial validation](reports/adversarial_results.json) и [график](figures/adversarial_validation.png).
- [Проверки BIO/padding/длинных текстов и NER-пилот](reports/NER_PREFLIGHT_AND_PILOT.md), [ноутбук](ner_preflight.ipynb), [кривые обучения](figures/ner_pilot_learning.png).
- [Анализ ошибок checkpoint](reports/CHECKPOINT_ERROR_ANALYSIS.md): знакомые/новые названия, OOV, границы, типы и приоритеты для команды.
- [Контекстный разбор 200 случаев](reports/CHECKPOINT_REVIEW_200.md), [таблица с комментариями](reports/checkpoint_review_200_annotated.csv) и [ноутбук](checkpoint_error_analysis.ipynb).

Анализ охватывает весь исходный украинский CSV и подготовленный Parquet (по 1 010 000 строк), все 477 тестовых текстов и 26 размеченных примеров. Русский вспомогательный корпус не анализировался. Тестовые ответы неизвестны: пустые `locations` являются заглушками.

## Повторить анализ

Команды ниже выполняются из корня существующего проекта. Используется его `.venv`.

```powershell
uv --cache-dir .uv-cache pip install --python .venv/Scripts/python.exe -r assignment3/requirements.txt
.venv/Scripts/python.exe assignment3/analyze.py
.venv/Scripts/python.exe assignment3/build_deliverables.py
.venv/Scripts/python.exe -m unittest discover -s assignment3 -p test_metrics.py
```

## Повторить валидацию

Нужен PyTorch для вашей CPU/GPU-конфигурации; использован уже установленный `torch 2.11.0+cu130`.

```powershell
.venv/Scripts/python.exe assignment3/validation.py
.venv/Scripts/python.exe assignment3/build_validation_report.py
.venv/Scripts/python.exe -m unittest discover -s assignment3 -p "test_*.py"
```

Разбиение: 999 948 fit, 4 000 checkpoint, 2 000 calibration, 4 000 audit; 52 пересекающихся fit-строки исключены. Manifest `data/validation_splits.csv` сохранён, его хеш и проверки — в `reports/validation_splits_summary.json`. Эти части одинаковы для всех будущих экспериментов. Audit оценивается после фиксации модели и декодирования, а не при подборе параметров.

Adversarial validation использует 477 исходных и 477 тестовых текстов, трёхfold групповые OOF-предсказания и общий для domain/NER энкодер BiGRU. После унификации токенизации с NER-пилотом OOF ROC-AUC = 0.9493: входные распределения заметно различаются. Это не NER-F1. Если финальная основная модель использует другой recurrent-энкодер или токенизацию, AV нужно повторить с ними. Реальные NER-результаты записаны в `reports/experiment_ledger.csv`; CV, final audit и LB остаются неизмеренными.

## Проверки перед большим обучением

```powershell
.venv/Scripts/python.exe assignment3/audit_ner_preprocessing.py
.venv/Scripts/python.exe assignment3/pilot_ner.py
.venv/Scripts/python.exe assignment3/check_ner_inference.py
.venv/Scripts/python.exe assignment3/build_ner_preflight_report.py
```

В fit обнаружены 1 485 строк с границами внутри слов. Они исключены из словного BIO-обучения, а исходные данные сохранены. Полная checkpoint-часть, включая несовместимые строки, остаётся в оценке сущностей. Пилот: 12 000 fit-текстов, восемь эпох, F1 0.8131 на train-monitor и 0.5682 на checkpoint. Есть заметный разрыв обобщения; это не гарантия качества большого обучения. Все тестовые токены обработаны окнами, final audit и calibration для NER не оценивались.

## Анализ ошибок для передачи команде

```powershell
.venv/Scripts/python.exe assignment3/analyze_checkpoint_errors.py
.venv/Scripts/python.exe assignment3/review_checkpoint_cases.py
.venv/Scripts/python.exe assignment3/build_checkpoint_error_report.py
```

Весь checkpoint: 917 FP и 431 FN, включая 117 FP и 108 FN с пересечением границ. Для поверхностей, встречавшихся как LOC в фактическом 12k fit, F1=0.8312; для новых F1=0.2011. Это группировка нормализованных кандидатов, а не оценка только текстов с новыми именами. Recall для известных поверхностей 85.8%, для новых 30.0%.

200 отдельных текстов разобраны по контексту (100 FP, 100 FN); выборка диагностическая, а не оценка распространённости шума. Замечания Codex являются AI-assisted review, не независимой экспертной разметкой. Есть кандидаты на ошибки типов, неполную разметку и противоречия в гранулярности. Gold, веса модели и разбиение не изменены. Не переносите checkpoint-примеры в обучение и не исправляйте ответы под модель; сначала согласуйте правила LOC и версионирование любых правок.

Ожидаются файлы `data/uk_geo_dataset.csv`, `data/uk_geo_dataset_processed_v1.parquet`, `data/test.csv`, `data/labeling_sample.csv` внутри этой папки. Они скачаны с Kaggle и исключены существующим правилом `.gitignore` для `data`. В `sources/` сохранены официальные страницы соревнования, включая точный опубликованный код оценки.

Для новой загрузки при настроенном Kaggle API и принятии правил:

```powershell
.venv/Scripts/kaggle.exe competitions download -c kse-ua-location-extraction-2026 -f uk_geo_dataset.csv -p assignment3/data
.venv/Scripts/kaggle.exe competitions download -c kse-ua-location-extraction-2026 -f uk_geo_dataset_processed_v1.parquet -p assignment3/data
.venv/Scripts/kaggle.exe competitions download -c kse-ua-location-extraction-2026 -f test.csv -p assignment3/data
.venv/Scripts/kaggle.exe competitions download -c kse-ua-location-extraction-2026 -f labeling_sample.csv -p assignment3/data
```

`download_data.py --file uk_geo_dataset.csv` — дополнительный загрузчик с возобновлением по HTTP-диапазонам через авторизацию Kaggle. Он не сохраняет ключи и подписанные URL. При обычном соединении достаточно стандартных команд выше.

## Существенные ограничения

`metrics.py` намеренно воспроизводит опубликованное регулярное выражение, включая дефект интерполяции. Это проверено сравнением с опубликованной функцией, а не с закрытым серверным кодом Kaggle. Исправленный вариант нормализации предназначен только для диагностики. Предложенная альтернативная метрика не подменяет целевую метрику соревнования.

В EDA нет обучения моделей и вымышленных CV/LB-оценок. В отчёте отмечено использование Codex; для раскрытия AI-помощи приложите реальную ссылку на чат или его экспорт, как требует задание.
