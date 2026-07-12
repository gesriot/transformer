# Transformer / KAN

Локальное приложение и CLI для численной регрессии, сравнения моделей и
интерпретации зависимостей. Поддерживаются три численные архитектуры:

- `transformer` — encoder-decoder Transformer для табличных признаков;
- `mlp` — полносвязный baseline;
- `kan` — Kolmogorov-Arnold Network: обучаемая одномерная функция на каждом
  ребре вместо фиксированной активации в узле.

KAN реализована на существующем autograd-движке проекта. Каждое ребро сочетает
кубический B-сплайн на сетке `[-3, 3]` и базовую ветку `w·gelu(x)`, поэтому
модель остаётся обучаемой и вне сплайн-сетки.

## Возможности

- обучение на встроенных black-box задачах или на `.tnum`, подготовленном из
  CSV/Excel;
- сравнение Transformer, MLP и KAN в `sweep` и GUI Optimize;
- activation-L1, hard-prune рёбер и fine-tune KAN;
- structural compaction: физическое удаление неиспользуемых и постоянных
  скрытых узлов без изменения функции модели;
- графики функций KAN-рёбер и символьная экстракция формул в исходных единицах
  входов/выходов;
- сохранение и загрузка моделей, batch-predict в Excel, диагностика и
  epoch-sweep;
- macOS GUI и CLI без внешнего ML-фреймворка.

## Быстрый старт

Нужен стабильный Rust toolchain. GUI включён по умолчанию.

```bash
cargo build --release
./target/release/transformer gui
```

Без аргументов бинарник также открывает GUI. Для лёгкой CLI-сборки без GUI:

```bash
cargo build --release --no-default-features
```

### Подготовка Excel

```bash
./target/release/transformer prepare data.xlsx data.tnum \
  --inputs 3 --outputs 3 --has-header
```

`prepare` принимает CSV, TSV и Excel. Категориальные колонки задаются флагом
`--categorical колонка:число_классов`.

### Обучение KAN и получение формул

```bash
./target/release/transformer numeric-file data.tnum \
  --model-kind kan --kan-width 16 --kan-layers 2 --kan-grid 16 \
  --lr 3e-3 --epochs 80 --seed 0 \
  --kan-l1 1e-3 --kan-prune 0.05 --kan-finetune-epochs 20 \
  --kan-compact --kan-symbolic --model model-kan.bin
```

Порядок фаз: обучение → L1 → prune → fine-tune с `λ=0` → compact → symbolic
extraction → сохранение. `--kan-compact` имеет смысл после prune: он уменьшает
реальное число параметров, а не только число активных масок.

`--kan-symbolic` печатает послойные формулы. Входы `x0`, `x1`, … и выходы
`y0`, `y1`, … уже находятся в исходных единицах таблицы; внутренние `hK_J`
остаются безразмерными промежуточными узлами. Рядом выводятся качество фита
каждого ребра и честная метрика формульной модели на test-наборе.

### Сравнение архитектур

```bash
./target/release/transformer sweep projectile \
  --model-kinds transformer,mlp,kan \
  --d-models 32 --layers-list 2 \
  --mlp-widths 64,128 --mlp-layers-list 3 \
  --kan-widths 8,16 --kan-layers-list 2 --kan-grids 8,16 \
  --lrs 1e-3,3e-3 --epochs 80 --batch-size 64 --seeds 0,1
```

Для табличного файла используйте GUI **Optimize** или `epoch-sweep`, чтобы
подобрать число эпох для выбранного конфига.

## GUI

После обучения или загрузки численной модели доступны вкладки:

- **Train**, **Predict**, batch-predict в Excel, **Diagnose**, **Optimize** и
  **Epoch-sweep**;
- **KAN curves** — график `φ(input → output)` выбранного ребра. Для первого
  слоя ось X нормализована, для последующих — это активация предыдущего слоя;
- **KAN formulas** — формулы в исходных единицах, кнопка копирования, R²
  формульной модели и список рёбер с фитами ниже `R² = 0.99`.

Фит формул вычисляется в worker-потоке; UI получает только готовый текст,
метрики и простые числовые данные.

## Checkpoint-ы

KAN-checkpoint сохраняет обучаемые параметры, hard-prune маски, реальные
размеры слоёв после compaction и равномерную выборку до 256 сырых train-строк
для символического фита. Поэтому новая сохранённая KAN может показать формулы
после загрузки, хотя test-метрики тогда недоступны: test-набор не хранится в
checkpoint-е.

Старые checkpoint-ы без `kan_dims`, масок или калибровочной секции продолжают
загружаться. Для старой KAN без калибровки формулы можно получить, повторно
обучив модель на исходном `.tnum` и сохранив её новой версией.

## Проверка и упаковка

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
./scripts/package-macos.sh
```

macOS-упаковщик создаёт `dist/Transformer.app`, ZIP и
`dist/Transformer.dmg`, затем проверяет DMG через `hdiutil`. Для генерации
иконки нужен модуль Pillow для `python3`.
