# Transformer / KAN

Локальное Rust-приложение и CLI для численной регрессии, сравнения моделей и
интерпретации зависимостей. Оно обучает surrogate-модель по таблице
`x0…xN → y0…yM` и работает в GUI или headless CLI.

Поддерживаются три архитектуры:

- `transformer` – encoder-decoder модель для табличных признаков;
- `mlp` – полносвязный baseline;
- `kan` – Kolmogorov-Arnold Network с обучаемой функцией на каждом ребре.

Полное руководство: [manual.md](manual.md).

## Быстрый старт

```bash
cargo build --release
./target/release/transformer gui
```

Без аргументов приложение тоже открывает GUI. Для CLI без GUI:

```bash
cargo build --release --no-default-features
```

Подготовить Excel/CSV и обучить интерпретируемую KAN:

```bash
./target/release/transformer prepare data.xlsx data.tnum \
  --inputs 3 --outputs 3 --has-header

./target/release/transformer numeric-file data.tnum \
  --model-kind kan --kan-width 16 --kan-layers 2 --kan-grid 16 \
  --lr 3e-3 --epochs 80 --seed 0 \
  --kan-l1 1e-3 --kan-prune 0.05 --kan-finetune-epochs 20 \
  --kan-compact --kan-symbolic --model model-kan.bin
```

Пайплайн KAN: обучение → activation-L1 → hard-prune → fine-tune → structural
compaction → формулы в исходных единицах. Формулы, кривые рёбер и слабые
символьные фиты доступны также в GUI.

## Протокол оценки

Данные делятся на train / validation / test (по умолчанию holdout 70/15/15).
Архитектура, lr, число эпох и порог прунинга подбираются **только по
validation**: поисковые функции не получают test даже по типу. В CLI выбранная
конфигурация затем переобучается на train + validation, и test открывается один
раз. Подробности — в [manual.md](manual.md#протокол-оценки).

## Что входит

- подготовка CSV/TSV/текстовых таблиц и Excel в `.tnum`;
- Optimize, Sweep и Epoch-sweep для сравнения Transformer/MLP/KAN;
- диагностика, предупреждения об экстраполяции и batch-predict в Excel;
- checkpoint-ы с масками прунинга, топологией сжатой KAN и калибровкой для
  формул после загрузки;
- macOS-пакетирование в `.app`, ZIP и DMG.

## Проверка и упаковка

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
./scripts/package-macos.sh
```

Упаковщик использует версию из `Cargo.toml`, создаёт `dist/Transformer.app`,
`dist/Transformer.app.zip` и `dist/Transformer.dmg`, затем проверяет DMG.
