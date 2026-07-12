# Руководство пользователя: Transformer / KAN

Проект обучает surrogate-модели для численного «чёрного ящика»:

```text
model(inputs) ≈ black_box(inputs)
```

Основной интерфейс – GUI на `egui`; CLI равноправен для автоматизации и
воспроизводимых прогонов. Полный список кратких команд есть в
[README.md](README.md).

## Запуск

Нужен стабильный Rust toolchain.

```bash
cargo build --release
./target/release/transformer          # GUI
./target/release/transformer gui      # GUI явно
```

Для headless CLI без зависимостей GUI:

```bash
cargo build --release --no-default-features
./target/release/transformer numeric sum --epochs 5
```

## Данные и `.tnum`

Одна строка исходной таблицы имеет вид:

```text
x0 x1 ... xN y0 y1 ... yM
```

Continuous-входы нормализуются по train-части z-score преобразованием;
категориальные входы остаются целыми кодами. Выходы всегда нормализуются на
время обучения и возвращаются в исходные единицы для метрик и предсказаний.

Подготовка таблицы:

```bash
./target/release/transformer prepare raw.xlsx train.tnum \
  --inputs 10 --outputs 3 --has-header
```

`prepare` и GUI **Prepare** принимают CSV, TSV, пробельные таблицы и Excel
(`.xlsx`, `.xlsm`, `.xlsb`, `.xls`, `.ods`). Для текстовых таблиц пустые строки
и комментарии после `#` пропускаются. Входы-категории задаются так:

```bash
--categorical 2:5,7:3
```

Здесь `x2` содержит коды `0..4`, а `x7` – коды `0..2`. Коды обязаны быть
целыми и находиться в указанном диапазоне. Если заголовок имеет схему
`x0,x1,...,y0,y1`, Prepare умеет определить число входов, выходов и простые
поля вроде `material_id` автоматически.

## Архитектуры

Выбор производится флагом `--model-kind transformer|mlp|kan` или в **Train**.

| Модель | Когда начинать с неё | Основные параметры |
| --- | --- | --- |
| Transformer | сложные табличные зависимости, Fourier-кодирование | `d_model`, `heads`, `layers`, `d_ff`, value encoder |
| MLP | сильный простой baseline для фиксированного числа числовых входов | `mlp-width`, `mlp-layers` |
| KAN | важна компактность и интерпретируемость функций | `kan-width`, `kan-layers`, `kan-grid` |

`value-encoder linear|mlp|fourier` доступен только Transformer. Для MLP и KAN
он принудительно линейный: неподходящий флаг является ошибкой, а не молча
игнорируется.

### KAN

В KAN нелинейность расположена на ребре, а не в фиксированной активации узла:

```text
φ(x) = w_base · gelu(x) + Σ_t w_spline,t · B_t(x)
```

`B_t` – кубические B-сплайны на сетке `[-3, 3]` в нормализованном пространстве.
Базовая ветка `gelu` даёт гладкую экстраполяцию за пределами сплайн-сетки.

Типичный CLI-проход:

```bash
./target/release/transformer numeric-file train.tnum \
  --model-kind kan --kan-width 16 --kan-layers 2 --kan-grid 16 \
  --lr 3e-3 --epochs 80 --seed 0 \
  --kan-l1 1e-3 --kan-prune 0.05 --kan-finetune-epochs 20 \
  --kan-compact --kan-symbolic --model model-kan.bin
```

Фазы выполняются в таком порядке:

1. `--kan-l1 λ` добавляет activation-L1: среднее `|φ|` каждого ребра по
   реальным активациям всех слоёв.
2. `--kan-prune θ` удаляет рёбра, чья p95-важность ниже `θ` от максимума в
   слое; маска блокирует и вклад, и градиент.
3. `--kan-finetune-epochs N` дообучает оставшиеся рёбра с `λ=0`.
4. `--kan-compact` физически вырезает неиспользуемые скрытые узлы и постоянные
   узлы, сворачивая их вклад в bias следующего слоя. Предсказания не меняются,
   а число параметров уменьшается.
5. `--kan-symbolic` подбирает к активным рёбрам примитивы `x`, `x²`, `x³`,
   `sqrt`, `exp`, `log`, `1/x`, `sin`, `tanh` в аффинной форме
   `a·f(bx+c)+d` и печатает послойные формулы.

Символьные формулы используют **исходные единицы** таблицы: в них можно
подставлять значения прямо из Excel. Внутренние `hK_J` – безразмерные скрытые
узлы. Отчёт показывает R² каждой подгонки и выделяет рёбра с `R² < 0.99`;
поэтому формула всегда сопровождается её измеримой точностью, а не выдаётся за
тождество исходной сети.

## GUI

Долгие операции исполняются в worker-потоке. **Cancel** останавливает
обучение, sweep и optimize кооперативно между minibatch-ами.

### Train и Predict

Во **Train** выберите встроенный black-box (`sum`, `product`, `sine`,
`polynomial`, `projectile`) либо `.tnum`. Задайте модель, её архитектуру,
`lr`, `batch`, `epochs`, `seed` и scheduler. После обучения показаны train
loss и test-метрики в исходных единицах: RMSE, MAE, относительная ошибка и R².

Во **Predict** можно ввести один набор `x0…xN` в физических единицах или
загрузить модель из `.bin`. Кнопка **Заполнить Excel** принимает первый лист с
колонками `x0…xN` и `y0…yM`, вычисляет все строки и записывает новый `.xlsx`.
Если вход вышел за train-диапазон, UI покажет предупреждение об экстраполяции.

### KAN curves и KAN formulas

- **KAN curves** показывает `φ(input → output)` выбранного ребра. На первом
  слое ось X нормализована; на глубоких слоях это активация предыдущего слоя.
- **KAN formulas** запускает symbolic extraction в worker, выводит копируемые
  формулы в исходных единицах, R² формульной модели против R² KAN и таблицу
  слабых рёбер.

Для формул нужны реальные калибровочные входы. Новые KAN-checkpoint-ы хранят
до 256 сырых train-строк, поэтому формулы доступны и после загрузки. У
загруженной модели нет test-набора: UI в этом случае честно показывает только
качество фита рёбер, а не test-метрики. Старый checkpoint без calibration
нужно пересохранить после обучения на исходном `.tnum`.

### Diagnose, Optimize, Sweep и Epoch-sweep

**Diagnose** доступна после обучения в текущей GUI-сессии и включает
overfit-пробу, проверку экстраполяции test-строк, диагностику остатков и для
встроенных задач – чувствительность.

**Optimize** подбирает конфигурацию на `.tnum`. В пресетах Quick, Balanced и
Deep можно включить Transformer, MLP и KAN. Ранжирование поддерживает
`worst-output R²`, aggregate/mean R² и nRMSE. Лучший конфиг переносится в
Train или Epoch-sweep.

**Sweep** перебирает конфигурации на встроенных black-box задачах.
**Epoch-sweep** обучает одну воспроизводимую модель до максимальной эпохи и снимает
метрики в заданных контрольных точках – это исключает шум от переинициализации
между точками. Рекомендация остановки выбирает первую точку с целевым R²,
иначе плато по `ΔR²`, иначе последнюю точку. Результаты можно сохранить в CSV.

**Text** – отдельная демонстрация char-level encoder-decoder LM; он не связан
с численным pipeline.

## CLI

Все подкоманды запускаются так:

```bash
./target/release/transformer <command> ...
```

### Обучение и предсказание

```bash
# Встроенная задача
./target/release/transformer numeric projectile --epochs 40 --model projectile.bin

# Свой .tnum
./target/release/transformer numeric-file train.tnum \
  --model-kind mlp --mlp-width 128 --mlp-layers 3 \
  --epochs 80 --lr 1e-3 --model model.bin

# Инференс в исходных единицах
./target/release/transformer predict model.bin 0.25 0.75
```

Общие флаги: `--epochs`, `--model`, `--lr`, `--batch-size`, `--seed`,
`--scheduler constant|warmup-cosine`, `--warmup`, `--min-lr-ratio` и
`--diagnose`. CLI строго отвергает неизвестные и несовместимые флаги.

### Sweep и Epoch-sweep

```bash
./target/release/transformer sweep projectile \
  --model-kinds transformer,mlp,kan \
  --d-models 32,64 --layers-list 2 \
  --mlp-widths 64,128 --mlp-layers-list 3 \
  --kan-widths 8,16 --kan-layers-list 2 --kan-grids 8,16 \
  --lrs 1e-3,3e-3 --epochs 80 --batch-size 64 --seeds 0,1

./target/release/transformer epoch-sweep train.tnum \
  --model-kind kan --kan-width 16 --kan-layers 2 --kan-grid 16 \
  --epochs 20,40,60,80,120 --target-r2 0.99 \
  --plateau-min-r2 0.95 --min-r2-gain 0.002 --out-dir runs
```

`epoch-sweep` пишет `runs/epoch_sweep_results.csv` и печатает sparklines
loss/R². KAN-операции `--kan-l1`, `--kan-prune`, `--kan-finetune-epochs`,
`--kan-compact` и `--kan-symbolic` намеренно не поддерживаются внутри
epoch-sweep: это разные фазы final-training, которые нельзя молча смешивать с
поиском числа эпох.

## Checkpoint-ы

`.bin` – секционный little-endian формат. Он хранит тип модели, конфигурацию,
параметры, спецификации входов, нормализаторы и train-диапазоны. KAN добавляет
hard-prune маски, реальные размеры слоёв после compaction и калибровочную
выборку для формул. Невалидные размеры сжатой топологии и калибровка неверной
формы отвергаются при загрузке.

Старые checkpoint-ы остаются читаемыми: отсутствие новых KAN-секций означает
старое поведение. После загрузки старой KAN можно предсказывать; для формул её
нужно пересохранить новой версией после обучения с исходными данными.

## Метрики и практический цикл

R² – основная метрика качества на test. RMSE и MAE измеряются в исходных
единицах, относительная ошибка может быть нестабильной возле нулевого target.

Практический цикл:

```text
Prepare → Optimize → Epoch-sweep → Train → (KAN: prune/compact/formulas) → Save → Predict
```

Если качество недостаточно, сначала проверьте Diagnose и покрытие train-
диапазона, затем сравните все три архитектуры через Optimize/Sweep. Для
периодических зависимостей Transformer с Fourier-encoder может быть полезен;
для читаемой компактной модели начните с KAN и увеличивайте `kan-grid` раньше,
чем ширину или глубину.

## Старая Python-автоматизация

В `tools/` сохранены legacy-скрипты для совместимости:

| Скрипт | Rust UI/CLI | Что остаётся только в Python |
| --- | --- | --- |
| `tools/prepare_numeric_dataset.py` | Полностью покрыт и расширен: Excel, авто-схема, GUI Prepare, строгая валидация `.tnum` | Нет функционального преимущества |
| `tools/epoch_sweep/epoch_sweep.py` | Покрыты обучение, CSV, рекомендация остановки, live-график GUI; Rust использует одну траекторию обучения вместо независимых переинициализаций | Автоматический PNG графика и отдельный `.bin` на каждую контрольную эпоху |

Python epoch-sweep парсит текстовый вывод внешнего бинарника и не знает о
современных KAN-фазах. Используйте Rust-интерфейсы по умолчанию. Legacy-скрипт
имеет смысл только если нужен именно PNG-артефакт или набор checkpoint-ов для
каждой контрольной эпохи.

## Проверка и macOS-пакет

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
./scripts/package-macos.sh
```

Скрипт создаёт `dist/Transformer.app`, ZIP и DMG, проверяет DMG через
`hdiutil` и берёт версию приложения из `Cargo.toml`. Для обработки иконки
нужен Pillow для `python3`.
