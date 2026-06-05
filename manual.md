# Manual: transformer surrogate

Проект обучает небольшие нейросети на Rust приближать численный расчетный
"черный ящик":

```text
model(inputs) ~= black_box(inputs)
```

Основной режим теперь GUI на `egui`: запуск без подкоманды открывает окно.
CLI остается полноценным вторым интерфейсом для скриптов, автоматизации и
воспроизводимых прогонов.

## Быстрый Старт

GUI:

```bash
cargo run --release
```

CLI на своих данных:

```bash
cargo run --release -- prepare raw.csv train.tnum \
  --inputs 10 \
  --outputs 3 \
  --has-header

cargo run --release -- numeric-file train.tnum \
  --epochs 80 \
  --model model.bin \
  --scheduler warmup-cosine \
  --diagnose

cargo run --release -- predict model.bin 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
```

`predict` принимает входы в исходных физических единицах. Нормализация входов и
выходов хранится внутри `.bin`.

## Что Запускать

Обычный запуск с GUI:

```bash
cargo run --release
```

Явный запуск GUI:

```bash
cargo run --release -- gui
```

Легкая CLI-сборка без `egui`:

```bash
cargo build --release --no-default-features
cargo run --release --no-default-features -- numeric sum --epochs 5
```

Если бинарник собран без default-фич, запуск без подкоманды сообщит, что GUI
недоступен, и покажет CLI-команды.

## GUI

Верхняя строка переключает вкладки. Нижняя строка показывает текущий статус,
ошибки валидации, отмену и завершение задач. Долгие операции выполняются в
worker-потоке; UI остается отзывчивым, а `Cancel` останавливает обучение между
minibatch-ами.

### Train

Вкладка обучает численную surrogate-модель.

Источник данных:

- `Черный ящик` — один из встроенных synthetic blackbox: `sum`, `product`,
  `sine`, `polynomial`, `projectile`.
- `.tnum файл` — ваш датасет, подготовленный вкладкой Prepare или CLI-командой
  `prepare`.

Параметры модели:

- `transformer` — encoder-decoder surrogate. Поддерживает `d_model`, `heads`,
  `layers`, `d_ff`, `value-encoder`.
- `mlp` — простой baseline. Часто очень силен для фиксированной схемы из
  10-15 числовых входов.

`value-encoder` доступен только для transformer:

- `linear` — дефолт, стабильный базовый вариант;
- `mlp` — более гибкое кодирование скаляра;
- `fourier` — sin/cos-базис для периодических или высокочастотных зависимостей.

Для `fourier` подбираются:

- `fourier bands` — число частотных бэндов;
- `fourier scale` — масштаб частот. Это чувствительный гиперпараметр, его лучше
  проверять Sweep/Epoch-sweep.

Параметры обучения:

- `lr` — learning rate;
- `batch` — размер minibatch;
- `epochs` — число эпох;
- `seed` — seed для воспроизводимости;
- `scheduler: warmup-cosine` — включает warmup + cosine decay;
- `warmup` и `min-lr-ratio` — параметры scheduler-а.

Кнопки:

- `Train` — запускает обучение;
- `Cancel` — кооперативно останавливает текущий прогон;
- `Save model…` — сохраняет текущую обученную/загруженную численную модель в
  `.bin`.

После обучения показываются живая кривая train loss и метрики на test в
исходных единицах: `RMSE`, `MAE`, `rel.error`, `R²`. Обученная модель сразу
доступна во вкладках Predict и Diagnose в рамках текущей GUI-сессии. Для
долговременного использования сохраните ее кнопкой `Save model…`.

### Predict

Вкладка делает инференс численной модели.

Варианты:

- обучите модель во вкладке Train, затем перейдите в Predict;
- или нажмите `Загрузить модель (.bin)…` и выберите сохраненный файл.

После загрузки/обучения UI покажет число входов и выходов. Введите `x0..xN` в
исходных единицах и нажмите `Predict`. Выходы `y0..yM` также выводятся в
исходных единицах.

Если вход выходит за диапазон обучающих данных, Predict показывает предупреждение
с номером признака, значением и train-диапазоном. Это не запрет на расчет, но
точность вне train-области не гарантируется.

### Diagnose

Диагностика доступна после обучения модели в текущей GUI-сессии. Для просто
загруженной `.bin` данных train/test нет, поэтому Diagnose не запускается.

Инструменты:

- `Overfit-проба` — свежая модель пытается подогнать маленький train-subset.
  Низкий loss означает, что емкости хватает; высокий loss указывает на underfit.
- `Экстраполяция` — сколько test-строк имеют входы вне train-диапазона.
- `Остаток по входным признакам` — форма ошибки `y_hat - y` по каждому входу.
- `Чувствительность` — для встроенных blackbox оценивает `||dy||/||dx||`.

Как читать:

- низкий overfit loss, плохой test — чаще проблема в данных или покрытии;
- высокий overfit loss — пробуйте MLP, `value-encoder mlp/fourier`, ширину,
  глубину или scheduler;
- много out-of-range — нужны данные в этих областях;
- частая смена знака остатка — вероятна частотная проблема, пробуйте Fourier;
- большой `tail/inner` — ошибка растет в хвостах, проверьте масштаб и данные;
- высокая чувствительность — у задачи может быть физический потолок точности.

### Sweep

Вкладка перебирает сетку конфигов для встроенного blackbox.

Оси задаются CSV-строками:

- `seeds`
- `d_models`
- `layers`
- `d_ffs`
- `lrs`
- `value_encoders`
- `fourier_scales`
- `fourier_bands`
- `schedulers`
- `epochs`
- `batch`

`Run sweep` запускает все комбинации и потоково обновляет ранжированную таблицу.
Лучшая строка помечается первой. Ранжирование идет по среднему `R²`; `nRMSE` и
`rel` показаны для контекста. `Cancel` останавливает свип и оставляет уже
посчитанные строки.

Сейчас Sweep работает для встроенных blackbox-задач. Для `.tnum` используйте
Epoch-sweep для подбора числа эпох и обычный Train для финального обучения.

### Text

Вкладка демонстрирует char-level encoder-decoder LM на текстовом корпусе.

1. Выберите `.txt`.
2. Задайте `d_model`, `heads`, `layers`, `d_ff`, `steps`, `batch`, `ctx_len`,
   `tgt_len`, `lr`, `seed`.
3. Нажмите `Train text`.
4. Следите за perplexity-графиком.
5. После обучения задайте `seed text`, `new chars`, `temperature`, `top_k`,
   `rng seed` и нажмите `Generate`.

Seed text должен быть не короче `ctx_len`, а все символы seed-а должны быть в
словаре обучающего корпуса.

### Prepare

Вкладка конвертирует CSV/TSV/пробельную таблицу в `.tnum`.

Поля:

- `Вход…` — исходная таблица;
- `Выход .tnum…` — куда записать `.tnum`;
- `inputs` — сколько первых колонок являются входами;
- `outputs` — сколько следующих колонок являются выходами;
- `delimiter` — `auto`, `comma`, `tab`, `space`;
- `categorical` — категориальные входы в формате `index:cardinality`;
- `has header` — первая строка является заголовком.

Примеры `categorical`:

```text
3:8
2:5,7:3
```

`3:8` означает, что входная колонка `x3` категориальная с кодами `0..7`.
Категориальные коды должны быть целыми и лежать в диапазоне. Continuous-колонки
нормализуются автоматически при обучении.

### Epoch-sweep

Вкладка подбирает разумное число эпох для `.tnum`.

Поля:

- `.tnum` файл;
- тип модели и архитектура;
- `epochs` — список точек, например `1,2,5,10,20,40`;
- `lr`, `batch`, `seed`, scheduler;
- `target-r2` — остановиться, если достигнут этот R²;
- `min-r2-gain` — минимальный прирост R² между соседними точками;
- `plateau-min-r2` — R², после которого маленький прирост считается плато.

`Run` обучает модель заново для каждой точки из списка эпох, строит живой график
`R²` и train loss, таблицу метрик и рекомендацию остановки. `Save CSV` сохраняет
таблицу результатов.

Если график не помещается в окно, прокрутите вкладку: центральная область GUI
скроллится вертикально.

## CLI

Все CLI-подкоманды запускаются после `--`:

```bash
cargo run --release -- <command> ...
```

### `prepare`

Нативный Rust-порт старого `tools/prepare_numeric_dataset.py`.

```bash
cargo run --release -- prepare <input> <output.tnum> \
  --inputs N \
  --outputs M \
  [--delimiter auto|comma|tab|space] \
  [--has-header] \
  [--categorical 2:5,7:3]
```

Примеры:

```bash
cargo run --release -- prepare raw.csv train.tnum \
  --inputs 10 \
  --outputs 3 \
  --has-header

cargo run --release -- prepare raw.txt train.tnum \
  --inputs 12 \
  --outputs 2 \
  --delimiter space

cargo run --release -- prepare materials.csv train.tnum \
  --inputs 4 \
  --outputs 2 \
  --has-header \
  --categorical 3:8
```

Правила:

- пустые строки и строки-комментарии `#...` пропускаются;
- в каждой строке должно быть ровно `inputs + outputs` чисел;
- числа должны быть конечными;
- категориальные коды должны быть целыми и в диапазоне.

### `numeric`

Обучение на встроенном blackbox:

```bash
cargo run --release -- numeric projectile \
  --epochs 40 \
  --model projectile.bin
```

Доступные blackbox:

```text
sum, product, sine, polynomial, projectile
```

Эти задачи полезны для sanity-check, сравнения MLP/Transformer, Sweep и
проверки scheduler/Fourier.

### `numeric-file`

Обучение на `.tnum`:

```bash
cargo run --release -- numeric-file train.tnum \
  --epochs 80 \
  --model model.bin \
  --diagnose
```

CLI делает:

1. читает `.tnum`;
2. делает детерминированный train/test split 80/20;
3. fit-ит нормализаторы только на train;
4. обучает модель;
5. печатает метрики на test в исходных единицах;
6. если указан `--model`, сохраняет `.bin` и проверяет загрузку.

Старая позиционная форма тоже работает:

```bash
cargo run --release -- numeric-file train.tnum 40 model.bin
```

### `predict`

Инференс из `.bin`:

```bash
cargo run --release -- predict model.bin 0.25 0.75
```

Для модели с 12 входами надо передать 12 чисел:

```bash
cargo run --release -- predict model.bin x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11
```

Если число входов неверное или аргумент не парсится как число, CLI завершится с
понятной ошибкой.

### `diagnose`

Отдельной подкоманды `diagnose` нет. Диагностика включается флагом:

```bash
cargo run --release -- numeric-file train.tnum \
  --epochs 40 \
  --diagnose
```

Для `numeric` sensitivity-проба доступна, потому blackbox вызываемый. Для
`numeric-file` sensitivity-проба пропускается: `.tnum` содержит только сэмплы.

### `epoch-sweep`

Нативный Rust-порт старого `tools/epoch_sweep/epoch_sweep.py`.

```bash
cargo run --release -- epoch-sweep train.tnum \
  --epochs 1,2,5,10,20,40 \
  --target-r2 0.98 \
  --plateau-min-r2 0.90 \
  --min-r2-gain 0.01 \
  --out-dir runs
```

Выводит таблицу:

```text
epochs  train_loss     RMSE       MAE      rel.err     R²
```

Также печатает ASCII-sparkline для `R²` и `loss`, рекомендацию остановки и пишет:

```text
runs/epoch_sweep_results.csv
```

Критерии рекомендации:

- если `R² >= target-r2`, можно остановиться;
- если `R² >= plateau-min-r2`, а прирост меньше `min-r2-gain`, это плато;
- иначе рекомендация будет последняя проверенная точка.

### `sweep`

Перебор конфигов для встроенного blackbox:

```bash
cargo run --release -- sweep projectile \
  --epochs 30 \
  --seeds 0,1,2 \
  --d-models 32,64 \
  --layers-list 2,3 \
  --d-ffs 64,128 \
  --lrs 0.001,0.003 \
  --value-encoders linear,fourier \
  --fourier-scales 1,2,4 \
  --schedulers constant,warmup-cosine
```

Минимальный быстрый прогон:

```bash
cargo run --release -- sweep sum \
  --epochs 1 \
  --d-models 8 \
  --layers-list 1 \
  --d-ffs 16 \
  --lrs 0.001 \
  --value-encoders linear \
  --seeds 0
```

Sweep ранжирует конфиги по среднему `R²`; `rel.error` не используется для
ранжирования, потому плохо ведет себя около нулевых target-значений.

### `text`

CLI-демо char-level text model:

```bash
cargo run --release -- text data/tiny_shakespeare.txt 1500
```

В CLI конфиг text-модели фиксированный, меняется только число шагов. Для
настраиваемых параметров используйте вкладку Text в GUI.

## Флаги Модели И Обучения

Общие флаги:

```text
--epochs N
--model path.bin
--lr VALUE
--batch-size N
--seed N
--diagnose
```

Выбор модели:

```text
--model-kind transformer|mlp
```

Transformer:

```text
--d-model N
--heads N
--layers N
--enc-layers N
--dec-layers N
--d-ff N
```

`--layers` задает сразу encoder и decoder layers, если `--enc-layers` /
`--dec-layers` не указаны отдельно. `d_model` должен делиться на `heads`.

MLP:

```text
--model-kind mlp
--mlp-width N
--mlp-layers N
```

Для MLP `--value-encoder` запрещен, чтобы флаг не игнорировался молча.

Value encoder для transformer:

```text
--value-encoder linear|mlp|fourier
--fourier-bands N
--fourier-scale VALUE
```

Fourier считается по нормализованному continuous-входу и содержит raw-канал,
то есть сохраняет линейный тренд и добавляет sin/cos-базис.

Learning-rate schedule:

```text
--scheduler constant|warmup-cosine
--warmup VALUE
--min-lr-ratio VALUE
```

Для более крупных transformer-моделей часто полезно:

```bash
cargo run --release -- numeric projectile \
  --epochs 60 \
  --d-model 64 \
  --layers 3 \
  --scheduler warmup-cosine \
  --warmup 0.1 \
  --min-lr-ratio 0.1
```

## Формат `.tnum`

Одна строка исходной таблицы:

```text
x0 x1 ... xN y0 y1 ... yM
```

Где:

- `x0..xN` — входы расчетной программы;
- `y0..yM` — выходы расчетной программы;
- число входов и выходов задается при конвертации;
- continuous-входы нормализуются автоматически;
- категориальные входы остаются целыми кодами.

Внутри `.tnum`:

```text
TRNUM1
inputs 2
outputs 1
specs C C
rows 3
data
0.1 0.2 0.3
0.4 0.5 0.9
0.7 0.8 1.5
```

`specs`:

- `C` — continuous-признак;
- `K:8` — категориальный признак с 8 категориями.

Обычно `.tnum` руками писать не нужно: используйте Prepare или CLI `prepare`.

## Где Сохраняется `.bin`

В GUI `.bin` сохраняется туда, что выбрано в file picker после нажатия
`Save model…` во вкладке Train.

В CLI `.bin` сохраняется туда, что передано через `--model`:

```bash
cargo run --release -- numeric-file train.tnum --epochs 40 --model model.bin
```

Файл появится как:

```text
./model.bin
```

Абсолютный путь тоже работает:

```bash
cargo run --release -- numeric-file train.tnum --epochs 40 --model /tmp/model.bin
```

Если `--model` не указан в CLI, обучение и метрики выполнятся, но файл модели не
будет сохранен.

В `.bin` сохраняются:

- веса модели;
- тип модели: `transformer` или `mlp`;
- конфиг архитектуры;
- типы входных признаков (`Continuous` / `Categorical`);
- нормализатор входов;
- нормализатор выходов;
- train-диапазоны входов для предупреждения об экстраполяции;
- число выходов.

Формат бинарный, little-endian, секционный. Неизвестные будущие секции и поля
пропускаются загрузчиком.

## Метрики

После обучения печатаются:

- `train loss` — loss в нормализованном пространстве;
- `RMSE` — среднеквадратичная ошибка в исходных единицах;
- `MAE` — средняя абсолютная ошибка в исходных единицах;
- `rel.error` — относительная ошибка, полезна не всегда;
- `R²` — основная метрика качества на test.

Как понять, что эпох достаточно:

- `R²` близок к требуемому порогу для вашей задачи;
- train loss перестал заметно падать;
- в Epoch-sweep следующая точка дает маленький прирост `R²`;
- test-метрики не улучшаются, хотя train loss продолжает падать;
- Diagnose не показывает большую экстраполяцию test-set.

## Работа Со Своей Расчетной Программой

Статическая regression-задача:

```text
input  = параметры расчета + начальные условия
output = финальный результат / наблюдаемые величины
```

Для RK4 или другой ODE-симуляции, если нужен только финальный результат,
траекторию в модель передавать не нужно. Сгенерируйте таблицу:

```text
param0 param1 ... initial0 initial1 ... final0 final1 ...
```

Если шаг интегрирования `dt` фиксирован, он не является входом модели. Если
время остановки является результатом события, например "тело упало на землю",
это тоже обычно не вход, а возможный дополнительный output.

Рекомендуемый цикл:

1. Определите рабочие диапазоны входов.
2. Сгенерируйте точки внутри диапазонов.
3. Запустите вашу расчетную программу для каждой точки.
4. Запишите `x... y...` в CSV/TSV/txt.
5. Конвертируйте таблицу через Prepare или `prepare`.
6. Подберите число эпох через Epoch-sweep.
7. Обучите финальную модель через Train или CLI `numeric-file`.
8. Проверьте Diagnose.
9. Для production-инференса сохраните `.bin` через GUI `Save model…` или CLI
   `--model`.
10. Используйте Predict или CLI `predict`.

Важно: модель хорошо интерполирует внутри train-области, но не обязана
экстраполировать за ее пределы.

## Если Точность Плохая

Порядок действий:

1. Запустите Diagnose.
2. Проверьте, нет ли out-of-range в test/predict.
3. Сравните transformer и MLP.
4. Подберите число эпох через Epoch-sweep.
5. Для периодических зависимостей попробуйте `value-encoder fourier`.
6. Для больших transformer попробуйте `scheduler warmup-cosine`.
7. Увеличьте `d_model`, `d_ff`, `layers` или MLP width/layers.
8. Добавьте данных в зоны с локальными всплесками ошибки.

Сравнение transformer и MLP:

```bash
cargo run --release -- numeric-file train.tnum --epochs 40 --model-kind transformer
cargo run --release -- numeric-file train.tnum --epochs 40 --model-kind mlp
```

Fourier:

```bash
cargo run --release -- numeric-file train.tnum \
  --epochs 60 \
  --value-encoder fourier \
  --fourier-bands 6 \
  --fourier-scale 2
```

Scheduler:

```bash
cargo run --release -- numeric-file train.tnum \
  --epochs 60 \
  --d-model 64 \
  --layers 3 \
  --scheduler warmup-cosine
```

## Минимальный Пример

Создайте `raw.csv`:

```csv
x0,x1,y0
0.1,0.2,0.3
0.4,0.5,0.9
0.7,0.8,1.5
0.2,0.9,1.1
0.6,0.1,0.7
```

Конвертация:

```bash
cargo run --release -- prepare raw.csv train.tnum \
  --inputs 2 \
  --outputs 1 \
  --has-header
```

Обучение:

```bash
cargo run --release -- numeric-file train.tnum \
  --epochs 40 \
  --model model.bin
```

Предсказание:

```bash
cargo run --release -- predict model.bin 0.25 0.75
```

Для реальной задачи строк должно быть больше: обычно хотя бы сотни или тысячи,
особенно если входов 10-15.

## Старые Python Tools

Файлы в `tools/` оставлены как reference/legacy:

- `tools/prepare_numeric_dataset.py`;
- `tools/epoch_sweep/epoch_sweep.py`.

Основной путь теперь нативный:

- GUI Prepare вместо `prepare_numeric_dataset.py`;
- CLI `prepare` вместо `prepare_numeric_dataset.py`;
- GUI Epoch-sweep вместо `epoch_sweep.py`;
- CLI `epoch-sweep` вместо `epoch_sweep.py`.
