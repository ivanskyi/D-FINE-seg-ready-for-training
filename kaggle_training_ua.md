# Інструкція з тренування моделі D-FINE на Kaggle

Ця інструкція допоможе вам запустити тренування моделі D-FINE на платформі Kaggle. Kaggle надає безкоштовні GPU (наприклад, T4x2 або P100), що чудово підходить для навчання.

## Крок 1: Клонування проекту та встановлення залежностей

У першій комірці вашого Kaggle Notebook виконайте наступні команди:

```bash
# Переходимо в робочу директорію Kaggle (тут можна зберігати та змінювати файли)
%cd /kaggle/working/

# Клонуємо репозиторій
!git clone https://github.com/ivanskyi/D-FINE-seg-ready-for-training /kaggle/working/D-FINE-seg

%cd /kaggle/working/D-FINE-seg

# Встановлення залежностей через uv
!pip install uv
!uv sync
```

## Крок 2: Підготовка датасету

У Kaggle найзручніше завантажувати дані через "Add Data" (вкладка справа) -> "Upload Dataset". Припустимо, ви завантажили ваш архів `dataset.zip` як датасет, його шлях буде в `/kaggle/input/ВАШ_ДАТАСЕТ_В_KAGGLE/`.

Розпакуємо його в робочу директорію, щоб мати можливість з ним працювати:

```bash
# 1) видалити старий датасет (якщо є)
!rm -rf /kaggle/working/my_dataset

# 2) створити чисту папку
!mkdir -p /kaggle/working/my_dataset

# 3) розпакувати новий архів (змініть шлях до вашого zip файлу)
!unzip -q "/kaggle/input/ВАШ_ДАТАСЕТ_В_KAGGLE/dataset.zip" -d "/kaggle/working/my_dataset"

# 4) перевірка
!ls /kaggle/working/my_dataset
print("✅ dataset fully reset and extracted")
```
*(Якщо ви додали вже розпакований датасет через Kaggle Datasets, його шлях буде приблизно `/kaggle/input/dataset-name`. Тоді цей крок розпакування не потрібен — просто вкажіть цей шлях у `train.data_path` на Кроці 5).*

## Крок 3: Валідація датасету

Перевіримо, чи датасет зібраний правильно (виконайте в Python-комірці):

```python
import os

# Якщо ви розпакували дані у working:
root = "/kaggle/working/my_dataset/"
required = ["train.json", "val.json", "test.json", "images"]

for item in required:
    path = os.path.join(root, item)
    if not os.path.exists(path):
        raise Exception(f"❌ Missing: {path}")

print("✅ dataset structure valid")
```

## Крок 4: Завантаження найбільшої попередньо натренованої моделі (dfine_x)

Завантажуємо ваги моделі в папку `pretrained`:

```bash
!mkdir -p /kaggle/working/D-FINE-seg/pretrained
!wget https://github.com/Peterande/D-FINE/releases/download/v2.0/dfine_x_coco.pt -O /kaggle/working/D-FINE-seg/pretrained/dfine_x_coco.pt
```

## Крок 5: Запуск тренування

Запускаємо процес тренування. Налаштування включають захист від фрагментації пам'яті (що особливо критично для GPU на Kaggle), 30 епох, розмір 640x640:

```bash
%cd /kaggle/working/D-FINE-seg

# Залишаємо захист від фрагментації пам'яті
%env HYDRA_FULL_ERROR=1
%env PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

!python -m src.dl.train \
    model_name=x \
    task=segment \
    train.device=cuda \
    train.epochs=30 \
    train.batch_size=12 \
    train.num_workers=4 \
    train.use_wandb=False \
    train.coco_dataset=True \
    train.use_one_class=False \
    train.pretrained_dataset=coco \
    train.pretrained_model_path=/kaggle/working/D-FINE-seg/pretrained/dfine_x_coco.pt \
    train.img_size=[640,640] \
    train.keep_ratio=True \
    train.label_to_name="{1:'car', 2:'interior', 3:'license-plate', 4:'podium'}" \
    train.mosaic_augs.mosaic_prob=0.0 \
    train.augs.rotation_p=0.0 \
    train.root=/kaggle/working/D-FINE-seg \
    train.data_path=/kaggle/working/my_dataset \
    train.path_to_save=/kaggle/working/D-FINE-seg/output/models \
    train.debug_img_path=/kaggle/working/D-FINE-seg/output/debug_images \
    train.eval_preds_path=/kaggle/working/D-FINE-seg/output/eval_preds \
    train.bench_img_path=/kaggle/working/D-FINE-seg/output/bench_imgs \
    train.infer_path=/kaggle/working/D-FINE-seg/output/infer \
    +train.gradient_accumulation_steps=1
```

## Важливо для Kaggle: Збереження результатів
Щоб не втратити чекпоінти після завершення або переривання сесії:
- Рекомендується запускати тренування у фоні: використовуйте кнопку **Save Version** (у правому верхньому куті) -> **Save & Run All (Commit)**. 
- Проект адаптований для збереження ваг після кожної епохи (`epoch_X.pt`). Вони будуть збережені у папці `/kaggle/working/D-FINE-seg/output/models`.
- Після завершення комміту (Save & Run All), ви зможете завантажити збережені моделі з вкладки **"Output"** вашого Kaggle Notebook. Усі помилкові анотації, що виходять за межі, автоматично ігноруватимуться під час тренування.
