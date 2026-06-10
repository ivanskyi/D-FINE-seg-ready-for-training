# Інструкція з тренування моделі D-FINE на Google Colab

Ця інструкція допоможе вам швидко та легко запустити тренування моделі D-FINE для сегментації номерних знаків (або інших об'єктів) на Google Colab. 

Проект уже налаштований для **ігнорування поламаних анотацій** (що виходять за межі) та автоматично **зберігає ваги після кожної епохи**. Структура вашого датасету (COCO формат: `images/`, `train.json`, `val.json`, `test.json`) повністю підходить для тренування.

## Крок 1: Клонування проекту та встановлення залежностей

У першій комірці Google Colab виконайте наступні команди:

```bash
# Якщо ваш код знаходиться на GitHub, клонуйте його:
!git clone https://github.com/ivanskyi/D-FINE-seg-ready-for-training /content/D-FINE-seg
# Якщо ви завантажуєте проект архівом, розпакуйте його в папку /content/D-FINE-seg

%cd /content/D-FINE-seg

# Встановлення залежностей через uv (за наявності pyproject.toml / uv.lock)
!pip install uv
!uv sync
```

## Крок 2: Розпакування датасету з Google Drive

Підключіть ваш Google Drive (зазвичай це робиться через бічну панель Colab або скриптом `from google.colab import drive; drive.mount('/content/drive')`), після чого виконайте скрипт:

```bash
# 1) видалити старий датасет (якщо є)
!rm -rf /content/my_dataset

# 2) створити чисту папку
!mkdir -p /content/my_dataset

# 3) розпакувати новий архів
!unzip -q "/content/drive/MyDrive/dataset/dataset.zip" -d "/content/my_dataset"

# 4) перевірка
!ls /content/my_dataset
print("✅ dataset fully reset and extracted")
```

## Крок 3: Валідація датасету

Перевіримо, чи всі необхідні файли на місці (можете виконати це в окремій Python-комірці):

```python
import os

root = "/content/my_dataset/"
required = ["train.json", "val.json", "test.json", "images"]

for item in required:
    path = os.path.join(root, item)
    if not os.path.exists(path):
        raise Exception(f"❌ Missing: {path}")

print("✅ dataset structure valid")
```

## Крок 4: Завантаження найбільшої попередньо натренованої моделі (dfine_x)

Оскільки ми тренуватимемо найбільшу модель (`x`), завантажимо її ваги в папку `pretrained`:

```bash
!mkdir -p /content/D-FINE-seg/pretrained
!wget https://github.com/Peterande/D-FINE/releases/download/v2.0/dfine_x_coco.pt -O /content/D-FINE-seg/pretrained/dfine_x_coco.pt
```

## Крок 5: Запуск тренування

Запускаємо тренування. Налаштування включають розмір зображення 640x640, 30 епох, та захист від фрагментації пам'яті:

```bash
%cd /content/D-FINE-seg

# Залишаємо захист від фрагментації, розширюємо ліміти
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
    train.pretrained_model_path=/content/D-FINE-seg/pretrained/dfine_x_coco.pt \
    train.img_size=[640,640] \
    train.keep_ratio=True \
    train.label_to_name="{1:'car', 2:'interior', 3:'license-plate', 4:'podium'}" \
    train.mosaic_augs.mosaic_prob=0.0 \
    train.augs.rotation_p=0.0 \
    train.root=/content/D-FINE-seg \
    train.data_path=/content/my_dataset \
    train.path_to_save=/content/D-FINE-seg/output/models \
    train.debug_img_path=/content/D-FINE-seg/output/debug_images \
    train.eval_preds_path=/content/D-FINE-seg/output/eval_preds \
    train.bench_img_path=/content/D-FINE-seg/output/bench_imgs \
    train.infer_path=/content/D-FINE-seg/output/infer \
    +train.gradient_accumulation_steps=1
```

### Особливості, які вже впроваджено в код проекту:
1. **Збереження кожної епохи**: Код у файлі `src/dl/train.py` було оновлено, тому після кожної епохи буде зберігатися окремий файл `epoch_{X}.pt` у папці `/content/D-FINE-seg/output/models`.
2. **Пропуск поламаних анотацій**: У файл `src/dl/dataset.py` додано `try-except` блок при трансформації анотацій. Якщо через якісь проблеми (наприклад, полігони вийшли за межі зображення) падає помилка, зображення безпечно ігнорується, і процес тренування триває далі без збоїв.
