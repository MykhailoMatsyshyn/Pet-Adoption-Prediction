# Структура проекту

Проект для прогнозування ймовірності усиновлення тварин з притулків.

## Дерево каталогу

```
Pract_5
├─ configs
│  ├─ columns.py
│  ├─ model_best_hyperparameters.py
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ columns.cpython-311.pyc
│     ├─ model_best_hyperparameters.cpython-311.pyc
│     └─ __init__.cpython-311.pyc
├─ data
│  ├─ dataset.csv
│  ├─ new_data.csv
│  ├─ new_data_with_actual.csv
│  ├─ prediction_results.csv
│  └─ train_data.csv
├─ docs
│  ├─ ProjectStructure.md
│  └─ Readme.md
├─ models
│  ├─ finalized_model.sav
│  ├─ logs
│  │  ├─ comparison_histogram.png
│  │  ├─ confusion_matrix.png
│  │  ├─ evaluation_metrics.txt
│  │  ├─ feature_importances.csv
│  │  ├─ feature_importances.png
│  │  ├─ prediction_distribution.png
│  │  └─ roc_curve_multiclass.png
│  └─ param_dict.pickle
├─ pipeline
│  ├─ predict.py
│  └─ train.py
├─ requirements.txt
├─ src
│  ├─ notebooks
│  ├─ Pract_1.ipynb
│  ├─ Pract_2.ipynb
│  ├─ Pract_3-4.ipynb
│  ├─ result_lr2.csv
│  └─ variant_3.csv
└─ utils
   ├─ analize.ipynb
   ├─ plotting_functions.py
   ├─ utils for dataset split and save.ipynb
   ├─ __init__.py
   └─ __pycache__
      ├─ plotting_functions.cpython-311.pyc
      └─ __init__.cpython-311.pyc
```

## Опис каталогів і файлів

- **configs/**: Файли конфігурацій, які містять параметри для інженерії ознак і налаштування моделі.
  - `columns.py`: Містить інформацію про використані колонки у наборі даних.
  - `model_best_hyperparameters.py`: Гіперпараметри, використані для тренування моделі.
- **data/**: Містить навчальні, тестові дані, а також файли з результатами.

  - `dataset.csv`: Повний набір даних для тренування та тестування.
  - `new_data.csv`: Нові дані для передбачення.
  - `new_data_with_actual.csv`: Нові дані з доданими фактичними значеннями для порівняння.
  - `prediction_results.csv`: Файл з результатами передбачення.
  - `train_data.csv`: Дані для тренування.

- **docs/**: Документація проекту.

  - `Readme.md`: Основна документація проекту з інформацією про використання, вимоги, і установку.

- **models/**: Містить треновану модель та результати.

  - `logs/`: Логи тренування моделі та візуалізації.
    - `comparison_histogram.png`: Гістограма порівняння реальних і передбачених значень.
    - `confusion_matrix.png`: Матриця плутанини.
    - `evaluation_metrics.txt`: Текстовий файл з метриками оцінки моделі.
    - `feature_importances.csv`: CSV файл з важливістю ознак.
    - `feature_importances.png`: Візуалізація важливості ознак.
    - `prediction_distribution.png`: Розподіл передбачених значень.
    - `roc_curve_multiclass.png`: ROC крива для мультикласової задачі.
  - `finalized_model.sav`: Файл із збереженою моделлю.
  - `param_dict.pickle`: Параметри для обробки даних.

- **pipeline/**: Скрипти для тренування і передбачення моделі.

  - `train.py`: Скрипт для тренування моделі.
  - `predict.py`: Скрипт для передбачення на нових даних.

- **utils/**: Утилітні файли та скрипти.

  - `analize.ipynb`: Jupyter Notebook для аналізу результатів.
  - `plotting_functions.py`: Скрипт з функціями для візуалізації даних.
  - `utils for dataset split and save.ipynb`: Jupyter Notebook з функціями для поділу і збереження даних.

- **requirements.txt**: Файл із залежностями проекту для швидкої установки необхідних бібліотек.
