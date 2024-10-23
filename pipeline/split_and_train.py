import pandas as pd
from sklearn.model_selection import train_test_split

# Завантажуємо вже підготовлений датасет
ds = pd.read_csv("data/prepared_dataset.csv")

# Розділяємо дані у співвідношенні 90:10 (train:test)
train_data, test_data = train_test_split(ds, train_size=0.9, test_size=0.1, stratify=ds['AdoptionSpeed'])

# Зберігаємо розділені дані
train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)