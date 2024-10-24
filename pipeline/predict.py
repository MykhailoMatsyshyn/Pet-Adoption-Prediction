import sys
import os

import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs import columns
from configs import model_best_hyperparameters

ds = pd.read_csv("data/new_data.csv")

# Перевірка на наявність порожнього набору даних
if ds.empty:
print("Помилка: Набір вхідних даних порожній. Будь ласка, надайте коректні дані для передбачення.")
sys.exit(1)

print('Розмір нових даних:', ds.shape)

# Завантаження параметрів для інженерії ознак
param_dict = pickle.load(open('models/param_dict.pickle', 'rb'))

#################################### feature engineering ####################################

# Імпутація відсутніх даних
def impute_na(df, variable, value):
    return df[variable].fillna(value)

def random_impute(column):
    non_null_values = column.dropna()
    return column.apply(lambda x: np.random.choice(non_null_values) if pd.isna(x) else x)

# Випадкова заміна пропущених значень
for col in columns.random_impute_columns:
    ds[col] = random_impute(ds[col])

# Заміна пропущених значень за допомогою Forward Fill і Backward Fill
for col in columns.fillna_ffill_bfill_columns:
    ds[col + '_Filled'] = ds[col].ffill()
    ds[col + '_Filled'] = ds[col + '_Filled'].bfill()

# Заміна пропущених значень значенням 'Unknown'
for col in columns.fillna_unknown_columns:
    ds[col] = impute_na(ds, col, 'Unknown')

# Обробка викидів
for column in columns.outlier_columns:
    ds[column] = ds[column].astype(float)
    ds = ds[~ np.where(ds[column] > param_dict['upper_lower_limits'][column+'_upper_limit'], True,
                       np.where(ds[column] < param_dict['upper_lower_limits'][column+'_lower_limit'], True, False))]

################## Кодування категоріальних ознак

# One-Hot Encoding
one_hot_encoder = param_dict['map_dicts']['one_hot_encoder']

# Виконання кодування
encoded_color = one_hot_encoder.transform(ds[columns.one_hot_columns])

# Створення нових колонок для закодованих значень
encoded_color_df = pd.DataFrame(encoded_color, columns=one_hot_encoder.get_feature_names_out(columns.one_hot_columns))

# Оновлення індексів для коректного поєднання з основним датафреймом
encoded_color_df.index = ds.index

# Додавання нових закодованих колонок до основного датафрейму
ds = pd.concat([ds, encoded_color_df], axis=1)

# Видалення старих категоріальних колонок, які вже закодовані
ds.drop(columns=columns.one_hot_columns, inplace=True)

# Оновлення списку колонок ознак для подальшого використання в моделі
columns.X_columns = [col for col in columns.X_columns if col not in columns.one_hot_columns] + list(encoded_color_df.columns)

# Інтегральне кодування для 'BreedName_x', 'BreedName_y'
ds[columns.breed_columns[0]] = ds[columns.breed_columns[0]].map(param_dict['map_dicts']['breed_mapping'])
ds[columns.breed_columns[1]] = ds[columns.breed_columns[1]].map(param_dict['map_dicts']['breed_mapping'])

# Інтегральне кодування для 'StateName_x'
ds[columns.state_column[0]] = ds[columns.state_column[0]].map(param_dict['map_dicts']['state_name_mapping'])

# Масштабування ознак
scaler = param_dict['map_dicts']['scaler']
ds[columns.scaling_columns] = scaler.transform(ds[columns.scaling_columns])

#################################### передбачення ####################################

print(columns.X_columns)

# Визначення ознак
X = ds[columns.X_columns]

# Завантаження моделі
rf = pickle.load(open('models/finalized_model.sav', 'rb'))

print(X)
print(rf.predict(X))

# Виконання передбачення
ds['AdoptinSpeed_pred'] = rf.predict(X)

# Збереження передбачення у файл
ds.to_csv('data/prediction_results.csv', index=False)

print("Передбачення завершено. Результати збережено в 'data/prediction_results.csv'")
