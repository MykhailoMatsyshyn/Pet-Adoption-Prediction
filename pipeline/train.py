import sys
import os

import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import TomekLinks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs import columns
from configs import model_best_hyperparameters
from utils.plotting_functions import plot_confusion_matrix, plot_roc_curve_multiclass

# Завантаження даних для тренування
ds = pd.read_csv("data/train_data.csv")

#################################### feature engineering ####################################

# Видалення непотрібних колонок
columns_to_drop = ['Unnamed: 0', 'RescuerID', 'PetID', 'Name', 'Description', 'State']
ds = ds.drop(columns=columns_to_drop)

################## Missing data imputation

def impute_na(df, variable, value):
    return df[variable].fillna(value)

# Випадкова заміна
def random_impute(column):
    non_null_values = column.dropna()
    return column.apply(lambda x: np.random.choice(non_null_values) if pd.isna(x) else x)

for col in columns.random_impute_columns:
    ds[col] = random_impute(ds[col])

# Заміна за допомогою Forward Fill і Backward Fill
for col in columns.fillna_ffill_bfill_columns:
    ds[col + '_Filled'] = ds[col].ffill()
    ds[col + '_Filled'] = ds[col + '_Filled'].bfill()

# Заміна пропущених значень на 'Unknown'
for col in columns.fillna_unknown_columns:
    ds[col] = impute_na(ds, col, 'Unknown')

################## Outlier Engineering

# Функція для знаходження меж для аномальних значень (Outliers) на основі IQR
def find_skewed_boundaries(df, variable, distance):
    df[variable] = pd.to_numeric(df[variable], errors='coerce') 
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25) 
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance) 
    return upper_boundary, lower_boundary

# Визначення меж аномальних значень для кожної колонки з outlier_columns
upper_lower_limits = dict()
for col in columns.outlier_columns:
    upper_lower_limits[col + '_upper_limit'], upper_lower_limits[col + '_lower_limit'] = find_skewed_boundaries(ds, col, 1.5)

# Видалення рядків з аномальними значеннями
for col in columns.outlier_columns:
    upper = upper_lower_limits[col + '_upper_limit']
    lower = upper_lower_limits[col + '_lower_limit']
    ds = ds[(ds[col] >= lower) & (ds[col] <= upper)]


################## Categorical encoding

# Ініціалізуємо словник для збереження мапінгу та параметрів кодування
map_dicts = dict()

# 1. One-Hot Encoding для колонок 'ColorName_x', 'ColorName_y', 'ColorName'
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_color = one_hot_encoder.fit_transform(ds[columns.one_hot_columns])

encoded_color_df = pd.DataFrame(encoded_color, columns=one_hot_encoder.get_feature_names_out(columns.one_hot_columns))

encoded_color_df.index = ds.index

ds = pd.concat([ds, encoded_color_df], axis=1)
ds.drop(columns=columns.one_hot_columns, inplace=True)

# Оновлюємо список ознак
new_one_hot_columns = one_hot_encoder.get_feature_names_out(columns.one_hot_columns).tolist()
columns.X_columns = [col for col in columns.X_columns if col not in columns.one_hot_columns] + new_one_hot_columns

map_dicts['one_hot_encoder'] = one_hot_encoder


# 2. Integer Encoding для 'BreedName_x', 'BreedName_y'
unique_breeds = pd.unique(ds[columns.breed_columns].values.ravel('K'))
breed_mapping = {breed: idx for idx, breed in enumerate(unique_breeds)}
ds[columns.breed_columns[0]] = ds[columns.breed_columns[0]].map(breed_mapping)
ds[columns.breed_columns[1]] = ds[columns.breed_columns[1]].map(breed_mapping)

map_dicts['breed_mapping'] = breed_mapping


# 3. Integer Encoding для 'StateName_x'
ordinal_mapping = {k: i for i, k in enumerate(ds[columns.state_column[0]].unique(), 0)}
ds[columns.state_column[0]] = ds[columns.state_column[0]].map(ordinal_mapping)

# Зберігаємо мапінг для 'StateName_x'
map_dicts['state_name_mapping'] = ordinal_mapping


################## Scaling to Minimum and Maximum values

scaler = MinMaxScaler()

ds[columns.scaling_columns] = scaler.fit_transform(ds[columns.scaling_columns])

map_dicts['scaler'] = scaler


################## save parameters

# Збереження параметрів для майбутнього використання
param_dict = {
    'upper_lower_limits': upper_lower_limits,
    'map_dicts': map_dicts
}

# Збереження параметрів у файл
with open('models/param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#################################### Тренування моделі ####################################
print(ds.columns)
print(columns.X_columns)

# Визначаємо цільову змінну та ознаки
X = ds[columns.X_columns]
y = ds[columns.y_column]

# Розділяємо дані на тренувальний і тестовий набір у співвідношенні 90:10
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

# Використовуємо Tomek Links для балансування
tomek_links = TomekLinks()
X_train, y_train = tomek_links.fit_resample(X_train, y_train)

y_train = y_train.values.ravel()

# Створюємо та тренуємо модель RandomForest
rf = RandomForestClassifier(**model_best_hyperparameters.params)
rf.fit(X_train, y_train)

# Робимо передбачення на тестовому наборі
y_pred = rf.predict(X_test)

# Виводимо метрики для тестового набору
print('\n\nОцінка на тестовому наборі: \n', metrics.classification_report(y_test, y_pred, zero_division=0))

with open('models/logs/evaluation_metrics.txt', 'w') as f:
    f.write(metrics.classification_report(y_test, y_pred, zero_division=0))

importances = rf.feature_importances_
feature_importances = pd.DataFrame(importances, index=columns.X_columns, columns=['importance']).sort_values('importance', ascending=False)
feature_importances.to_csv('models/logs/feature_importances.csv', index=True)

import matplotlib.pyplot as plt
import pandas as pd

# Отримуємо важливість ознак з моделі RandomForest
importances = rf.feature_importances_
feature_names = columns.X_columns

# Створюємо DataFrame для зручного відображення
feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Сортуємо за важливістю
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Побудова діаграми важливості ознак
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
plt.xlabel('Важливість')
plt.title('Важливість ознак (Random Forest)')
plt.gca().invert_yaxis()

# Зберігаємо діаграму у файл
plt.savefig('models/logs/feature_importances.png')

# Виведення та збереження матриці плутанини
plot_confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4], save_path='models/logs/confusion_matrix.png')

# Отримуємо ймовірності для кожного класу
y_score = rf.predict_proba(X_test)

# Виведення та збереження ROC-кривої
plot_roc_curve_multiclass(y_test, y_score, classes=[0, 1, 2, 3, 4], save_path='models/logs/roc_curve_multiclass.png')

# Зберігаємо треновану модель
filename = 'models/finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))

###########################################################################################
