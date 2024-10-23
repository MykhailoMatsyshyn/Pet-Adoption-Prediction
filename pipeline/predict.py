import pickle
import pandas as pd
import columns

# Завантажуємо дані
ds = pd.read_csv("data/test.csv")

# Завантажуємо натреновану модель
rf = pickle.load(open('models/finalized_model.sav', 'rb'))

# Визначаємо ознаки
X = ds[columns.X_columns]

# Реальні значення цільової змінної
y_true = ds[columns.y_column].astype(int).values

# Прогнозуємо
y_pred = rf.predict(X).astype(int) 

# Додаємо передбачення до даних і зберігаємо результати
ds['AdoptionSpeed_pred'] = y_pred
ds = ds.drop(columns=['AdoptionSpeed'])
ds.to_csv('data/prediction.csv', index=False)

accuracy = (y_pred == y_true).mean()
print(f"Точність моделі: {accuracy:.2%}")
