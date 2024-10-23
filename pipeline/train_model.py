import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import columns
import model_best_hyperparameters

# Завантажуємо тренувальні і тестові дані
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Визначаємо цільові змінні та ознаки
X_train = train_data[columns.X_columns]
y_train = train_data[columns.y_column]
X_test = test_data[columns.X_columns]
y_test = test_data[columns.y_column]

# Створюємо і тренуємо модель RandomForest
rf = RandomForestClassifier(**model_best_hyperparameters.params)
rf.fit(X_train, y_train)

# Прогнозуємо на тестових даних
y_pred = rf.predict(X_test)

# Виводимо метрики оцінки
print("Звіт про метрики класифікації:")
print(metrics.classification_report(y_test, y_pred))

# Зберігаємо модель у файл
with open('models/finalized_model.sav', 'wb') as model_file:
    pickle.dump(rf, model_file)