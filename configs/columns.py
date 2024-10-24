# Цільова змінна
y_column = ['AdoptionSpeed']

# Ознаки
X_columns = ['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength',
             'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity',
             'Fee', 'VideoAmt', 'PhotoAmt', 'ColorName_x', 
             'ColorName_y', 'ColorName', 'BreedName_x', 
             'BreedName_y', 'StateName_x']

#=============================================================================================

# Колонки для випадкової імпутації
random_impute_columns = ['Gender', 'MaturitySize']

# Колонки для заповнення пропущених значень за допомогою Forward Fill і Backward Fill
fillna_ffill_bfill_columns = ['ColorName']

# Колонки для заміни відсутніх даних на 'Unknown'
fillna_unknown_columns = ['BreedName_x', 'BreedName_y']

#=============================================================================================

# Колонки для обробки аномальних значень
outlier_columns = ['Age']

# Колонки для One-Hot Encoding
one_hot_columns = ['ColorName_x', 'ColorName_y', 'ColorName']

# Колонки для Integer Encoding
breed_columns = ['BreedName_x', 'BreedName_y']
state_column = ['StateName_x']

# Колонки для масштабування (Normalization/Standardization)
scaling_columns = ['Age', 'Quantity', 'Fee', 'PhotoAmt']