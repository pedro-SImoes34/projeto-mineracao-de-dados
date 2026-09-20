import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/creditcard_amostra.csv')

#Para uma melhor visualização e entendimento humano, transformamos o tempo da transação de segundos para horas
# %24 - especificamos exatamente a hora do dia em que a transação ocorreu
df['Hour'] = (df['Time'] / 3600) % 24 

x = df.drop(columns = ['Class', 'Time'])
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.20, random_state = 42, stratify = y
)

scaler = StandardScaler()
x_train[['Amount', 'Hour']] = scaler.fit_transform(x_train[['Amount', 'Hour']])
x_test[['Amount', 'Hour']] = scaler.fit_transform(x_test[['Amount', 'Hour']])

print(f"Proporção de classes antes do SMOTE:\n {y_train.value_counts()}")

#criação de transações fictícias para que tenha um balanceamento de 50% entre as transações reais e fraudulentas
smote = SMOTE(random_state = 42, k_neighbors = 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

print(f"\n Proporção de classes após o SMOTE:\n {y_train_res.value_counts()}")