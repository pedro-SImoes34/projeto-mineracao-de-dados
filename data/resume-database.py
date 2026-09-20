import pandas as pd

df_amostra = pd.read_csv('data/creditcard.csv', nrows = 1000)
df_amostra.to_csv('data/creditcard_amostra.csv', index = False)