import pandas as pd
import os


lista = os.listdir('./Valia')

consolidado = pd.DataFrame()

for file in lista:

    df = pd.read_csv('./Valia/'+file,sep=';',encoding='latin1')
    consolidado = consolidado.append(df, ignore_index=True)

print(consolidado)


consolidado.to_csv('Consolidado.csv',sep=';',encoding='latin1')