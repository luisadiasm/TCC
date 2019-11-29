import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import sklearn
from sklearn import preprocessing as pre
from sklearn.neural_network import MLPRegressor
import timeit


scaler = pre.StandardScaler()
df = pd.read_csv('dataframe_v02.csv',sep=';') 
test_size = 15
output = pd.DataFrame()
dicty = {}
del df['NomeArquivo']
sizes = [4,8,12,24,32,64,128,256]
no_layers = 1

for size in sizes:
    for i in range(50):
        start_time = timeit.default_timer()

        df = sklearn.utils.shuffle(df)

        x = df.drop('Preco_arq',axis=1).values
        #y = df['Preco_arq'].values
        #y = np.asarray(df['Preco_arq'], dtype="|S6")
        y = list(df['Preco_arq'])

        x_train = x[:-test_size]
        y_train = y[:-test_size]

        x_test = x[-test_size:]
        y_test = y[-test_size:]

        X_train_scaled = scaler.fit_transform(x_train)
        X_test_scaled = scaler.fit_transform(x_test)

        regressor_MLP = MLPRegressor(hidden_layer_sizes=(size), max_iter=10000)
        regressor_MLP.fit(x_train, y_train)
        y_pred_mlp = regressor_MLP.predict(x_test)

        

        Mean_abs_error_mlp = round(metrics.mean_absolute_error(y_test, y_pred_mlp),3)

        print('Mean Absolute Error:', Mean_abs_error_mlp)  

        elapsed = timeit.default_timer() - start_time
        dicty.update({'Absolute Error MLP':Mean_abs_error_mlp,'Hidden layer Size':size,'No Hidden Layers':no_layers,'Execution Time':round(elapsed,2)})

        output = output.append(dicty, ignore_index=True)

#output.to_csv('output.csv',sep=';',decimal=',')

holder = {}
holder2 = {}
for size in sizes:
        dfsize = output.loc[(output['Hidden layer Size']==size)]
        dfsize = dfsize[['Absolute Error MLP','Execution Time']]
        holder.update({str(size):dfsize['Absolute Error MLP'].mean()})
        holder2.update({str(size):dfsize['Execution Time'].mean()})

plt.bar(*zip(*holder.items()))
plt.title('Erro medio pelo tamanho da camanha oculta ('+str(no_layers)+' camadas)')  
plt.show()

plt.bar(*zip(*holder2.items()))
plt.title('Tempo medio pelo tamanho da camanha oculta ('+str(no_layers)+' camadas)')  
plt.show()
