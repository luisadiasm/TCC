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

scaler = pre.StandardScaler()
df = pd.read_csv('dataframe_v02.csv',sep=';') 

output = pd.DataFrame()
dicty = {}
del df['NomeArquivo']

for i in range(25):
    df = sklearn.utils.shuffle(df)

    x = df.drop('Preco_arq',axis=1).values
    #y = df['Preco_arq'].values
    #y = np.asarray(df['Preco_arq'], dtype="|S6")
    y = list(df['Preco_arq'])
    test_size = 15

    x_train = x[:-test_size]
    y_train = y[:-test_size]

    x_test = x[-test_size:]
    y_test = y[-test_size:]

    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.fit_transform(x_test)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_pred_slr = regressor.predict(x_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_slr))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_slr)) 
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr)))


    Mean_abs_error_slr = round(metrics.mean_absolute_error(y_test, y_pred_slr),3)
    Mean_sqr_error_slr = round(metrics.mean_squared_error(y_test, y_pred_slr),3)
    Mean_root_sqr_error_slr = round(metrics.mean_squared_error(y_test, y_pred_slr),3)

    dicty.update({'Absolute Error SLR':Mean_abs_error_slr,'Squared Error SLR':Mean_sqr_error_slr,'Root Mean Squared Error SLR':Mean_root_sqr_error_slr})


    regressor_MLP = MLPRegressor(hidden_layer_sizes=(100,100,100,100), max_iter=10000)
    regressor_MLP.fit(x_train, y_train)
    y_pred_mlp = regressor_MLP.predict(x_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_mlp))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_mlp)) 
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlp)))


    Mean_abs_error_mlp = round(metrics.mean_absolute_error(y_test, y_pred_mlp),3)
    Mean_sqr_error_mlp = round(metrics.mean_squared_error(y_test, y_pred_mlp),3)
    Mean_root_sqr_error_mlp = round(metrics.mean_squared_error(y_test, y_pred_mlp),3)

    dicty.update({'Absolute Error MLP':Mean_abs_error_mlp,'Squared Error MLP':Mean_sqr_error_mlp,'Root Mean Squared Error MLP':Mean_root_sqr_error_mlp})

    output = output.append(dicty, ignore_index=True)



output.to_csv('output.csv',sep=';',decimal=',')


#output = output.sort_values(['Absolute Error SLR'], ascending=[False])
df3 = output.head(20)

dfAbsolute = df3[['Absolute Error MLP','Absolute Error SLR']].copy()
dfSquared = df3[['Root Mean Squared Error MLP','Root Mean Squared Error SLR']].copy()

dfAbsolute.rename(columns={'Absolute Error MLP': 'MultiLayer Perceptron', 'Absolute Error SLR': 'Simple Linear Regression'}, inplace=True)
dfSquared.rename(columns={'Root Mean Squared Error MLP': 'MultiLayer Perceptron', 'Root Mean Squared Error SLR': 'Simple Linear Regression'}, inplace=True)

dfAbsolute.plot(kind='bar',figsize=(10,6))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Mean Absolute Error')  
plt.show()

dfSquared.plot(kind='bar',figsize=(10,6))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Root Mean Squared Error')  
plt.show()
