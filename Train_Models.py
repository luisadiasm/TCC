import pandas as pd  
import sklearn
from sklearn import svm, preprocessing, naive_bayes,neighbors, neural_network, tree
import numpy as np
from joblib import dump, load
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing as pre
scaler = pre.StandardScaler()

df = pd.read_csv('dataframe_v02.csv',sep=';',decimal=',') 
#df['Preco_arq'] = df['Preco_arq']/10000
df = df.fillna(0)
#df.drop(df.tail(1).index,inplace=True)
df = sklearn.utils.shuffle(df)

del df['NomeArquivo']

x = df.drop('Preco_arq',axis=1).values
#y = df['Preco_arq'].values
#y = np.asarray(df['Preco_arq'], dtype="|S6")
y = list(df['Preco_arq'])
test_size = 20

x_train = x[:-test_size]
y_train = y[:-test_size]

x_test = x[-test_size:]
y_test = y[-test_size:]

X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.fit_transform(x_test)

print('---MLP---')
clf = MLPRegressor(hidden_layer_sizes=(100,100,100,100), max_iter=10000)
clf.fit(X_train_scaled,y_train)

print(y_test)
print(clf.predict(X_test_scaled))

predicted = clf.predict(X_test_scaled)

dicty = {}
output = pd.DataFrame()
for i in range(len(predicted)):
    dicty.update({'PREDICTED':round(predicted[i],2),'REAL':y_test[i]})
    output = output.append(dicty, ignore_index=True)

output.to_csv('Output.csv',sep=';',decimal=',')