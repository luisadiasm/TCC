import pandas as pd  
import sklearn
from sklearn import svm, preprocessing, naive_bayes,neighbors, neural_network, tree
import numpy as np
from joblib import dump, load
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('dataframe.csv',sep=';') 
df['Preco_arq'] = df['Preco_arq']/10000
df = df.fillna(0)
#df.drop(df.tail(1).index,inplace=True)
df = sklearn.utils.shuffle(df)

del df['NomeArquivo']

x = df.drop('Preco_arq',axis=1).values
#y = df['Preco_arq'].values
#y = np.asarray(df['Preco_arq'], dtype="|S6")
y = list(df['Preco_arq'])
test_size = 70

x_train = x[:-test_size]
y_train = y[:-test_size]

x_test = x[-test_size:]
y_test = y[-test_size:]

print('---MLP---')
#clf = neural_network.MLPClassifier(hidden_layer_sizes=(10,10,5,5),max_iter=5000)
clf = MLPRegressor(hidden_layer_sizes=(100,200), max_iter=500)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
dump(clf, 'MLP.pkl', compress=9)

#for x,y in zip(x_test,y_test):
#    print(f"Model:{clf4.predict([x])[0]}, Actual: {y}")
