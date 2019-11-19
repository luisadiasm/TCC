import pandas as pd  
import sklearn
from sklearn import svm, preprocessing, naive_bayes,neighbors, neural_network, tree
import numpy as np
from joblib import dump, load

df = pd.read_csv('dataframe.csv', index_col=[0],sep=';') 

df = df.fillna(0)
#df.drop(df.tail(1).index,inplace=True)
df = sklearn.utils.shuffle(df)

del df['NomeArquivo']

x = df.drop('Preco_arq',axis=1).values
y = df['Preco_arq'].values

test_size = 50

x_train = x[:-test_size]
y_train = y[:-test_size]

x_test = x[-test_size:]
y_test = y[-test_size:]


print('---MLP---')
clf = neural_network.MLPClassifier(hidden_layer_sizes=(100,100,50,50))
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
dump(clf, 'MLP.pkl', compress=9)

print('---Naive Bayers---')
clf = naive_bayes.BernoulliNB()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
dump(clf, 'NVB.pkl', compress=9)

print('---Nearest Neightbors---')
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
dump(clf, 'NN.pkl', compress=9)

print('---Decision Tree---')
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
dump(clf, 'DT.pkl', compress=9)

#for x,y in zip(x_test,y_test):
#    print(f"Model:{clf4.predict([x])[0]}, Actual: {y}")
