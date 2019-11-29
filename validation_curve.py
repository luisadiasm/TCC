import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
import pandas as pd  
import sklearn
from sklearn import svm, preprocessing, naive_bayes,neighbors, neural_network, tree
from joblib import dump, load
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing as pre
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


scaler = pre.StandardScaler()

df = pd.read_csv('dataframe_v02.csv',sep=';') 
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

X = scaler.fit_transform(x)
#Y = scaler.fit_transform(y)

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(MLPRegressor(hidden_layer_sizes=(50,50,50,50), max_iter=1000), X, y, "alpha",np.logspace(-7, 3, 3),cv=5,n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()