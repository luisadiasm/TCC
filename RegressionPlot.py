print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause

import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing as pre
import pandas as pd

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
diabetes_y_train = y[:-test_size]

x_test = x[-test_size:]
diabetes_y_test = y[-test_size:]

diabetes_X_train = scaler.fit_transform(x_train)
diabetes_X_test = scaler.fit_transform(x_test)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()