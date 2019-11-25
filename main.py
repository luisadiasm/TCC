import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":

   # carrega o data-set
   data = pd.read_csv("./dataframe.csv", header=None, index_col=False, sep=';')

   X = np.array(data)

   XX = X[:,0:89]

   YY = X[:,50]

   # define (aletoriamente) o conjunto de treinamento (70%) e teste (30%)
   X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, stratify=YY, test_size=0.33, random_state=42)

   '''
   Configuracao do classificador MLP: alterar os parametros:
      - Documentacao:
         https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
      - hidden_layer_sizes = [a,b,...z]: 
         tamanho do vetor = qtde de camadas
         inteiro em cada posicao = qtde de neuronios por camada     
      - max_iter: 
         numero maximo de iteracoes da rede   
   '''
   mlp = MLPClassifier(activation='logistic', solver='lbfgs', hidden_layer_sizes=[10, 10], max_iter=500, random_state=0)

   mlp.fit(X_train, Y_train)

   # apresenta a acuracia da rede para treinamento e teste
   print("Acuracia sobre o conjunto de treinamento: {:.2f}".format(mlp.score(X_train, Y_train)))
   print("Acuracia sobre o conjunto de teste: {:.2f}".format(mlp.score(X_test, Y_test)))
