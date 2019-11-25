import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import operator
import re
from random import randrange
import timeit

#----TAMANHO DO GRAFO----
treeSize = 100
probalilidade_Aresta = 0.1
#----TAMANHO DO GRAFO----

#----USANDO A LIB NETWORKX PARA GERAR, CONTABILIZAR O GRAFO E ORDENAR-----
G = nx.fast_gnp_random_graph(treeSize, probalilidade_Aresta, seed=None, directed=False)
nx.draw(G, with_labels=True, node_color='skyblue', node_size=250, width=2.0, edge_cmap=plt.cm.Blues)
edge = nx.edges(G)
degrees = dict(nx.degree(G))
sorted_degree = sorted(degrees.items(), key=operator.itemgetter(1),reverse=True)
#----USANDO A LIB NETWORKX PARA GERAR, CONTABILIZAR O GRAFO E ORDENAR-----


sorted_degree = dict(sorted_degree)
firstTry = []

#------CONVERTE A RESPOSTA DA LIB EM UMA LISTA----- "FAMOSA MARRETA NO CODIGO"
string =''
def nextTry(vector):
    string = str(edge(firstTry)).replace('[','').replace(']','').replace(' ','').replace('),',');')
    mylist = string.split(";")
    return mylist
#------CONVERTE A RESPOSTA DA LIB EM UMA LISTA----- "FAMOSA MARRETA NO CODIGO"

#----CONVERTE A SOLUÇÃO PAR DECIMAL PARA VERIFICAR SE É UMA SOLUÇÃO----
def convertDecimal(vector):
    vectorDecimal = []
    for x in range(len(vector)):
        if vector[x] == 1:
            vectorDecimal.append(x)
    return vectorDecimal
#----CONVERTE A SOLUÇÃO PAR DECIMAL PARA VERIFICAR SE É UMA SOLUÇÃO----

#----VERIFICA SE A SOLUÇÃO È VALIDA----
def isSolution(vector):
    if len(edge(vector)) == len(edge):
        return True
    else:
        return False
#----VERIFICA SE A SOLUÇÃO È VALIDA----

#----GERANDO SOLUÇÃO BASEADO NO DEGREE----
firstTry = []
for x in sorted_degree:
    if isSolution(firstTry) == False:
        firstTry.append(x)
    else:
        break
#----GERANDO SOLUÇÃO BASEADO NO DEGREE----

#----CONVERTE A SOLUÇÃO PAR BINARIO PARA EXPLORAR OS VIZINHOS----
treeBinaria = []
def converterBinario(vector):
    treeBinaria = []
    for x in range(treeSize):
        treeBinaria.append(0)
    for y in vector:
        treeBinaria[y] = 1
    return treeBinaria
#----CONVERTE A SOLUÇÃO PAR BINARIO PARA EXPLORAR OS VIZINHOS----


print('Solução inicial pelo degree: ' + str(firstTry) + ' - Qt de nodos: ' + str(len(firstTry)))

#----METODO PARA EXPLORAR OS VIZINHOS DA SULUÇÃO---
InitialSolution = firstTry
best = InitialSolution
def findNeighbors(tentativa):
    best = tentativa
    for x in range(treeSize):
        pl = converterBinario(tentativa)
        if pl[x] == 1:
            pl[x] = 0
        elif pl[x] == 0:
            pl[x] = 1
        if isSolution(convertDecimal(pl)) == True:
            if len(convertDecimal(pl)) < len(tentativa):
                best = convertDecimal(pl)
    return best
#----METODO PARA EXPLORAR OS VIZINHOS DA SULUÇÃO---


iteracoes = 5

start_time = timeit.default_timer()
#------Testando solução pelo degree------
try0 = findNeighbors(firstTry)

for i in range(iteracoes):
    besty = try0
    try0 = findNeighbors(try0)
    if len(try0) <= len(besty) and try0 != besty:
        print('Melhor Solução DEGREE entre vizinhos do melhor filho nv'+ str(i) +':'+ ' '+ str(try0) + ' - Qt de nodos: ' + str(len(try0)))
    else:
        elapsed_degree = timeit.default_timer() - start_time
        print('Sem filhos melhores')
        print('Melhor Solução DEGREE encontrada com '+ str(i + 1) +' iteracoes:'+ ' '+ str(try0) + ' - Qt de nodos: ' + str(len(try0)))
        FinalSolutionDegree = try0
        break
#------Testando solução pelo degree------

#------Gerando Solução Aleatoria--------
randomSolution = []
while isSolution(randomSolution) == False:
    rnd = randrange(treeSize)
    if rnd not in randomSolution:
        randomSolution.append(rnd)

print('------- Solução Inicial ALEATORIA: ' + str(len(randomSolution)) + '-----------------')
#------Gerando Solução Aleatoria--------

#------Testando Solução Aleatoria-------
start_time = timeit.default_timer()
try1 = findNeighbors(randomSolution)
for i in range(iteracoes):
    besty = try1
    try1 = findNeighbors(try1)
    if len(try1) <= len(besty) and try1 != besty:
        print('Melhor Solução ALEATORIA entre vizinhos do melhor filho nv'+ str(i) +':'+ ' '+ str(try1) + ' - Qt de nodos: ' + str(len(try1)))
    else:
        elapsed_aleatorio = timeit.default_timer() - start_time
        print('Sem filhos melhores')
        print('Melhor Solução ALEATORIA encontrada com '+ str(i + 1) +' iteracoes:'+ ' '+ str(try1) + ' - Qt de nodos: ' + str(len(try1)))
        FinalSolutionALEATORIA = try1
        break
#------Testando Solução Aleatoria-------
print('         -----------------------')
print('------- Solução Inicial DEGREE: ' + str(len(firstTry)))
print('------- Solução Final DEGREE: ' + str(len(FinalSolutionDegree)) +', Tempo de processamento: '+ str(round(elapsed_degree,2)))
print('         -----------------------')
print('------- Solução Inicial ALEATORIA: ' + str(len(randomSolution)))
print('------- Solução Final ALEATORIA: ' + str(len(FinalSolutionALEATORIA))+', Tempo de processamento: '+ str(round(elapsed_aleatorio,2)))
print('         -----------------------')
plt.show()
