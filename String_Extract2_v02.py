from tika import parser
import re
from string import digits
from unicodedata import normalize
from collections import Counter
import pickle
from PyPDF2 import PdfFileReader
import os
import pandas as pd
import textstat
import random
import nltk
#nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def remover_acentos(txt):
    return normalize('NFKD', txt.strip()).encode('ASCII', 'ignore').decode('ASCII')
remove_digits = str.maketrans('', '', digits)

path  = "./diretorioING/"
dirs = os.listdir(path)
df = pd.DataFrame()

for file in dirs:

    #--- DICIONÁRIO COM AS FEATURES ---
    data_dict = {'NomeArquivo':'0','verbos':'0', 'substantivos':'0', 'nro_pags':'0', 'nro_caracteres':'0', 'med_caracs_pags':'0', \
    'nro_palavras':'0', 'plvrs_unicas':'0', 'plvr_outra_ling':'0', 'numbers':'0', 'Preco_arq':'0'}


    raw = parser.from_file(path + file)

    filename, file_extension = os.path.splitext(path + file)  
    print(path+file)
    pdf = PdfFileReader(open(path + file,'rb'))

    string = re.sub(' +', ' ',re.sub(r'[^\w\s]','',remover_acentos(str(raw['content'])).translate(remove_digits)))
    #string = re.sub(' +', ' ',re.sub(r'[^\w\s]','', str(path+file)))
    string = string.replace('\r', '').replace('\n', '')
    new_string = ' '.join([w for w in string.split() if len(w)<20])

    tokenized_word=word_tokenize(new_string)
    fdist = FreqDist(tokenized_word)

    #-----CRIANDO FEATURE COM O NOME DO ARQUIVO ANALISADO------
    data_dict.update({'NomeArquivo':'' + os.path.basename(file)})

    #PRINT NOME DO ARQUIVO
    print('Arquivo analisado: ' + os.path.basename(file))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE PÁGINAS -----
    data_dict.update({'nro_pags':'' + str(pdf.getNumPages())})

    #PRINT TOTAL DE PAGINAS
    print('TOTAL PAGINAS - ' + str(pdf.getNumPages()))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- QUANTIDADE DE CARACTERES -----
    data_dict.update({'nro_caracteres':'' + str(len(new_string))})
    
    #PRINT TOTAL CHARACTERS
    print('TOTAL CARATERES - ' +str(len(new_string)))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- MÉDIA DE CARACTERES POR PÁGINA -----
    data_dict.update({'med_caracs_pags':'' + str(round(len(new_string)/pdf.getNumPages(),0))})

    #PRINT MEDIA CARACTER/PAGINA
    print('MEDIA CARACTERES/PAGINAS - ' +str(round(len(new_string)/pdf.getNumPages(),0)))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- QUANTIDADE DE PALAVRAS -----
    data_dict.update({'nro_palavras':'' + str(len(tokenized_word))})

    #PRINT TOTAL PALAVRAS
    print('TOTAL PALAVRAS - ' + str(len(tokenized_word)))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE PALAVRAS ÚNICAS -----
    data_dict.update({'plvrs_unicas':'' + str(len(fdist.most_common()))})
    
    #TOTAL PALAVRAS UNICAS
    print('TOTAL DE PALAVRAS UNICAS - ' + str(len(fdist.most_common())))

    #--------------------------------CLASSIFICAR PALAVRAS--------------------------------------
    classificado = nltk.pos_tag(tokenized_word)

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE VERBOS -----
    countvb = 0
    for word, tag in classificado: 
        if tag.startswith('VB'): #'VB' significa os verbos
            countvb = countvb + 1
    
    data_dict.update({'verbos':'' + str(countvb)})
    
    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE VERBOS - ' + str(countvb))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE SUBSTANTIVOS -----
    countnn = 0
    for word, tag in classificado: 
        if tag.startswith('NN'): #'NN' significa os noun (substantivo em inglês)
            countnn = countnn + 1
    
    data_dict.update({'substantivos':'' + str(countnn)})
    
    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE SUBSTANTIVOS - ' + str(countnn))
    
    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE NÚMEROS -----
    countcd = 0
    for word, tag in classificado: 
        if tag.startswith('CD'): #'CD' significa Cardinal Digit
            countcd = countcd + 1
    
    data_dict.update({'numbers':'' + str(countcd)})
    
    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE NÚMEROS - ' + str(countcd))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE PALAVRAS DE OUTRO IDIOMA -----
    countfw = 0
    for word, tag in classificado: 
        if tag.startswith('FW'): #'FW' significa Foreign Word
            countfw = countfw + 1
    
    data_dict.update({'plvr_outra_ling':'' + str(countfw)})

    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE PALAVRAS DE OUTRO IDIOMA - ' + str(countfw))

    #----CALCULANDO INDICE FLESCH----
    indice = textstat.flesch_reading_ease(raw['content'])    

    print("---- ANÁLISE FLESCH ----")
    preco_arq = 0.00
    if indice >= 0.00 and indice <= 29.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (VERY CONFUSING)")
        preco_arq = round((100-indice) * 50,2)
        data_dict.update({'Preco_arq': '' + str(preco_arq)})
        print("PREÇO SIMULADO = R$" + str(preco_arq))
    elif indice >= 30.00 and indice <= 49.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (DIFFICULT)")
        preco_arq = round((100-indice) * 50,2)
        data_dict.update({'Preco_arq': '' + str(preco_arq)})
        print("PREÇO SIMULADO = R$" + str(preco_arq))
    elif indice >= 50.00 and indice <= 59.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (FAIRLY DIFFICULT)")
        preco_arq = round((100-indice) * 50,2)
        data_dict.update({'Preco_arq': '' + str(preco_arq)})
        print("PREÇO SIMULADO = R$" + str(preco_arq))
    elif indice >= 60.00 and indice <= 69.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (STANDARD)")
        preco_arq = round((100-indice) * 50,2)
        data_dict.update({'Preco_arq': '' + str(preco_arq)})
        print("PREÇO SIMULADO = R$" + str(preco_arq))
    elif indice >= 70.00 and indice <= 79.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (FAIRLY EASY)")
        preco_arq = round((100-indice) * 50,2)
        data_dict.update({'Preco_arq': '' + str(preco_arq)})
        print("PREÇO SIMULADO = R$" + str(preco_arq))
    elif indice >= 80.00 and indice <= 89.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (EASY)")
        preco_arq = round((100-indice) * 50,2)
        data_dict.update({'Preco_arq': '' + str(preco_arq)})
        print("PREÇO SIMULADO = R$" + str(preco_arq))
    else:
        print("ÍNDICE FLESCH: " + str(indice) + "  (VERY EASY)")
        preco_arq = round((100-indice) * 50,2)
        data_dict.update({'Preco_arq': '' + str(preco_arq)})
        print("PREÇO SIMULADO = R$" + str(preco_arq))

    #POPULANDO O DATAFRAME COM AS INFORMAÇÕES DO DICIONÁRIO
    df = df.append(data_dict, ignore_index=True)

    #----CONTAGEM TIPO DE PALAVRAS (VERIFICAR ARQUIVO LEGENDA.TXT)----
    print(Counter(tag for word,tag in classificado)) 

    print('\n') 
    

#---- CRIAÇÃO DO DATAFRAME COM OS DADOS DO DICIONÁRIO
df.to_csv('dataframe_v02.csv', sep=';',encoding='cp1252')
print('--------- Dataframe criado! ---------')

#                                   RESUMO A FAZER
# # # # #   1-Usar os.listdir para criar uma lista de todos os arquivos dentro de uma pasta especifica *OK*
# # # # #   2-Aplicar um loop nas funções acima para todos os arquivos dessa pasta *OK*
# # # # #   3-criar um dicionario com as features acima para cada arquivo *OK*
# # # # #   ("Para facilitar crie um grande 'else if' classificando os arquivos por lotes EX: PALAVRAS 0-100,PALAVRAS 101 - 200") *OK*
# # # # #   ("Aplicar a planilha de preco de cada arquivo ao final, 
# # # # #   ou seja vai existir uma coluca 'PRECO' para cada arquivo
# # # # #   que vai ser uma referencia a essa planilha ")
# # # # #   4-ao final de cada dicionario dar append em um DataFrame *OK*
# # # # #   5-Campos com o mesmo index no dicionario vão ser empilhados em colunas dentro do dataframe, indexs diferentes vai
# # # # #   ser criado uma nova coluna, ou seja para cada index unico vai existir uma coluna no DataFrame final *OK*
# # # # #   6-Aplicar algoritmos de machine learning ("verificar com Gustavo ou Ramon qual vai ser a melhor maneira.")

### https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html