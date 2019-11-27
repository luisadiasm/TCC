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

    #-----POPULANDO O DICIONÁRIO COM NÚMERO DE PÁGINAS DE CADA ARQUIVO-----
    data_dict.update({'nro_pags':'' + str(pdf.getNumPages())})

    #PRINT TOTAL DE PAGINAS
    print('TOTAL PAGINAS - ' + str(pdf.getNumPages()))

    #-----POPULANDO O DICIONÁRIO COM QUANTIDADE DE CARACTERES DE CADA ARQUIVO-----
    data_dict.update({'nro_caracteres':'' + str(len(new_string))})
    
    #PRINT TOTAL CHARACTERS
    print('TOTAL CARATERES - ' +str(len(new_string)))

    #-----POPULANDO O DICIONÁRIO COM MÉDIA DE CARACTERES POR PÁGINA DE CADA ARQUIVO-----
    data_dict.update({'med_caracs_pags':'' + str(round(len(new_string)/pdf.getNumPages(),0))})

    #PRINT MEDIA CARACTER/PAGINA
    print('MEDIA CARACTERES/PAGINAS - ' +str(round(len(new_string)/pdf.getNumPages(),0)))

    #-----POPULANDO O DICIONÁRIO COM QUANTIDADE DE PALAVRAS DE CADA ARQUIVO-----
    data_dict.update({'nro_palavras':'' + str(len(tokenized_word))})

    #PRINT TOTAL PALAVRAS
    print('TOTAL PALAVRAS - ' + str(len(tokenized_word)))

    #-----POPULANDO O DICIONÁRIO COM NÚMERO DE PALAVRAS ÚNICAS DE CADA ARQUIVO-----
    data_dict.update({'plvrs_unicas':'' + str(len(fdist.most_common()))})
    
    #TOTAL PALAVRAS UNICAS
    print('TOTAL DE PALAVRAS UNICAS - ' + str(len(fdist.most_common())))

    #--------------------------------CLASSIFICAR PALAVRAS--------------------------------------
    classificado = nltk.pos_tag(tokenized_word)

    #-----POPULANDO O DICIONÁRIO COM NÚMERO DE VERBOS DE CADA ARQUIVO-----
    countvb = 0
    for word, tag in classificado: 
        if tag.startswith('VB'): #'VB' significa os verbos
            countvb = countvb + 1
    
    data_dict.update({'verbos':'' + str(countvb)})
    
    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE VERBOS - ' + str(countvb))

    #-----POPULANDO O DICIONÁRIO COM NÚMERO DE SUBSTANTIVOS DE CADA ARQUIVO-----
    countnn = 0
    for word, tag in classificado: 
        if tag.startswith('NN'): #'NN' significa os noun (substantivo em inglês)
            countnn = countnn + 1
    
    data_dict.update({'substantivos':'' + str(countnn)})
    
    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE SUBSTANTIVOS - ' + str(countnn))
    
    #-----POPULANDO O DICIONÁRIO COM NÚMERO DE NÚMEROS DE CADA ARQUIVO-----
    countcd = 0
    for word, tag in classificado: 
        if tag.startswith('CD'): #'CD' significa Cardinal Digit
            countcd = countcd + 1
    
    data_dict.update({'numbers':'' + str(countcd)})
    
    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE NÚMEROS - ' + str(countcd))

    #-----POPULANDO O DICIONÁRIO COM NÚMERO DE PALAVRAS DE OUTRO IDIOMA DE CADA ARQUIVO-----
    countfw = 0
    for word, tag in classificado: 
        if tag.startswith('FW'): #'FW' significa Foreign Word
            countfw = countfw + 1
    
    data_dict.update({'plvr_outra_ling':'' + str(countfw)})

    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE PALAVRAS DE OUTRO IDIOMA - ' + str(countfw))

    #----CALCULANDO INDICE FLESCH----
    indice = textstat.flesch_reading_ease(raw['content'])   
    mult_indice = 0.00

    print("---- ANÁLISE FLESCH ----")
    preco_arq = 0.00
    if indice >= 0.00 and indice <= 29.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (VERY CONFUSING)")
        mult_indice = 1.6
    elif indice >= 30.00 and indice <= 49.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (DIFFICULT)")
        mult_indice = 1.4
    elif indice >= 50.00 and indice <= 59.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (FAIRLY DIFFICULT)")
        mult_indice = 1.2
    elif indice >= 60.00 and indice <= 69.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (STANDARD)")
        mult_indice = 1.0
    elif indice >= 70.00 and indice <= 79.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (FAIRLY EASY)")
        mult_indice = 0.8
    elif indice >= 80.00 and indice <= 89.99:
        print("ÍNDICE FLESCH: " + str(indice) + "  (EASY)")
        mult_indice = 0.6
    else:
        print("ÍNDICE FLESCH: " + str(indice) + "  (VERY EASY)")
        mult_indice = 0.4

    #----CONTAGEM TIPO DE PALAVRAS (VERIFICAR ARQUIVO LEGENDA.TXT)----
    print(Counter(tag for word,tag in classificado)) 

    #----DEFININDO PREÇOS DOS ARQUIVOS----
    preco_arq = round((((len(new_string))*0.005) + ((pdf.getNumPages())*5) + ((len(tokenized_word))*0.002) + (countcd*0.00005) + (countfw*0.002) + ((len(fdist.most_common()))*0.0003) + (countnn*0.007) + (countvb*0.002)) * mult_indice,2)
    data_dict.update({'Preco_arq':'' + str(preco_arq)})
    print("PREÇO CALCULADO: R$" + str(preco_arq))

    #POPULANDO O DATAFRAME COM AS INFORMAÇÕES DO DICIONÁRIO
    df = df.append(data_dict, ignore_index=True)

    print('\n') 

#---- CRIAÇÃO DO DATAFRAME COM OS DADOS DO DICIONÁRIO
df.to_csv('dataframe_v02.csv', sep=';',encoding='cp1252')
print('--------- Dataframe criado! ---------')
