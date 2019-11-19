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

def remover_acentos(txt):
    return normalize('NFKD', txt.strip()).encode('ASCII', 'ignore').decode('ASCII')
remove_digits = str.maketrans('', '', digits)

path  = "./diretorio/"
dirs = os.listdir(path)
df = pd.DataFrame()

for file in dirs:

    #--- DICIONÁRIO COM AS FEATURES ---
    data_dict = {'NomeArquivo':'0','verbos0a100':'0', 'verbos101a200':'0', 'verbos201a300':'0', 'verbos301a400':'0', 'verbos401a500':'0', \
    'verbos501a600':'0', 'verbos601a750':'0', 'verbos751a1000':'0', 'verbos1001up':'0', 'subs0a100':'0', 'subs101a200':'0', \
    'subs201a300':'0', 'subs301a400':'0', 'subs401a500':'0', 'subs501a600':'0', 'subs601a750':'0', 'subs751a1000':'0', 'subs1001up':'0', \
    'nro_pags1a10':'0', 'nro_pags11a20':'0', 'nro_pags21a30':'0', 'nro_pags31a40':'0', 'nro_pags41a50':'0', 'nro_pags51a60':'0',
    'nro_pags61a70':'0', 'nro_pags71a80':'0', 'nro_pags81a90':'0', 'nro_pags91a100':'0', 'nro_pags101up':'0', \
    'nro_caracs1a500':'0', 'nro_caracs501a1000':'0', 'nro_caracs1001a1500':'0', 'nro_caracs1501a2000':'0', \
    'nro_caracs2001a2500':'0', 'nro_caracs2501a3000':'0', 'nro_caracs3501a4000':'0', 'nro_caracs4001a5000':'0', \
    'nro_caracs5001a6000':'0', 'nro_caracs6001a6500':'0', 'nro_caracs6501a7000':'0', 'nro_caracs7501a8000':'0', \
    'nro_caracs8001a10000':'0', 'nro_caracs10001a15000':'0', 'nro_caracs15001a20000':'0', 'nro_caracs20001a30000':'0', \
    'nro_caracs30001a40000':'0', 'nro_caracs40001up':'0', 'med_caracs_pags1a500':'0', 'med_caracs_pags501a1000':'0', \
    'med_caracs_pags1001a1500':'0', 'med_caracs_pags1501a2000':'0', 'med_caracs_pags2001a2500':'0', \
    'med_caracs_pags2501up':'0', 'nro_palavras1a500':'0', 'nro_palavras501a1000':'0', 'nro_palavras1001a1500':'0', \
    'nro_palavras1501a2000':'0', 'nro_palavras2001a2500':'0', 'nro_palavras2501a3000':'0', 'nro_palavras3501a4000':'0', \
    'nro_palavras4001a5000':'0', 'nro_palavras5001a6000':'0', 'nro_palavras6001a6500':'0', 'nro_palavras6501a7000':'0', \
    'nro_palavras7501a8000':'0', 'nro_palavras8001a10000':'0', 'nro_palavras10001a15000':'0', 'nro_palavras15001a20000':'0', \
    'nro_palavras20001up':'0', 'plvrs_unicas0a10':'0', 'plvrs_unicas11a20':'0', 'plvrs_unicas21a30':'0', 'plvrs_unicas31a40':'0', \
    'plvrs_unicas41a50':'0', 'plvrs_unicas51up':'0', 'plvr_outra_ling1a10':'0', 'plvr_outra_ling11a20':'0', \
    'plvr_outra_ling21a30':'0', 'plvr_outra_ling31a40':'0', 'plvr_outra_ling41a50':'0', 'plvr_outra_ling51up':'0', \
    'numbers1a10':'0', 'numbers11a20':'0', 'numbers21a30':'0', 'numbers31a40':'0', 'numbers41a50':'0', 'numbers51up':'0', 'Preco_arq':'0'}


    raw = parser.from_file(path + file)

    filename, file_extension = os.path.splitext(path + file)  
    pdf = PdfFileReader(open(path + file,'rb'))

    string = re.sub(' +', ' ',re.sub(r'[^\w\s]','',remover_acentos(str(raw['content'])).translate(remove_digits)))
    #string = re.sub(' +', ' ',re.sub(r'[^\w\s]','', str(path+file)))
    string = string.replace('\r', '').replace('\n', '')
    new_string = ' '.join([w for w in string.split() if len(w)<20])

    import nltk
    #nltk.download('all')
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    tokenized_word=word_tokenize(new_string)
    fdist = FreqDist(tokenized_word)

    #-----CRIANDO FEATURE COM O NOME DO ARQUIVO ANALISADO------
    data_dict.update({'NomeArquivo':'' + os.path.basename(file)})

    #PRINT NOME DO ARQUIVO
    print('Arquivo analisado: ' + os.path.basename(file))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE PÁGINAS -----
    if pdf.getNumPages() >= 0 and pdf.getNumPages() <= 10:
        data_dict.update({'nro_pags1a10':'1'})
    elif pdf.getNumPages() >= 11 and pdf.getNumPages() <= 20:
        data_dict.update({'nro_pags11a20':'1'})
    elif pdf.getNumPages() >= 21 and pdf.getNumPages() <= 30:
        data_dict.update({'nro_pags21a30':'1'})
    elif pdf.getNumPages() >= 31 and pdf.getNumPages() <= 40:
        data_dict.update({'nro_pags31a40':'1'})
    elif pdf.getNumPages() >= 41 and pdf.getNumPages() <= 50:
        data_dict.update({'nro_pags41a50':'1'})
    elif pdf.getNumPages() >= 51 and pdf.getNumPages() <= 60:
        data_dict.update({'nro_pags51a60':'1'})
    elif pdf.getNumPages() >= 61 and pdf.getNumPages() <= 70:
        data_dict.update({'nro_pags61a70':'1'})
    elif pdf.getNumPages() >= 71 and pdf.getNumPages() <= 80:
        data_dict.update({'nro_pags71a80':'1'})
    elif pdf.getNumPages() >= 81 and pdf.getNumPages() <= 90:
        data_dict.update({'nro_pags81a90':'1'})
    elif pdf.getNumPages() >= 91 and pdf.getNumPages() <= 100:
        data_dict.update({'nro_pags91a100':'1'})
    elif pdf.getNumPages() >= 101:
        data_dict.update({'nro_pags101up':'1'})

    #PRINT TOTAL DE PAGINAS
    print('TOTAL PAGINAS - ' + str(pdf.getNumPages()))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- QUANTIDADE DE CARACTERES -----
    if len(new_string) >= 0 and len(new_string) <= 500:
        data_dict.update({'nro_caracs1a500':'1'})
    elif len(new_string) >= 501 and len(new_string) <= 1000:
        data_dict.update({'nro_caracs501a1000':'1'})
    elif len(new_string) >= 1001 and len(new_string) <= 1500:
        data_dict.update({'nro_caracs1001a1500':'1'})
    elif len(new_string) >= 1501 and len(new_string) <= 2000:
        data_dict.update({'nro_caracs1501a2000':'1'})
    elif len(new_string) >= 2001 and len(new_string) <= 2500:
        data_dict.update({'nro_caracs2001a2500':'1'})
    elif len(new_string) >= 2501 and len(new_string) <= 3000:
        data_dict.update({'nro_caracs2501a3000':'1'})
    elif len(new_string) >= 3501 and len(new_string) <= 4000:
        data_dict.update({'nro_caracs3501a4000':'1'})
    elif len(new_string) >= 4001 and len(new_string) <= 5000:
        data_dict.update({'nro_caracs4001a5000':'1'})
    elif len(new_string) >= 5001 and len(new_string) <= 6000:
        data_dict.update({'nro_caracs5001a6000':'1'})
    elif len(new_string) >= 6001 and len(new_string) <= 6500:
        data_dict.update({'nro_caracs6001a6500':'1'})
    elif len(new_string) >= 6501 and len(new_string) <= 7000:
        data_dict.update({'nro_caracs6501a7000':'1'})
    elif len(new_string) >= 7501 and len(new_string) <= 8000:
        data_dict.update({'nro_caracs7501a8000':'1'})
    elif len(new_string) >= 8001 and len(new_string) <= 10000:
        data_dict.update({'nro_caracs8001a10000':'1'})
    elif len(new_string) >= 10001 and len(new_string) <= 15000:
        data_dict.update({'nro_caracs10001a15000':'1'})
    elif len(new_string) >= 15001 and len(new_string) <= 20000:
        data_dict.update({'nro_caracs15001a20000':'1'})
    elif len(new_string) >= 20001 and len(new_string) <= 30000:
        data_dict.update({'nro_caracs20001a30000':'1'})
    elif len(new_string) >= 30001 and len(new_string) <= 40000:
        data_dict.update({'nro_caracs30001a40000':'1'})
    elif len(new_string) >= 40001:
        data_dict.update({'nro_caracs40001up':'1'})
    
    #PRINT TOTAL CHARACTERS
    print('TOTAL CARATERES - ' +str(len(new_string)))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- MÉDIA DE CARACTERES POR PÁGINA -----
    if round(len(new_string)/pdf.getNumPages(),0) >= 0 and round(len(new_string)/pdf.getNumPages(),0) <= 500:
        data_dict.update({'med_caracs_pags1a500':'1'})
    elif round(len(new_string)/pdf.getNumPages(),0) >= 501 and round(len(new_string)/pdf.getNumPages(),0) <= 1000:
        data_dict.update({'med_caracs_pags501a1000':'1'})
    elif round(len(new_string)/pdf.getNumPages(),0) >= 1001 and round(len(new_string)/pdf.getNumPages(),0) <= 1500:
        data_dict.update({'med_caracs_pags1001a1500':'1'})
    elif round(len(new_string)/pdf.getNumPages(),0) >= 1501 and round(len(new_string)/pdf.getNumPages(),0) <= 2000:
        data_dict.update({'med_caracs_pags1501a2000':'1'})
    elif round(len(new_string)/pdf.getNumPages(),0) >= 2001 and round(len(new_string)/pdf.getNumPages(),0) <= 2500:
        data_dict.update({'med_caracs_pags2001a2500':'1'})
    elif round(len(new_string)/pdf.getNumPages(),0) >= 2501:
        data_dict.update({'med_caracs_pags2501up':'1'})

    #PRINT MEDIA CARACTER/PAGINA
    print('MEDIA CARACTERES/PAGINAS - ' +str(round(len(new_string)/pdf.getNumPages(),0)))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- QUANTIDADE DE PALAVRAS -----
    if len(tokenized_word) >= 0 and len(tokenized_word) <= 500:
        data_dict.update({'nro_palavras1a500':'1'})
    elif len(tokenized_word) >= 501 and len(tokenized_word) <= 1000:
        data_dict.update({'nro_palavras501a1000':'1'})
    elif len(tokenized_word) >= 1001 and len(tokenized_word) <= 1500:
        data_dict.update({'nro_palavras1001a1500':'1'})
    elif len(tokenized_word) >= 1501 and len(tokenized_word) <= 2000:
        data_dict.update({'nro_palavras1501a2000':'1'})
    elif len(tokenized_word) >= 2001 and len(tokenized_word) <= 2500:
        data_dict.update({'nro_palavras2001a2500':'1'})
    elif len(tokenized_word) >= 2501 and len(tokenized_word) <= 3000:
        data_dict.update({'nro_palavras2501a3000':'1'})
    elif len(tokenized_word) >= 3501 and len(tokenized_word) <= 4000:
        data_dict.update({'nro_palavras3501a4000':'1'})
    elif len(tokenized_word) >= 4001 and len(tokenized_word) <= 5000:
        data_dict.update({'nro_palavras4001a5000':'1'})
    elif len(tokenized_word) >= 5001 and len(tokenized_word) <= 6000:
        data_dict.update({'nro_palavras5001a6000':'1'})
    elif len(tokenized_word) >= 6001 and len(tokenized_word) <= 6500:
        data_dict.update({'nro_palavras6001a6500':'1'})
    elif len(tokenized_word) >= 6501 and len(tokenized_word) <= 7000:
        data_dict.update({'nro_palavras6501a7000':'1'})
    elif len(tokenized_word) >= 7501 and len(tokenized_word) <= 8000:
        data_dict.update({'nro_palavras7501a8000':'1'})
    elif len(tokenized_word) >= 8001 and len(tokenized_word) <= 10000:
        data_dict.update({'nro_palavras8001a10000':'1'})
    elif len(tokenized_word) >= 10001 and len(tokenized_word) <= 15000:
        data_dict.update({'nro_palavras10001a15000':'1'})
    elif len(tokenized_word) >= 15001 and len(tokenized_word) <= 20000:
        data_dict.update({'nro_palavras15001a20000':'1'})
    elif len(tokenized_word) >= 20001:
        data_dict.update({'nro_palavras20001up':'1'})

    #PRINT TOTAL PALAVRAS
    print('TOTAL PALAVRAS - ' + str(len(tokenized_word)))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE PALAVRAS ÚNICAS -----
    if len(fdist.most_common()) >= 0 and len(fdist.most_common()) <= 10:
        data_dict.update({'plvrs_unicas0a10':'1'})
    elif len(fdist.most_common()) >= 11 and len(fdist.most_common()) <= 20:
        data_dict.update({'plvrs_unicas11a20':'1'})
    elif len(fdist.most_common()) >= 21 and len(fdist.most_common()) <= 30:
        data_dict.update({'plvrs_unicas21a30':'1'})
    elif len(fdist.most_common()) >= 31 and len(fdist.most_common()) <= 40:
        data_dict.update({'plvrs_unicas31a40':'1'})
    elif len(fdist.most_common()) >= 41 and len(fdist.most_common()) <= 50:
        data_dict.update({'plvrs_unicas41a50':'1'})
    elif len(fdist.most_common()) >= 51:
        data_dict.update({'plvrs_unicas51up':'1'})
    
    #TOTAL PALAVRAS UNICAS
    print('TOTAL DE PALAVRAS UNICAS - ' + str(len(fdist.most_common())))

    #--------------------------------CLASSIFICAR PALAVRAS--------------------------------------
    classificado = nltk.pos_tag(tokenized_word)

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE VERBOS -----
    countvb = 0
    for word, tag in classificado: 
        if tag.startswith('VB'): #'VB' significa os verbos
            countvb = countvb + 1
    
    if countvb >= 0 and countvb <= 100:
        data_dict.update({'verbos0a100':'1'})
    elif countvb >= 101 and countvb <= 200:
        data_dict.update({'verbos101a200':'1'})
    elif countvb >= 201 and countvb <= 300:
        data_dict.update({'verbos201a300':'1'})
    elif countvb >= 301 and countvb <= 400:
        data_dict.update({'verbos301a400':'1'})
    elif countvb >= 401 and countvb <= 500:
        data_dict.update({'verbos401a500':'1'})
    elif countvb >= 501 and countvb <= 600:
        data_dict.update({'verbos501a600':'1'})
    elif countvb >= 601 and countvb <= 750:
        data_dict.update({'verbos601a750':'1'})
    elif countvb >= 751 and countvb <= 100:
        data_dict.update({'verbos751a1000':'1'})
    elif countvb >= 1001:
        data_dict.update({'verbos1001up':'1'})

    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE VERBOS - ' + str(countvb))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE SUBSTANTIVOS -----
    countnn = 0
    for word, tag in classificado: 
        if tag.startswith('NN'): #'NN' significa os noun (substantivo em inglês)
            countnn = countnn + 1
    
    if countnn >= 0 and countnn <= 100:
        data_dict.update({'subs0a100':'1'})
    elif countnn >= 101 and countnn <= 200:
        data_dict.update({'subs101a200':'1'})
    elif countnn >= 201 and countnn <= 300:
        data_dict.update({'subs201a300':'1'})
    elif countnn >= 301 and countnn <= 400:
        data_dict.update({'subs301a400':'1'})
    elif countnn >= 401 and countnn <= 500:
        data_dict.update({'subs401a500':'1'})
    elif countnn >= 501 and countnn <= 600:
        data_dict.update({'subs501a600':'1'})
    elif countnn >= 601 and countnn <= 750:
        data_dict.update({'subs601a750':'1'})
    elif countnn >= 751 and countnn <= 100:
        data_dict.update({'subs751a1000':'1'})
    elif countnn >= 1001:
        data_dict.update({'subs1001up':'1'})

    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE SUBSTANTIVOS - ' + str(countnn))
    
    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE NÚMEROS -----
    countcd = 0
    for word, tag in classificado: 
        if tag.startswith('CD'): #'CD' significa Cardinal Digit
            countcd = countcd + 1
    
    if countcd >= 0 and countcd <= 10:
        data_dict.update({'numbers1a10':'1'})
    elif countcd >= 11 and countcd <= 20:
        data_dict.update({'numbers11a20':'1'})
    elif countcd >= 21 and countcd <= 30:
        data_dict.update({'numbers21a30':'1'})
    elif countcd >= 31 and countcd <= 40:
        data_dict.update({'numbers31a40':'1'})
    elif countcd >= 41 and countcd <= 50:
        data_dict.update({'numbers41a50':'1'})
    elif countcd >= 51:
        data_dict.update({'numbers51up':'1'})

    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE NÚMEROS - ' + str(countcd))

    #-----POPULANDO O DICIONÁRIO COM 1 e 0----- NÚMERO DE PALAVRAS DE OUTRO IDIOMA -----
    countfw = 0
    for word, tag in classificado: 
        if tag.startswith('FW'): #'FW' significa Foreign Word
            countfw = countfw + 1
    
    if countfw >= 0 and countfw <= 10:
        data_dict.update({'plvr_outra_ling1a10':'1'})
    elif countfw >= 11 and countfw <= 20:
        data_dict.update({'plvr_outra_ling11a20':'1'})
    elif countfw >= 21 and countfw <= 30:
        data_dict.update({'plvr_outra_ling21a30':'1'})
    elif countfw >= 31 and countfw <= 40:
        data_dict.update({'plvr_outra_ling31a40':'1'})
    elif countfw >= 41 and countfw <= 50:
        data_dict.update({'plvr_outra_ling41a50':'1'})
    elif countfw >= 51:
        data_dict.update({'plvr_outra_ling51up':'1'})

    #TOTAL PALAVRAS DE OUTRO IDIOMA
    print('TOTAL DE PALAVRAS DE OUTRO IDIOMA - ' + str(countfw))

    #----CALCULANDO INDICE FLESCH----
    indice = textstat.flesch_reading_ease(raw['content'])    

    if indice >= 0.00 and indice <= 29.99:
        print("ÍNDICE FLESCH: " + str(indice))
        print("DIFICULDADE: VERY CONFUSING")
        data_dict.update({'Preco_arq': '3500.00'})
    elif indice >= 30.00 and indice <= 49.99:
        print("ÍNDICE FLESCH: " + str(indice))
        print("DIFICULDADE: DIFFICULT")
        data_dict.update({'Preco_arq': '2800.00'})
    elif indice >= 50.00 and indice <= 59.99:
        print("ÍNDICE FLESCH: " + str(indice))
        print("DIFICULDADE: FAIRLY DIFFICULT")
        data_dict.update({'Preco_arq': '2000.00'})
    elif indice >= 60.00 and indice <= 69.99:
        print("ÍNDICE FLESCH: " + str(indice))
        print("DIFICULDADE: STANDARD")
        data_dict.update({'Preco_arq': '1500.00'})
    elif indice >= 70.00 and indice <= 79.99:
        print("ÍNDICE FLESCH: " + str(indice))
        print("DIFICULDADE: FAIRLY EASY")
        data_dict.update({'Preco_arq': '1000.00'})
    elif indice >= 80.00 and indice <= 89.99:
        print("ÍNDICE FLESCH: " + str(indice))
        print("DIFICULDADE: EASY")
        data_dict.update({'Preco_arq': '800.00'})
    else:
        print("ÍNDICE FLESCH: " + str(indice))
        print("DIFICULDADE: VERY EASY")
        data_dict.update({'Preco_arq': '500.00'})

    #POPULANDO O DATAFRAME COM AS INFORMAÇÕES DO DICIONÁRIO
    df = df.append(data_dict, ignore_index=True)

    #----CONTAGEM TIPO DE PALAVRAS (VERIFICAR ARQUIVO LEGENDA.TXT)----
    print(Counter(tag for word,tag in classificado)) 

    print('\n') 
    

#---- CRIAÇÃO DO DATAFRAME COM OS DADOS DO DICIONÁRIO
df.to_csv('dataframe.csv', sep=';',encoding='cp1252')
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