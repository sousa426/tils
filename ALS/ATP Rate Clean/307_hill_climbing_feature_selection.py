import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics, preprocessing #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import  cross_val_predict, LeaveOneOut
import random
import numpy
from progress_bar import printProgressBar
import re
import time
from pathlib import Path



import sys

#https://machinelearningmastery.com/stochastic-hill-climbing-in-python-from-scratch/
#https://machinelearningmastery.com/hill-climb-the-test-set-for-machine-learning/
#https://thispointer.com/pandas-add-two-columns-into-a-new-column-in-dataframe/


######################### FUNÇÔES ####################################
##Input: 
##  oper - tipo de operação (string)
##
##Output:
##  numero da operação (integer)

def get_number_of_oper(oper):
    switcher = {
        'add': 1,
        'sub': 2,
        'mult': 3,
        'div': 4,
        'sqrt': 5,
        'exp': 6,
        'del': 7,
        'new': 8      
    }
    return switcher.get(oper)


##Input: 
##    cols - nome das colunas para fazer operações
##    df - dataframe com os dados
##    full_df - dataframe completo para ir buscar valores da feature nova
##
def select_operation_ils(cols,df,full_df):
    
   
    l1 = df.columns.to_list()##dataset atual
    l2 = full_df.columns.to_list() ##dataset completo
    col_disp = list(set(l2)-set(l1)) ##colunas restantes da diferença entre ambos
    
    max_int_value = 0
    if len(col_disp) == 0:
        max_int_value = 7
    else:
        max_int_value = 8
    
    rand = random.randint(1,max_int_value)
    
    if rand == 1:       
        tp = 'add'
        col_name = ' + '.join(cols.columns.values.tolist())        
        cs_name = 'Adicionar: '+cols.columns.to_list()[0] + ' e '+cols.columns.to_list()[1]
        op = df[cols.columns.to_list()[0]] + df[cols.columns.to_list()[1]]
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)          
    elif rand == 2:       
        tp = 'sub'
        col_name = ' - '.join(cols.columns.values.tolist())       
        cs_name = 'Subtrair: '+cols.columns.to_list()[0] + ' e '+cols.columns.to_list()[1]
        op = df[cols.columns.to_list()[0]] - df[cols.columns.to_list()[1]]
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)        
    elif rand == 3:       
        tp = 'mult'
        col_name = ' * '.join(cols.columns.values.tolist())      
        cs_name = 'Multiplicar: '+cols.columns.to_list()[0] + ' e '+cols.columns.to_list()[1]
        op = df[cols.columns.to_list()[0]] * df[cols.columns.to_list()[1]]
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)       
    elif rand == 4:       
        tp = 'div'
        col_name = ' / '.join(cols.columns.values.tolist())       
        cs_name = 'Dividir: '+cols.columns.to_list()[0] + ' e '+cols.columns.to_list()[1]
        op = df[cols.columns.to_list()[0]].divide(df[cols.columns.to_list()[1]])
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)       
    elif rand == 5:        
        tp = 'sqrt'
        col_name = tp+' '+cols.columns.values.tolist()[0]     
        cs_name = 'Sqrt: '+cols.columns.to_list()[0]
        op = numpy.sqrt(df[cols.columns.to_list()[0]])
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)       
    elif rand == 6:      
        tp = 'exp'
        col_name = tp+' '+cols.columns.values.tolist()[0]      
        cs_name = 'Exp: '+cols.columns.to_list()[0]
        op = numpy.exp(df[cols.columns.to_list()[0]])
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)       
    elif rand == 7:        
        tp = 'del'
        col_name = cols.columns.values.tolist()[0]       
        cs_name = 'Apagar: '+cols.columns.to_list()[0]
        op = ''       
    elif rand == 8:       
        tp = 'new'
        col_name_ =  random.choice(col_disp)       
        col_name = col_name_
        cs_name = 'Nova: '+col_name_
        op = full_df[col_name_]       
    
    return tp,op,col_name,cs_name   

##Input: 
##    cols - nome das colunas para fazer operações
##    df - dataframe com os dados
##    full_df - dataframe completo para ir buscar valores da feature nova
##    last_operation - ultima operação com melhoria de performance
##
##Output:
##   tp - tipo de operação (add,sub,mult,div,sqrt,exp,del)
##   op - valores da operação para a nova coluna
##   col_name - nome da coluna para gravar no dataframe
##   cs_name - nome da operação para mostrar na consola
##   col_name_ - nome da coluna nova das que estao disponiveis pra escolha
##   res_oper - nome da coluna para guardar os resultados 

def select_operation(cols,df,full_df,last_operation):       
    
    l1 = df.columns.to_list()##dataset atual
    l2 = full_df.columns.to_list() ##dataset completo
    col_disp = list(set(l2)-set(l1)) ##colunas restantes da diferença entre ambos
    
    max_int_value = 0
    if len(col_disp) == 0:
        max_int_value = 7
    else:
        max_int_value = 8
    
    if tendency == 1:
        if last_operation == '':   
            rand = random.randint(1,max_int_value)
        else:
            if len(col_disp) == 0:
                rand = random.randint(1,max_int_value)
            else:
                rand = last_operation
    else:
        rand = random.randint(1,max_int_value) 
    
    if rand == 1:
        resultados['Adicionar']['Total'] += 1
        tp = 'add'
        col_name = ' + '.join(cols.columns.values.tolist())
        col_name_=''
        cs_name = 'Adicionar: '+cols.columns.to_list()[0] + ' e '+cols.columns.to_list()[1]
        op = df[cols.columns.to_list()[0]] + df[cols.columns.to_list()[1]]
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)
        res_oper = 'Adicionar'        
    elif rand == 2:
        resultados['Subtrair']['Total'] += 1
        tp = 'sub'
        col_name = ' - '.join(cols.columns.values.tolist())
        col_name_=''
        cs_name = 'Subtrair: '+cols.columns.to_list()[0] + ' e '+cols.columns.to_list()[1]
        op = df[cols.columns.to_list()[0]] - df[cols.columns.to_list()[1]]
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)
        res_oper = 'Subtrair' 
    elif rand == 3:
        resultados['Multiplicar']['Total'] += 1
        tp = 'mult'
        col_name = ' * '.join(cols.columns.values.tolist())
        col_name_=''
        cs_name = 'Multiplicar: '+cols.columns.to_list()[0] + ' e '+cols.columns.to_list()[1]
        op = df[cols.columns.to_list()[0]] * df[cols.columns.to_list()[1]]
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)
        res_oper = 'Multiplicar' 
    elif rand == 4:
        resultados['Dividir']['Total'] += 1
        tp = 'div'
        col_name = ' / '.join(cols.columns.values.tolist())
        col_name_=''
        cs_name = 'Dividir: '+cols.columns.to_list()[0] + ' e '+cols.columns.to_list()[1]
        op = df[cols.columns.to_list()[0]].divide(df[cols.columns.to_list()[1]])
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)
        res_oper = 'Dividir'
    elif rand == 5:
        resultados['Sqrt']['Total'] += 1
        tp = 'sqrt'
        col_name = tp+' '+cols.columns.values.tolist()[0]
        col_name_=''
        cs_name = 'Sqrt: '+cols.columns.to_list()[0]
        op = numpy.sqrt(df[cols.columns.to_list()[0]])
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)
        res_oper = 'Sqrt'
    elif rand == 6:
        resultados['Exp']['Total'] += 1
        tp = 'exp'
        col_name = tp+' '+cols.columns.values.tolist()[0]
        col_name_=''
        cs_name = 'Exp: '+cols.columns.to_list()[0]
        op = numpy.exp(df[cols.columns.to_list()[0]])
        op = op.replace([numpy.inf, -numpy.inf,numpy.nan], 0)
        res_oper = 'Exp'
    elif rand == 7:
        resultados['Apagar']['Total'] += 1
        tp = 'del'
        col_name = cols.columns.values.tolist()[0]
        col_name_=''
        cs_name = 'Apagar: '+cols.columns.to_list()[0]
        op = ''
        res_oper = 'Apagar'
    elif rand == 8: 
        resultados['Nova']['Total'] += 1
        tp = 'new'             
        col_name_ =  random.choice(col_disp)       
        col_name = col_name_
        cs_name = 'Nova: '+col_name_
        op = full_df[col_name_]
        res_oper = 'Nova'
    
    return tp,op,col_name,cs_name,col_name_,res_oper


######################### VARIAVEIS GLOBAIS #########################
decimal_cases = 3 ##casas decimais para o resultado
decimal_cases2 = 6 ## casas decimais para accuracy
type_of_run = 'loop' ## loop or target
type_of_run_value = 1000 ##se for target entao valor entre [0,1], tipo 0.75
verbose = 2 ## 1- mostra progressbar, 2 - nao mostra progressbar e mostra os detalhes 
size_of_inicial_dataset = 0.5  #escala de [0,1], tipo 0.2
tendency = 1 ## ativa ou desativa a tendenciação da operação. 0-Falso, 1- True
mumber_to_perturbe = 50
number_features_add = 5

##valor maximo do loop, condição de paragem pra nao estar a correr infinatamente.
max_iteration_loop = 100000

######################### INICIAR PROGRESS BAR #########################
if verbose == 1 :
    l = type_of_run_value if type_of_run == 'loop' else max_iteration_loop
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

total_iterations = 30
average_accuracy = []
average_time = []
for loop in range(26,total_iterations):
    start_time = time.time()   
    filename = './logs/loop_'+str(loop)+'.txt'    
    f = open(filename, "w")    
     
    ######################### VARIAVEIS RELATORIO #########################
    ##lista de features que foram apagadas
    cols_removed = []
    cols_removed_old = []
    ##lista de features que foram adicioanas ao dataset
    cols_new = []
    cols_new_old = []
    
    ##resultados da operaçoes
    resultados = {
      "Adicionar" : {
        "Total" : 0,
        "Total Usado" : 0,
        "Ganho" : 0,
        "%_Ganho" : 0
      },
     "Subtrair" : {
        "Total" : 0,
        "Total Usado" : 0,
        "Ganho" : 0,
        "%_Ganho" : 0
      },
     "Multiplicar" : {
        "Total" : 0,
        "Total Usado" : 0,
        "Ganho" : 0,
        "%_Ganho" : 0
      },
     "Dividir" : {
        "Total" : 0,
        "Total Usado" : 0,
        "Ganho" : 0,
        "%_Ganho" : 0
      }, 
     "Sqrt" : {
        "Total" : 0,
        "Total Usado" : 0,
        "Ganho" : 0,
        "%_Ganho" : 0
      },
      "Exp" : {
        "Total" : 0,
        "Total Usado" : 0,
        "Ganho" : 0,
        "%_Ganho" : 0
      },
       "Apagar" : {
        "Total" : 0,
        "Total Usado" : 0,
        "Ganho" : 0,
        "%_Ganho" : 0
      },
       "Nova" : {
        "Total" : 0,
        "Total Usado" : 0,
        "Ganho" : 0,
        "%_Ganho" : 0
      }
    }
    
    
    
    
    
    
    ######################### LER DATASET #########################
    mito_data = pd.read_excel('../dataset/Datasets_ALS.xlsx', sheet_name='ATP Rate clean',engine='openpyxl')
    y = mito_data.Target # Target variable
    
    mito_data.drop(['ID', 'Family','N','Date','Description','Target'], axis=1, inplace=True)
    
 
    
    ######################### ALTERAR NOMES COLUNAS #########################
    ##mudar os nomes das features para F1, F2, .... F100
    temp_cols = mito_data.columns
    final_cols = ['F' + str(x+1) for x in range(len(temp_cols))]
    mito_data.columns = final_cols
    
    ##CRIAR UMA ESPECIE DE DICIONARIO PARA SABER A CORRESPONDENCIA ENTRE COLUNAS
    zipbObj = zip(final_cols,temp_cols.to_list())
    dictOfWords = dict(zipbObj)
    ##print(dictOfWords)
    
    """
    f = open("dict.txt", "w")
    for k in dictOfWords.keys():
        f.write("'{}' - '{}'\n".format(k, dictOfWords[k]))
    f.close()
    
    sys.exit()
    """
    
    
    ######################### ESCOLHER FEATURES INICIAIS #########################
    #escolher aleatoriamente "size_of_inicial_dataset" features (percentage)
    n_features_ini = round(len(mito_data.columns) * size_of_inicial_dataset)
    X = mito_data.sample(n=n_features_ini,axis='columns')
    
    ######################### FEATURES RESTANTES PARA SELECAO #########################
    #colunas novas disponiveis para selecionar
    #l1 = X.columns.to_list()
    #l2 = mirna.columns.to_list()
    #col_disp_select = list(set(l2)-set(l1))
    
    ######################### NORMALIZAR DATASET #########################
    #Normalizar os dados para uma escala de [0,1] -  Coluna por coluna!
    scaler = preprocessing.MinMaxScaler() 
    arr_scaled = scaler.fit_transform(X) 
    X = pd.DataFrame(arr_scaled, columns=X.columns,index=X.index)
    
    
    
    ######################### FAZER 1A AVALIAÇÂO #########################
    predictions_ini1 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
    predictions_ini2 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
    predictions_ini3 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
    
    best_accuracy1 = round(metrics.accuracy_score(y, predictions_ini1),decimal_cases2)
    best_accuracy2 = round(metrics.accuracy_score(y, predictions_ini2),decimal_cases2)
    best_accuracy3 = round(metrics.accuracy_score(y, predictions_ini3),decimal_cases2)
    
    best_accuracy = round((best_accuracy1+best_accuracy2+best_accuracy3)/3,decimal_cases2)
    first_accuracy = best_accuracy
    if verbose == 2 :
        print(best_accuracy)
        print(best_accuracy,file=f)
    
    
        
    ######################### TREPA COLINAS VITAMINADO #########################
    last_oper_benefit = ''
    i = 0 #numero de interações sem melhorias
    Y_temp = pd.DataFrame()
    accuracy_y_temp = 0
    accuracy = 0
    
    for x in range(type_of_run_value if type_of_run == 'loop' else max_iteration_loop):
           
        if i >= mumber_to_perturbe:
            if Y_temp.empty == False:           
                            
                predictions_1 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
                predictions_2 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
                predictions_3 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
                
                accuracy_1 = round(metrics.accuracy_score(y, predictions_1),decimal_cases2) 
                accuracy_2 = round(metrics.accuracy_score(y, predictions_2),decimal_cases2)
                accuracy_3 = round(metrics.accuracy_score(y, predictions_3),decimal_cases2) 
                
                accuracy_x = round((accuracy_1+accuracy_2+accuracy_3)/3,decimal_cases2)
                
                
                predictions_y_1 = cross_val_predict(DecisionTreeClassifier(), Y_temp, y, cv=LeaveOneOut(),n_jobs=-1)
                predictions_y_2 = cross_val_predict(DecisionTreeClassifier(), Y_temp, y, cv=LeaveOneOut(),n_jobs=-1)
                predictions_y_3 = cross_val_predict(DecisionTreeClassifier(), Y_temp, y, cv=LeaveOneOut(),n_jobs=-1)
                
                accuracy_y_1 = round(metrics.accuracy_score(y, predictions_y_1),decimal_cases2) 
                accuracy_y_2 = round(metrics.accuracy_score(y, predictions_y_2),decimal_cases2)
                accuracy_y_3 = round(metrics.accuracy_score(y, predictions_y_3),decimal_cases2) 
                
                accuracy_y_temp = round((accuracy_y_1+accuracy_y_2+accuracy_y_3)/3,decimal_cases2)
                
                ##avaliação para determinar com qual fico - novo ou antigo
                if accuracy_y_temp > accuracy_x:
                    X = Y_temp.copy()   
                    best_accuracy = accuracy_y_temp
                    cols_removed = cols_removed_old.copy()
                    cols_new = cols_new_old.copy()
                    print('ILS - Comparação de Datasets - fico com o antigo')
                    print('ILS - Comparação de Datasets - fico com o antigo',file=f)
                else:
                    del Y_temp
                    Y_temp = pd.DataFrame() 
                    best_accuracy = accuracy_x
                    cols_removed_old.clear()
                    cols_new_old.clear()
                    print('ILS - Comparação de Datasets - fico com o atual')
                    print('ILS - Comparação de Datasets - fico com o atual',file=f)
                  
            print('xxxxxxx Perturbação xxxxxxxxxxxx')
            print('xxxxxxx Perturbação xxxxxxxxxxxx',file=f)
           
            Y_temp = X.copy()
            cols_removed_old  = cols_removed.copy()
            cols_new_old = cols_new.copy()
            
            ###adicionar a perturbação ao conjunto atual
            for x2 in range(number_features_add):
                colunas = X.sample(n=2,axis='columns')
                oper,collum_value,collum_name,console_name= select_operation_ils(colunas,X,mito_data)
                        
                print(console_name)
                print(console_name,file=f)
                
                if oper == 'del':              
                    del  X[collum_name]
                else:
                    X[collum_name] = collum_value
                    
              
                    
                arr_scaled = scaler.fit_transform(X) 
                X = pd.DataFrame(arr_scaled, columns=X.columns,index=X.index)            
               
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',file=f)
            i = 0        
        else:  
             #escolher duas colunas aleatoriamente 
            colunas = X.sample(n=2,axis='columns')
            #print(colunas.columns.values.tolist()) #colunas selecionadas para operação aritmetica
            #X['_'.join(colunas.columns.values.tolist())] = colunas.sum(axis=1) #nome da coluna com o nome das colunas escolhidas
            oper,collum_value,collum_name,console_name,collum_name_new,res_oper_name = select_operation(colunas,X,mito_data,last_oper_benefit)
            
            if verbose == 2 :
                print(x,'-',console_name)
                print(x,'-',console_name,file=f)
            
            if oper == 'del':
                X_temp = X.copy()
                del  X[collum_name]
            else:
                X[collum_name] = collum_value
                              
       
            arr_scaled = scaler.fit_transform(X) 
            X = pd.DataFrame(arr_scaled, columns=X.columns,index=X.index)
            
            predictions1 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
            ##predictions2 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
            ##predictions3 = cross_val_predict(DecisionTreeClassifier(), X, y, cv=LeaveOneOut(),n_jobs=-1)
            
            accuracy1 = round(metrics.accuracy_score(y, predictions1),decimal_cases2) 
            ##accuracy2 = round(metrics.accuracy_score(y, predictions2),decimal_cases2)
            ##accuracy3 = round(metrics.accuracy_score(y, predictions3),decimal_cases2)
            
            #accuracy = round((accuracy1+accuracy2+accuracy3)/3,decimal_cases2)
            
            accuracy = accuracy1
            
            if accuracy > best_accuracy :
                resultados[res_oper_name]['Total Usado'] += 1
                resultados[res_oper_name]['Ganho'] += round(accuracy-best_accuracy,decimal_cases)
                resultados[res_oper_name]['%_Ganho'] += round(((100*accuracy)/best_accuracy)-100,decimal_cases)
                last_oper_benefit = get_number_of_oper(oper)
                
                if verbose == 2 :
                    print(accuracy)
                    print(accuracy,file=f)
                
                best_accuracy = accuracy
               
                
                if oper == 'new': 
                    cols_new.append(collum_name)
                if oper== 'del':
                    cols_removed.append(collum_name)
                i = 0
               
            else:
                if oper == 'del':
                    X = X_temp.copy()
                else:
                    del  X[collum_name]
                last_oper_benefit = ''
                i+=1
                
        if verbose == 1 :
            printProgressBar(x + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
         
        if type_of_run == 'target':
            if best_accuracy >= type_of_run_value:
                break
                        
           
    print("Accuracy Inicial: ",first_accuracy)   
    print("Accuracy Final: ",best_accuracy)          
    print("Accuracy Improvemnt : ",round(((100*best_accuracy)/first_accuracy)-100,decimal_cases), "%") 
    res_out = X.columns.to_list()
    res_traduzido = [re.sub(r'\b\w+\b', lambda m: dictOfWords.get(m.group(), m.group()), s) for s in res_out]
    print(res_traduzido)
    print("Features Removed:",[re.sub(r'\b\w+\b', lambda m: dictOfWords.get(m.group(), m.group()), s) for s in cols_removed])
    print("Features New:",[re.sub(r'\b\w+\b', lambda m: dictOfWords.get(m.group(), m.group()), s) for s in cols_new])
    print(resultados)
    
    
    print("Accuracy Inicial: ",first_accuracy,file=f)
    print("Accuracy Final: ",best_accuracy,file=f)
    print("Accuracy Improvemnt : ",round(((100*best_accuracy)/first_accuracy)-100,decimal_cases), "%",file=f)
    print(res_traduzido,file=f)
    print("Features Removed:",[re.sub(r'\b\w+\b', lambda m: dictOfWords.get(m.group(), m.group()), s) for s in cols_removed],file=f)
    print("Features New:",[re.sub(r'\b\w+\b', lambda m: dictOfWords.get(m.group(), m.group()), s) for s in cols_new],file=f)
    print(resultados,file=f)
    
    average_accuracy.append(best_accuracy) 
    average_time.append(round(time.time() - start_time,4)) 
    
    f.close()


file_name2 = './logs/resume.txt'
f = open(file_name2, "w")
print('average_accuracy',file=f)
print(average_accuracy,file=f)
print(sum(average_accuracy) / len(average_accuracy),file=f)
print('average_time',file=f)
print(average_time,file=f)
print(sum(average_time) / len(average_time),file=f)
f.close()