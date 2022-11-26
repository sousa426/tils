import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from tpot import TPOTClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier

import time

start_time = time.time()


#https://github.com/EpistasisLab/tpot
#http://epistasislab.github.io/tpot/examples/


mirna = pd.read_excel('./dataset/DataSet_Bioinfo1718.xlsx', 
                              sheet_name='Sheet1',engine='openpyxl')


#print(mito_data.head())
header_names = list(mirna.columns)
header_names_no_target = header_names.copy()
header_names_no_target.remove('ID')

#print(header_names)
#print(header_names_no_target)


array = mirna.values
X = array[:,1:715]
Y = array[:,0]


#X são os valores das features
#Y sao os valores dos targets

X = X.astype('float32')
Y = LabelEncoder().fit_transform(Y.astype('str'))

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25, random_state=42)



#Inicio Parte 1 - determinal melhor modelo
#so necessaro para o tpot escolher o melhor modelo
#generations = 5 quer dizer que faz 5 iterações - demora algum tempo 20 min

tpot = TPOTClassifier(generations=5,verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mirna_best_modelv2_1.py')
#Fim Parte 1



#Inicio Parte 2
#pegar no ficheiro criado e ir la buscar a pipeline
#corrigir os imports
#tentar arranjar forma consuante o pipeline de saber as features


print("--- %s seconds ---" % round(time.time() - start_time,4))

