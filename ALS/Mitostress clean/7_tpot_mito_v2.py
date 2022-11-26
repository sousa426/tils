import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
import time
from tpot import TPOTClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

start_time = time.time()

mito_data = pd.read_excel('./dataset/Datasets_ALS.xlsx', 
                              sheet_name='Mitostress clean',engine='openpyxl')

mito_data.drop(['ID', 'Familiy','N','Data','Description'], axis=1, 
               inplace=True)

#print(mito_data.head())
header_names = list(mito_data.columns)
header_names_no_target = header_names.copy()
header_names_no_target.remove('Target')

#print(header_names)
print(header_names_no_target)

array = mito_data.values
X = array[:,1:12]
Y = array[:,0]

#X são os valores das features
#Y sao os valores dos targets

X = X.astype('float32')
Y = LabelEncoder().fit_transform(Y.astype('str'))


training_features, testing_features, training_target, testing_target = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25, random_state=42)



#Inicio Parte 1 - determinal melhor modelo
#so necessaro para o tpot escolher o melhor modelo
#generations = 5 quer dizer que faz 5 iterações - demora algum tempo 20 min
"""
tpot = TPOTClassifier(generations=5,verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mirna_best_modelv2_1.py')
#Fim Parte 1
"""
exported_pipeline = make_pipeline(
    PCA(iterated_power=4, svd_solver="randomized"),
    RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.4, min_samples_leaf=4, min_samples_split=20, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
print(exported_pipeline.named_steps['pca'].feature_names_in_)

print("--- %s seconds ---" % round(time.time() - start_time,4))

