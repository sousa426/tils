import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive
import time
from tpot import TPOTClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

start_time = time.time()

mito_data = pd.read_excel('../dataset/Datasets_ALS.xlsx', 
                              sheet_name='ATP Rate clean',engine='openpyxl')

mito_data.drop(['ID', 'Family','N','Date','Description'], axis=1, 
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
#generations = 5 quer dizer que faz 5 iterações - demora algum tempo

"""
tpot = TPOTClassifier(verbosity=2)
tpot.fit(training_features, training_target)
print(tpot.score(testing_features, testing_target))
tpot.export('tpot_best_modelv2.py')
#Fim Parte 1

"""


exported_pipeline = make_pipeline(
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.8500000000000001, n_estimators=100), step=0.8),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=1.0, max_depth=6, max_features=0.7000000000000001, min_samples_leaf=8, min_samples_split=5, n_estimators=100, subsample=0.25)),
    RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.8, min_samples_leaf=2, min_samples_split=14, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
#print(exported_pipeline.named_steps['rfe'].coef_)

selected_features = []
for feature_list_index in exported_pipeline.named_steps['rfe'].get_support(indices=True):
    #print(header_names_no_target[feature_list_index])
    selected_features.append(header_names_no_target[feature_list_index])
    
print(selected_features)


print("--- %s seconds ---" % round(time.time() - start_time,4))

