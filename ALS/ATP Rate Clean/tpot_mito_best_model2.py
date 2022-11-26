import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import FunctionTransformer
from tpot.builtins import StackingEstimator
from copy import copy
from sklearn.neural_network import MLPClassifier

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

training_features, testing_features, training_target, testing_target = train_test_split(X, Y, random_state=42)


"""
# Average CV score on the training set was: 0.7914556962025316
exported_pipeline = make_pipeline(
    make_union(
        MaxAbsScaler(),
        Normalizer(norm="l1")
    ),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.5, min_samples_leaf=2, min_samples_split=12, n_estimators=100)),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.2, min_samples_leaf=3, min_samples_split=8, n_estimators=100)
)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)
"""
# Average CV score on the training set was: 0.8064556962025318
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        Normalizer(norm="l1")
    ),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.5, min_samples_leaf=2, min_samples_split=11, n_estimators=100)),
    StackingEstimator(estimator=MLPClassifier(alpha=0.1, learning_rate_init=0.01)),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.2, min_samples_leaf=12, min_samples_split=3, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)


"""
exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

"""
exported_pipeline.fit(testing_features, testing_target)


print(exported_pipeline.feature_importances_)

sfm = SelectFromModel(exported_pipeline)
sfm.fit(testing_features, testing_target)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(header_names_no_target[feature_list_index])
    
