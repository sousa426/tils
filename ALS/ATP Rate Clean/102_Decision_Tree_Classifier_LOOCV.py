import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
#from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#from matplotlib import pyplot as plt
#from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
import time

start_time = time.time()


#https://www.datacamp.com/community/tutorials/decision-tree-classification-python
#https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6



mito_data = pd.read_excel('../dataset/Datasets_ALS.xlsx', 
                              sheet_name='ATP Rate clean',engine='openpyxl')

mito_data.drop(['ID', 'Family','N','Date','Description'], axis=1, 
               inplace=True)

#print(list(mito_data.columns))
header_names = list(mito_data.columns)
header_names_no_target = header_names.copy()
header_names_no_target.remove('Target')


#split dataset in features and target variable
feature_cols_chi2 = ['Total ATP Production Rate (Basal) (pmol/min)', 
                     'glyco-ATP Production Rate (Basal) (pmol/min)', 
                     'mito-ATP Production Rate (Basal) (pmol/min)', 
                     'XF ATP Rate Index (Basal)']

feature_cols_rfe = ['glyco-ATP Production Rate (Basal) (pmol/min)', 
                    'XF ATP Rate Index (Basal)']

feature_cols_ridge = ['XF ATP Rate Index (Basal)']

feature_cols_lasso = ['glyco-ATP Production Rate (Basal) (pmol/min)',
                     'mito-ATP Production Rate (Basal) (pmol/min)']

feature_cols_infogain = ['glyco-ATP Production Rate (Basal) (pmol/min)',
                'XF ATP Rate Index (Basal)']

feature_cols_autofeat = ['XF ATP Rate Index (Basal)']

feature_cols_tpot = ['glyco-ATP Production Rate (Basal) (pmol/min)',
                'XF ATP Rate Index (Basal)']

feature_cols_h2o = ['glyco-ATP Production Rate (Basal) (pmol/min)',
                'XF ATP Rate Index (Basal)']

feature_cols_featuretools = ['Total ATP Production Rate (Basal) (pmol/min)', 
                     'glyco-ATP Production Rate (Basal) (pmol/min)', 
                     'mito-ATP Production Rate (Basal) (pmol/min)', 
                     'XF ATP Rate Index (Basal)']



X = mito_data[feature_cols_chi2] # Features
#X = mito_data[header_names_no_target] # Features
y = mito_data.Target # Target variable


# Create Decision Tree classifer object
model = DecisionTreeClassifier()

predictions = cross_val_predict(model, X, y, cv=LeaveOneOut())
    
print("accuracy: " , metrics.accuracy_score(y, predictions))
print("precision: " , metrics.precision_score(y, predictions,average='macro'))

print(classification_report(y, predictions))


print("--- %s seconds ---" % round(time.time() - start_time,4))


