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


mito_data = pd.read_excel('./dataset/Datasets_ALS.xlsx', 
                              sheet_name='Mitostress clean',engine='openpyxl')

mito_data.drop(['ID','Familiy','N','Data','Description'], axis=1, 
               inplace=True)

#print(list(mito_data.columns))
header_names = list(mito_data.columns)
header_names_no_target = header_names.copy()
header_names_no_target.remove('Target')


#split dataset in features and target variable
feature_cols_chi2 = ['Maximal respiration','Spare respiratory capacity',
                'Baseline OCR','Stressed OCR']

feature_cols_rfe = ['Proton leak','Spare respiratory capacity',
                'Baseline OCR','Baseline ECAR','Stressed ECAR']

feature_cols_ridge = ['Basal respiration ','Proton leak',
                'Spare respiratory capacity','Baseline OCR','Baseline ECAR']

feature_cols_lasso = ['Basal respiration ','Spare respiratory capacity',
                'Baseline OCR','Baseline ECAR']

feature_cols_infogain = ['Proton leak','ATP production',
                'Spare respiratory capacity','Baseline ECAR']

feature_cols_autofeat = ['Proton leak','Baseline OCR',
                'Baseline ECAR','Spare respiratory capacity']

feature_cols_tpot = ['Baseline ECAR','Stressed ECAR']

feature_cols_h2o = ['Baseline ECAR','Stressed ECAR',
                    'Spare respiratory capacity','Basal respiration ']

feature_cols_featuretools = ['Non mitochondrial respiration','Basal respiration ',
                    'Maximal respiration','Proton leak','Baseline ECAR']

features_by_rank_5 = ['Baseline ECAR', 'Spare respiratory capacity', 
                    'Proton leak', 'Baseline OCR', 'Basal respiration ']

features_by_rank_4 = ['Baseline ECAR', 'Spare respiratory capacity', 
                    'Proton leak', 'Baseline OCR']



X = mito_data[features_by_rank_4] # Features
#X = mito_data[header_names_no_target] # Features
y = mito_data.Target # Target variable


# Create Decision Tree classifer object
model = DecisionTreeClassifier()

predictions = cross_val_predict(model, X, y, cv=LeaveOneOut())
    
print("accuracy: " , metrics.accuracy_score(y, predictions))
print("precision: " , metrics.precision_score(y, predictions,average='macro'))

print(classification_report(y, predictions))


print("--- %s seconds ---" % round(time.time() - start_time,4))


