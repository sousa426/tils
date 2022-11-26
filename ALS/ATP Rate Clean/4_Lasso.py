import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
import time

start_time = time.time()


#https://machinelearningmastery.com/lasso-regression-with-python/

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

Y = LabelEncoder().fit_transform(Y.astype('str'))


# define model
model = Lasso(alpha=1.0)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))


sfm = SelectFromModel(model)
sfm.fit(X, Y)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(header_names_no_target[feature_list_index])

print("--- %s seconds ---" % round(time.time() - start_time,4))
