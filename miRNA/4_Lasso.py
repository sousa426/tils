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

selected_features = []
# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    #print(header_names_no_target[feature_list_index])
    selected_features.append(header_names_no_target[feature_list_index])


print(selected_features)

print("--- %s seconds ---" % round(time.time() - start_time,4))



"""
['let-7b', 'let-7c', 'let-7d', 'let-7e', 'let-7f', 'let-7g', 'let-7i', 
 'miR-1', 'miR-100', 'miR-106b*', 'miR-10b', 'miR-125a-5p', 'miR-125b',
 'miR-126', 'miR-126*', 'miR-130a', 'miR-141', 'miR-151-5p', 'miR-155',
 'miR-15a', 'miR-181a', 'miR-195', 'miR-196a', 'miR-196b', 'miR-200b', 
 'miR-200c', 'miR-203', 'miR-205', 'miR-206', 'miR-21', 'miR-21*', 'miR-22', 
 'miR-223', 'miR-23a', 'miR-23b', 'miR-24', 'miR-26a', 'miR-26b', 'miR-27a',
 'miR-28-3p', 'miR-29a', 'miR-29b', 'miR-29c', 'miR-30a', 'miR-30c', 'miR-30d', 
 'miR-31', 'miR-335', 'miR-365', 'miR-374a', 'miR-375', 'miR-378', 'miR-424',
 'miR-450a', 'miR-451', 'miR-484', 'miR-486-5p', 'miR-7', 'miR-768-5p', 
 'miR-92a', 'Candidate-5']

"""
