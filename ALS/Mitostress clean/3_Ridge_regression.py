import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
import time

start_time = time.time()

#https://www.datacamp.com/community/tutorials/feature-selection-python

# A helper method for pretty-printing the coefficients
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

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

Y = LabelEncoder().fit_transform(Y.astype('str'))


ridge = Ridge(alpha=1.0)
ridge.fit(X,Y)

print ("Ridge model:", pretty_print_coefs(ridge.coef_))

sfm = SelectFromModel(ridge)
sfm.fit(X, Y)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(header_names_no_target[feature_list_index])
    
print("--- %s seconds ---" % round(time.time() - start_time,4))


