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



Y = LabelEncoder().fit_transform(Y.astype('str'))


ridge = Ridge(alpha=1.0)
#ridge.fit(X,Y)

#print ("Ridge model:", pretty_print_coefs(ridge.coef_))

sfm = SelectFromModel(ridge)
sfm.fit(X, Y)

selected_features = []
# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    #print(header_names_no_target[feature_list_index])
    selected_features.append(header_names_no_target[feature_list_index])


print(selected_features)

print("--- %s seconds ---" % round(time.time() - start_time,4))



"""
['let-7a*', 'let-7b', 'let-7b*', 'let-7c', 'let-7d', 'let-7e', 'let-7f', 
 'let-7g', 'miR-1', 'miR-100', 'miR-101', 'miR-103', 'miR-106b*', 'miR-107',
 'miR-10a', 'miR-10b', 'miR-125a-5p', 'miR-125b', 'miR-126*', 'miR-128',
 'miR-130a', 'miR-130b', 'miR-133a', 'miR-133b', 'miR-135a', 'miR-135b', 
 'miR-136', 'miR-140-5p', 'miR-141', 'miR-142-3p', 'miR-142-5p', 'miR-143*', 
 'miR-144', 'miR-145*', 'miR-146a', 'miR-146b-5p', 'miR-148a', 'miR-150', 
 'miR-151-3p', 'miR-151-5p', 'miR-155', 'miR-15a', 'miR-16', 'miR-16-2*',
 'miR-17', 'miR-17*', 'miR-181a', 'miR-181a*', 'miR-181b', 'miR-181c', 
 'miR-184', 'miR-185', 'miR-188-5p', 'miR-191', 'miR-192', 'miR-193a-3p',
 'miR-193a-5p', 'miR-193b*', 'miR-194', 'miR-195', 'miR-196a', 'miR-196b',
 'miR-197', 'miR-199a-5p', 'miR-199b-3p', 'miR-199b-5p', 'miR-19a', 'miR-19b',
 'miR-200a', 'miR-200a*', 'miR-200b', 'miR-203', 'miR-204', 'miR-205', 'miR-206', 
 'miR-210', 'miR-214', 'miR-215', 'miR-22', 'miR-221', 'miR-222', 'miR-223', 
 'miR-224', 'miR-23a', 'miR-23b', 'miR-24', 'miR-25', 'miR-26a', 'miR-26b',
 'miR-27a', 'miR-27a*', 'miR-27b*', 'miR-28-3p', 'miR-28-5p', 'miR-29a', 
 'miR-29b', 'miR-29c', 'miR-30a', 'miR-30a*', 'miR-30b', 'miR-30d', 'miR-30d*',
 'miR-30e', 'miR-31', 'miR-320a', 'miR-324-5p', 'miR-326', 'miR-335', 
 'miR-338-3p', 'miR-339-3p', 'miR-339-5p', 'miR-342-3p', 'miR-34a', 'miR-34b',
 'miR-34b*', 'miR-34c-3p', 'miR-34c-5p', 'miR-365', 'miR-374a', 'miR-374b',
 'miR-377', 'miR-378', 'miR-378*', 'miR-423-5p', 'miR-424', 'miR-424*', 
 'miR-425', 'miR-425*', 'miR-450a', 'miR-451', 'miR-452', 'miR-455-3p',
 'miR-483-5p', 'miR-484', 'miR-486-5p', 'miR-497', 'miR-503', 'miR-505',
 'miR-574-3p', 'miR-652', 'miR-660', 'miR-663', 'miR-671-5p', 'miR-675', 
 'miR-7', 'miR-708', 'miR-765', 'miR-873', 'miR-876-5p', 'miR-877', 
 'miR-886-5p', 'miR-9', 'miR-923', 'miR-92a-1*', 'miR-92b', 'miR-93',
 'miR-93*', 'miR-940', 'miR-944', 'miR-95', 'miR-98', 'miR-99a', 'miR-99a*',
 'miR-99b', 'Candidate-2', 'Candidate-5', 'Candidate-8', 'Candidate-14',
 'Candidate-15', 'Candidate-21-3p', 'Candidate-21-5p', 'Candidate-29-5p', 
 'Candidate-36-3p', 'Candidate-51-5p']
"""






