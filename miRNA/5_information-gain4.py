import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
#from sklearn.model_selection import cross_val_score

import time

start_time = time.time()


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

# Create Decision Tree classifer object
clf2 = DecisionTreeClassifier(criterion = "entropy",max_features='auto')


#print(cross_val_score(clf2, X, Y, cv=10))

# Train Decision Tree Classifer
clf2 = clf2.fit(X,Y)


print(clf2.feature_importances_)

sfm = SelectFromModel(clf2)
sfm.fit(X, Y)

selected_features = []
# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    #print(header_names_no_target[feature_list_index])
    selected_features.append(header_names_no_target[feature_list_index])


print(selected_features)
print("--- %s seconds ---" % round(time.time() - start_time,4))


"""
['miR-133a', 'miR-194', 'miR-324-5p', 'miR-338-3p', 'miR-497', 'miR-509-3p',
 'miR-524-5p', 'miR-944', 'Candidate-54-3-3p']
"""