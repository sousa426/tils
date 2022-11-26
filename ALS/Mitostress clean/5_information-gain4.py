import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
import time

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

Y = LabelEncoder().fit_transform(Y.astype('str'))

# Create Decision Tree classifer object
clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf2 = clf2.fit(X,Y)


print(clf2.feature_importances_)

sfm = SelectFromModel(clf2)
sfm.fit(X, Y)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(header_names_no_target[feature_list_index])

print("--- %s seconds ---" % round(time.time() - start_time,4))
