import pandas as pd
from sklearn.preprocessing import LabelEncoder
from autofeat import AutoFeatClassifier 
#from autofeat import FeatureSelector
from sklearn.model_selection import train_test_split

import time

start_time = time.time()

#https://github.com/cod3licious/autofeat/blob/master/autofeat_examples.ipynb

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

X = X.astype('float32')
Y = LabelEncoder().fit_transform(Y.astype('str'))

"""
#fsel = FeatureSelector(verbose=1)
fsel = FeatureSelector(problem_type="classification",featsel_runs=5,verbose=1)

#print(pd.DataFrame(X,columns=header_names_no_target))


new_X = fsel.fit_transform(pd.DataFrame(X,  
                                        columns=header_names_no_target), Y)

print(new_X.columns)


"""

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=.3,random_state =0)

model = AutoFeatClassifier(verbose=1, feateng_steps=1)

df = model.fit_transform(X, Y)
y_pred = model.predict(X_test)

print("Final Accuracy: %.4f" % model.score(df, Y))
print("autofeat new features:", len(model.new_feat_cols_))


print("--- %s seconds ---" % round(time.time() - start_time,4))

"""
5 runs
['miR-133a', 'miR-513c', 'miR-328', 'miR-765', 'miR-374b', 'miR-199b-3p']

20 runs 
['miR-133a', 'miR-513c', 'miR-328', 'miR-765']
"""