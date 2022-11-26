import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

import time

start_time = time.time()

#https://github.com/EpistasisLab/tpot
#http://epistasislab.github.io/tpot/examples/


mito_data = pd.read_excel('../dataset/Datasets_ALS.xlsx', 
                              sheet_name='ATP Rate clean',engine='openpyxl')

mito_data.drop(['ID', 'Family','N','Date','Description'], axis=1, 
               inplace=True)


#print(mito_data.head())
header_names = list(mito_data.columns)
header_names_no_target = header_names.copy()
header_names_no_target.remove('Target')

#print(header_names)
#print(header_names_no_target)

array = mito_data.values
X = array[:,1:12]
Y = array[:,0]

#X são os valores das features
#Y sao os valores dos targets

X = X.astype('float32')
Y = LabelEncoder().fit_transform(Y.astype('str'))


X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42,cv=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mito_best_model100.py')


"""
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, 
                                                    test_size=0.25, 
                                                    random_state=42)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

tpot = TPOTClassifier(generations=1, population_size=50, verbosity=2, 
                     random_state=42,cv=cv)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

"""

"""
# define model evaluation
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
model = TPOTClassifier(generations=1, population_size=50, cv=cv, 
                       scoring='accuracy', verbosity=2, random_state=1,
                       n_jobs=-1)



model.fit(X, Y)

model.export('tpot_mito_best_model.py')

"""
print("--- %s seconds ---" % round(time.time() - start_time,4))

