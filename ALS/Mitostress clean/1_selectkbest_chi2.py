import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pandas import DataFrame
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
#print(header_names_no_target)

array = mito_data.values
X = array[:,1:12]
Y = array[:,0]

#X são os valores das features
#Y sao os valores dos targets

# Feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision=3)
#print(fit.scores_)

#features = fit.transform(X)

df = DataFrame(fit.scores_,columns=['Scores'])
i = df.nlargest(4, 'Scores')
index_of_selected_features = i.index.values.tolist()
#print(index_of_selected_features)

for xx in range(len(header_names_no_target)): 
    if xx in index_of_selected_features:
        print(header_names_no_target[xx])

print("--- %s seconds ---" % round(time.time() - start_time,4))