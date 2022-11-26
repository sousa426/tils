import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pandas import DataFrame

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

# Feature extraction
test = SelectKBest(score_func=chi2, k='all')
fit = test.fit(X, Y)

# Summarize scores
np.set_printoptions(precision=3)
#print(fit.scores_)

#features = fit.transform(X)

df = DataFrame(fit.scores_,columns=['Scores'])
i = df.nlargest(40, 'Scores')
index_of_selected_features = i.index.values.tolist()
#print(index_of_selected_features)

selected_features = []
for xx in range(len(header_names_no_target)): 
    if xx in index_of_selected_features:
        ##print(header_names_no_target[xx])
         selected_features.append(header_names_no_target[xx])


print(selected_features)


"""
['let-7a', 'let-7c', 'let-7e', 'let-7f', 'let-7g', 'miR-1', 'miR-100', 
 'miR-10b', 'miR-125a-5p', 'miR-125b', 'miR-126', 'miR-126*', 'miR-130a', 
 'miR-142-3p', 'miR-142-5p', 'miR-143', 'miR-145', 'miR-145*', 'miR-151-5p', 
 'miR-15b', 'miR-195', 'miR-200b', 'miR-200c', 'miR-203', 'miR-205', 'miR-21',
 'miR-21*', 'miR-223', 'miR-23b', 'miR-26a', 'miR-26b', 'miR-27b', 'miR-30a*',
 'miR-31', 'miR-335', 'miR-365', 'miR-375', 'miR-424', 'miR-450a', 'Candidate-5']
"""

print("--- %s seconds ---" % round(time.time() - start_time,4))



