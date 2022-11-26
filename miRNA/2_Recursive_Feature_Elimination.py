import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
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

model = LogisticRegression(max_iter=99999999)
rfe = RFE(model,n_features_to_select=40)
fit = rfe.fit(X, Y)
#print("Num Features: %s" % (fit.n_features_))
#print("Selected Features: %s" % (fit.support_))
#print("Feature Ranking: %s" % (fit.ranking_))


selected_features = np.array(header_names_no_target)[fit.support_]

print(selected_features)



print("--- %s seconds ---" % round(time.time() - start_time,4))




"""

['let-7b' 'let-7e' 'let-7g' 'let-7i' 'miR-1' 'miR-100' 'miR-125a-5p'
 'miR-125b' 'miR-130a' 'miR-142-5p' 'miR-143' 'miR-144' 'miR-151-3p'
 'miR-151-5p' 'miR-17*' 'miR-181a' 'miR-185' 'miR-191' 'miR-192' 'miR-19b'
 'miR-200c' 'miR-205' 'miR-21' 'miR-26a' 'miR-27b' 'miR-28-3p'
 'miR-30c-1*' 'miR-30e*' 'miR-320a' 'miR-335' 'miR-423-5p' 'miR-424'
 'miR-425*' 'miR-451' 'miR-483-5p' 'miR-486-5p' 'miR-92a' 'miR-99a'
 'Candidate-5' 'Candidate-14']
"""