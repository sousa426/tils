import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
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


model = LogisticRegression(max_iter=5000)
rfe = RFE(model)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

"""
'Non mitochondrial respiration' False
 'Basal respiration ' False
 'Maximal respiration' False
 'Proton leak' True
 'ATP production'False
 'Spare respiratory capacity' True
 'Baseline OCR'True
 'Baseline ECAR'True
 'Stressed OCR'False
 'Stressed ECAR'True              
"""

print("--- %s seconds ---" % round(time.time() - start_time,4))