import pandas as pd
import featuretools as ft
from sklearn.preprocessing import LabelEncoder
from featuretools.selection import (
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features,
)
import warnings
import time

start_time = time.time()

warnings.simplefilter('ignore')


#https://featuretools.alteryx.com/en/stable/guides/feature_selection.html
#https://www.analyticsvidhya.com/blog/2018/08/guide-automated-feature-engineering-featuretools-python/
#https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics


mito_data = pd.read_excel('../dataset/Datasets_ALS.xlsx', 
                              sheet_name='ATP Rate clean',engine='openpyxl')

mito_data.drop(['ID', 'Family','N','Date','Description'], axis=1, 
               inplace=True)

#print(mito_data.head())

mito_data["Target"] = LabelEncoder().fit_transform(
    mito_data["Target"].astype('str'))

mito_data=mito_data.drop(columns=['Target'])


# Create the entity set
es = ft.EntitySet(id='Mito Data')
es = es.entity_from_dataframe(entity_id = 'mito_data', dataframe = mito_data, 
                               index = 'ID')

#print(es['mito_data'])


fm, features  = ft.dfs(entityset=es,
                                       target_entity = 'mito_data',
                                       trans_primitives=['add_numeric'],
                                       #features_only= True,
                                      # max_depth = 6,
                                       verbose= True)

print("2")

new_fm,feature_defs2 = ft.selection.remove_highly_null_features(fm,features = features )

print("3")

new_fm3, new_features3 = remove_single_value_features(new_fm, features=feature_defs2)

print("4")

cenas = set(features) - set(new_features3)

print("5")

new_fm4, new_features4 = remove_highly_correlated_features(new_fm3, features=new_features3)
print("6")




cenas = set(features) - set(new_features4)

print("7")



print(cenas)

print(len(cenas))

print("--- %s seconds ---" % round(time.time() - start_time,4))

