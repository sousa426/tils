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

mirna = pd.read_excel('./dataset/DataSet_Bioinfo1718.xlsx', 
                              sheet_name='Sheet1',engine='openpyxl')



#print(mirna.head())


#mirna["ID"] = LabelEncoder().fit_transform(mirna["ID"].astype('str'))

mirna=mirna.drop(columns=['ID'])





# Create the entity set
es = ft.EntitySet(id='miRNA Data')
es = es.entity_from_dataframe(entity_id = 'mirna', dataframe = mirna, 
                               make_index  = True,index='index' )

##es['mirna']

#print(es["mirna"].df)



print("1")
feature_matrix,feature_defs = ft.dfs(entityset = es, 
                                      target_entity = 'mirna',
                                      trans_primitives =  ['add_numeric'],
                                      max_depth = 1 ,                                     
                                      #features_only= True,
                                      verbose = True)

print("2")

new_fm,feature_defs2 = ft.selection.remove_highly_null_features(feature_matrix,features = feature_defs )

print("3")

new_fm3, new_features3 = remove_single_value_features(new_fm, features=feature_defs2)

print("4")

cenas = set(feature_defs) - set(new_features3)

print("5")

new_fm4, new_features4 = remove_highly_correlated_features(new_fm3, features=new_features3)
print("6")




cenas = set(feature_defs) - set(new_features4)

print("7")



print(cenas)

print(len(cenas))






print("--- %s seconds ---" % round(time.time() - start_time,4))

