import pandas as pd
from sklearn.preprocessing import LabelEncoder
from autofeat import AutoFeatClassifier 
from autofeat import FeatureSelector
import time
from sklearn.model_selection import train_test_split

start_time = time.time()

mito_data = pd.read_excel('../dataset/Datasets_ALS.xlsx', 
                              sheet_name='ATP Rate clean',engine='openpyxl')

mito_data.drop(['ID', 'Family','N','Date','Description'], axis=1, 
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

X = X.astype('float32')
Y = LabelEncoder().fit_transform(Y.astype('str'))

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=.3,random_state =0)

model = AutoFeatClassifier(verbose=1)

df = model.fit_transform(X, Y)
y_pred = model.predict(X_test)

print("Final Accuracy: %.4f" % model.score(df, Y))
print("autofeat new features:", len(model.new_feat_cols_))

print(model.new_feat_cols_)
print(model.original_columns_)


print("--- %s seconds ---" % round(time.time() - start_time,4))
