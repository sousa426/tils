import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML
import time

start_time = time.time()

h2o.shutdown()

#https://docs.h2o.ai/h2o/latest-stable/h2o-docs/starting-h2o.html
#https://www.kaggle.com/sudalairajkumar/getting-started-with-h2o


"""
h2o.init()

mito_data = h2o.import_file("./dataset/Mitostress_clean2.csv",header=1)

print(mito_data.columns)


y = "Target"
x = mito_data.columns
x.remove(y)


aml = H2OAutoML(max_models = 10, seed = 1,max_runtime_secs = 30)
aml.train(x = x, y = y, training_frame = mito_data)

lb = aml.leaderboard
print(lb)
print(lb.head())

lb2 = aml.leaderboard.as_data_frame()
print(lb2.head())

m = h2o.get_model(lb[0,"model_id"])
print(m.varimp(use_pandas=True))

m = h2o.get_model(lb[1,"model_id"])
print(m.varimp(use_pandas=True))

m = h2o.get_model(lb[2,"model_id"])
print(m.varimp(use_pandas=True))

m = h2o.get_model(lb[3,"model_id"])
print(m.varimp(use_pandas=True))

"""

print("--- %s seconds ---" % round(time.time() - start_time,4))

