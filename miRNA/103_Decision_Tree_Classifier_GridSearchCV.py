import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline
import time




#https://www.datacamp.com/community/tutorials/decision-tree-classification-python

mirna = pd.read_excel('./dataset/DataSet_Bioinfo1718.xlsx', 
                              sheet_name='Sheet1',engine='openpyxl')


print(mirna.shape)

#print(list(mirna.columns))
header_names = list(mirna.columns)
header_names_no_target = header_names.copy()
header_names_no_target.remove('ID')


#split dataset in features and target variable
feature_cols_chi2 = ['let-7a', 'let-7c', 'let-7e', 'let-7f', 'let-7g', 'miR-1', 'miR-100', 
 'miR-10b', 'miR-125a-5p', 'miR-125b', 'miR-126', 'miR-126*', 'miR-130a', 
 'miR-142-3p', 'miR-142-5p', 'miR-143', 'miR-145', 'miR-145*', 'miR-151-5p', 
 'miR-15b', 'miR-195', 'miR-200b', 'miR-200c', 'miR-203', 'miR-205', 'miR-21',
 'miR-21*', 'miR-223', 'miR-23b', 'miR-26a', 'miR-26b', 'miR-27b', 'miR-30a*',
 'miR-31', 'miR-335', 'miR-365', 'miR-375', 'miR-424', 'miR-450a', 'Candidate-5']

feature_cols_rfe = ['let-7b' ,'let-7e', 'let-7g', 'let-7i' ,'miR-1', 'miR-100' ,'miR-125a-5p',
 'miR-125b', 'miR-130a', 'miR-142-5p', 'miR-143', 'miR-144', 'miR-151-3p',
 'miR-151-5p' ,'miR-17*' ,'miR-181a', 'miR-185', 'miR-191' ,'miR-192' ,'miR-19b',
 'miR-200c' ,'miR-205', 'miR-21', 'miR-26a' ,'miR-27b', 'miR-28-3p',
 'miR-30c-1*', 'miR-30e*' ,'miR-320a', 'miR-335' ,'miR-423-5p', 'miR-424',
 'miR-425*', 'miR-451' ,'miR-483-5p', 'miR-486-5p', 'miR-92a', 'miR-99a',
 'Candidate-5' ,'Candidate-14']

feature_cols_ridge = ['let-7a*', 'let-7b', 'let-7b*', 'let-7c', 'let-7d', 'let-7e', 'let-7f', 
 'let-7g', 'miR-1', 'miR-100', 'miR-101', 'miR-103', 'miR-106b*', 'miR-107',
 'miR-10a', 'miR-10b', 'miR-125a-5p', 'miR-125b', 'miR-126*', 'miR-128',
 'miR-130a', 'miR-130b', 'miR-133a', 'miR-133b', 'miR-135a', 'miR-135b', 
 'miR-136', 'miR-140-5p', 'miR-141', 'miR-142-3p', 'miR-142-5p', 'miR-143*', 
 'miR-144', 'miR-145*', 'miR-146a', 'miR-146b-5p', 'miR-148a', 'miR-150', 
 'miR-151-3p', 'miR-151-5p', 'miR-155', 'miR-15a', 'miR-16', 'miR-16-2*',
 'miR-17', 'miR-17*', 'miR-181a', 'miR-181a*', 'miR-181b', 'miR-181c', 
 'miR-184', 'miR-185', 'miR-188-5p', 'miR-191', 'miR-192', 'miR-193a-3p',
 'miR-193a-5p', 'miR-193b*', 'miR-194', 'miR-195', 'miR-196a', 'miR-196b',
 'miR-197', 'miR-199a-5p', 'miR-199b-3p', 'miR-199b-5p', 'miR-19a', 'miR-19b',
 'miR-200a', 'miR-200a*', 'miR-200b', 'miR-203', 'miR-204', 'miR-205', 'miR-206', 
 'miR-210', 'miR-214', 'miR-215', 'miR-22', 'miR-221', 'miR-222', 'miR-223', 
 'miR-224', 'miR-23a', 'miR-23b', 'miR-24', 'miR-25', 'miR-26a', 'miR-26b',
 'miR-27a', 'miR-27a*', 'miR-27b*', 'miR-28-3p', 'miR-28-5p', 'miR-29a', 
 'miR-29b', 'miR-29c', 'miR-30a', 'miR-30a*', 'miR-30b', 'miR-30d', 'miR-30d*',
 'miR-30e', 'miR-31', 'miR-320a', 'miR-324-5p', 'miR-326', 'miR-335', 
 'miR-338-3p', 'miR-339-3p', 'miR-339-5p', 'miR-342-3p', 'miR-34a', 'miR-34b',
 'miR-34b*', 'miR-34c-3p', 'miR-34c-5p', 'miR-365', 'miR-374a', 'miR-374b',
 'miR-377', 'miR-378', 'miR-378*', 'miR-423-5p', 'miR-424', 'miR-424*', 
 'miR-425', 'miR-425*', 'miR-450a', 'miR-451', 'miR-452', 'miR-455-3p',
 'miR-483-5p', 'miR-484', 'miR-486-5p', 'miR-497', 'miR-503', 'miR-505',
 'miR-574-3p', 'miR-652', 'miR-660', 'miR-663', 'miR-671-5p', 'miR-675', 
 'miR-7', 'miR-708', 'miR-765', 'miR-873', 'miR-876-5p', 'miR-877', 
 'miR-886-5p', 'miR-9', 'miR-923', 'miR-92a-1*', 'miR-92b', 'miR-93',
 'miR-93*', 'miR-940', 'miR-944', 'miR-95', 'miR-98', 'miR-99a', 'miR-99a*',
 'miR-99b', 'Candidate-2', 'Candidate-5', 'Candidate-8', 'Candidate-14',
 'Candidate-15', 'Candidate-21-3p', 'Candidate-21-5p', 'Candidate-29-5p', 
 'Candidate-36-3p', 'Candidate-51-5p']

feature_cols_lasso = ['let-7b', 'let-7c', 'let-7d', 'let-7e', 'let-7f', 'let-7g', 'let-7i', 
 'miR-1', 'miR-100', 'miR-106b*', 'miR-10b', 'miR-125a-5p', 'miR-125b',
 'miR-126', 'miR-126*', 'miR-130a', 'miR-141', 'miR-151-5p', 'miR-155',
 'miR-15a', 'miR-181a', 'miR-195', 'miR-196a', 'miR-196b', 'miR-200b', 
 'miR-200c', 'miR-203', 'miR-205', 'miR-206', 'miR-21', 'miR-21*', 'miR-22', 
 'miR-223', 'miR-23a', 'miR-23b', 'miR-24', 'miR-26a', 'miR-26b', 'miR-27a',
 'miR-28-3p', 'miR-29a', 'miR-29b', 'miR-29c', 'miR-30a', 'miR-30c', 'miR-30d', 
 'miR-31', 'miR-335', 'miR-365', 'miR-374a', 'miR-375', 'miR-378', 'miR-424',
 'miR-450a', 'miR-451', 'miR-484', 'miR-486-5p', 'miR-7', 'miR-768-5p', 
 'miR-92a', 'Candidate-5']

feature_cols_infogain = ['miR-133a', 'miR-194', 'miR-324-5p', 'miR-338-3p', 'miR-497', 'miR-509-3p',
 'miR-524-5p', 'miR-944', 'Candidate-54-3-3p']

feature_cols_autofeat =['miR-133a', 'miR-513c', 'miR-328', 'miR-765', 
                        'miR-374b', 'miR-199b-3p']

feature_cols_tpot = ['let-7a*', 'let-7b', 'let-7b*', 'let-7c', 'let-7c*', 
                     'let-7d', 'let-7d*', 'let-7e', 'let-7e*', 'let-7f', 
                     'let-7f-1*', 'let-7f-2*', 'let-7g', 'let-7g*', 
                     'let-7i', 'let-7i*', 'miR-1', 'miR-100', 'miR-100*', 
                     'miR-101', 'miR-101*', 'miR-103', 'miR-105', 'miR-105*', 
                     'miR-106a', 'miR-106a*', 'miR-106b', 'miR-106b*', 'miR-107',
                     'miR-10a', 'miR-10a*', 'miR-10b', 'miR-10b*', 'miR-122',
                     'miR-122*', 'miR-1224-3p', 'miR-1224-5p', 'miR-1225-3p',
                     'miR-1225-5p', 'miR-1226', 'miR-1227', 'miR-1228', 
                     'miR-1228*', 'miR-1229', 'miR-1233', 'miR-1236', 
                     'miR-124', 'miR-125a-5p', 'miR-125b', 'miR-125b-1*',
                     'miR-125b-2*', 'miR-126', 'miR-126*', 'miR-1277', 
                     'miR-129-3p', 'miR-130b', 'miR-130b*', 'miR-133a', 
                     'miR-133b', 'miR-134', 'miR-135a', 'miR-135b', 
                     'miR-135b*', 'miR-137', 'miR-139-3p', 'miR-139-5p', 
                     'miR-140-3p', 'miR-140-5p', 'miR-141', 'miR-141*', 
                     'miR-142-3p', 'miR-142-5p', 'miR-143', 'miR-144', 
                     'miR-145*', 'miR-146a', 'miR-147b', 'miR-148b', 
                     'miR-149', 'miR-150*', 'miR-151-3p', 'miR-151-5p', 
                     'miR-155', 'miR-17', 'miR-17*', 'miR-181b', 'miR-181c',
                     'miR-182', 'miR-183*', 'miR-184', 'miR-185', 'miR-185*',
                     'miR-186', 'miR-187', 'miR-188-3p', 'miR-18a', 'miR-190',
                     'miR-191*', 'miR-192', 'miR-193a-3p', 'miR-193a-5p', 
                     'miR-193b', 'miR-193b*', 'miR-194', 'miR-194*', 
                     'miR-195*', 'miR-196a*', 'miR-196b', 'miR-197',
                     'miR-199a-5p', 'miR-199b-3p', 'miR-19a*', 'miR-19b-1*', 
                     'miR-200a', 'miR-200a*', 'miR-200c', 'miR-200c*', 
                     'miR-202*', 'miR-203', 'miR-204', 'miR-205', 'miR-20a',
                     'miR-20b', 'miR-20b*', 'miR-21', 'miR-21*', 'miR-210', 
                     'miR-211', 'miR-212', 'miR-214', 'miR-215', 'miR-218', 
                     'miR-218-1*', 'miR-219-1-3p', 'miR-22*', 'miR-221', 
                     'miR-221*', 'miR-222*', 'miR-23a*', 'miR-23b', 'miR-24',
                     'miR-24-1*', 'miR-24-2*', 'miR-25*', 'miR-26a', 'miR-26a-2*', 
                     'miR-26b', 'miR-27a', 'miR-27a*', 'miR-27b', 'miR-27b*',
                     'miR-28-3p', 'miR-28-5p', 'miR-296-3p', 'miR-299-3p',
                     'miR-299-5p', 'miR-29a', 'miR-29a*', 'miR-29b', 'miR-29b-1*', 
                     'miR-29b-2*', 'miR-29c', 'miR-29c*', 'miR-301a', 'miR-30a',
                     'miR-30b*', 'miR-30c-1*', 'miR-30c-2*', 'miR-30d', 'miR-31',
                     'miR-31*', 'miR-324-3p', 'miR-326', 'miR-328', 'miR-329',
                     'miR-330-3p', 'miR-330-5p', 'miR-331-3p', 'miR-331-5p', 
                     'miR-335', 'miR-337-5p', 'miR-338-3p', 'miR-338-5p', 'miR-339-5p',
                     'miR-33a', 'miR-33b', 'miR-33b*', 'miR-340', 'miR-342-3p',
                     'miR-345', 'miR-34b', 'miR-34b*', 'miR-361-3p', 'miR-363',
                     'miR-365', 'miR-374b', 'miR-375', 'miR-376a*', 'miR-377', 
                     'miR-377*', 'miR-378', 'miR-378*', 'miR-379', 'miR-379*', 
                     'miR-383', 'miR-409-3p', 'miR-410', 'miR-421', 'miR-423-3p',
                     'miR-423-5p', 'miR-424', 'miR-424*', 'miR-425', 'miR-425*', 
                     'miR-432', 'miR-449a', 'miR-449b', 'miR-450a', 'miR-450b-5p',
                     'miR-451', 'miR-455-3p', 'miR-455-5p', 'miR-483-3p', 'miR-483-5p',
                     'miR-484', 'miR-486-3p', 'miR-486-5p', 'miR-487b', 'miR-491-3p',
                     'miR-491-5p', 'miR-493', 'miR-494', 'miR-495', 'miR-497', 
                     'miR-497*', 'miR-501-3p', 'miR-502-3p', 'miR-502-5p', 'miR-503',
                     'miR-505', 'miR-506', 'miR-507', 'miR-508-3p', 'miR-509-3p', 
                     'miR-513c', 'miR-514', 'miR-515-5p', 'miR-519c-3p', 'miR-524-5p', 
                     'miR-532-5p', 'miR-539', 'miR-542-3p', 'miR-542-5p', 'miR-545*',
                     'miR-548b-5p', 'miR-548d-5p', 'miR-549-3p', 'miR-551b', 'miR-556-5p',
                     'miR-570', 'miR-574-3p', 'miR-576-3p', 'miR-576-5p', 'miR-589',
                     'miR-590-5p', 'miR-598', 'miR-610', 'miR-615-5p', 'miR-618',
                     'miR-628-3p', 'miR-629*', 'miR-639', 'miR-642', 'miR-651',
                     'miR-652', 'miR-653', 'miR-655', 'miR-658', 'miR-660', 'miR-671-5p', 
                     'miR-7', 'miR-708', 'miR-708*', 'miR-744', 'miR-760',
                     'miR-765', 'miR-766', 'miR-768-5p', 'miR-770-5p', 'miR-873', 
                     'miR-876-3p', 'miR-876-5p', 'miR-877', 'miR-885-5p',
                     'miR-886-5p', 'miR-888', 'miR-889', 'miR-9', 'miR-92a', 
                     'miR-92a-1*', 'miR-92b', 'miR-93', 'miR-93*', 'miR-934', 
                     'miR-940', 'miR-944', 'miR-96*', 'miR-98', 'miR-99a', 'miR-99b',
                     'miR-99b*', 'Candidate-1-5p', 'Candidate-2', 'Candidate-3',
                     'Candidate-4', 'Candidate-5', 'Candidate-6', 'Candidate-7-3p', 
                     'Candidate-11', 'Candidate-12-3p', 'Candidate-12-5p', 'Candidate-14',
                     'Candidate-15', 'Candidate-16', 'Candidate-17', 'Candidate-18-5p',
                     'Candidate-20', 'Candidate-21-3p', 'Candidate-22', 'Candidate-23-3p', 
                     'Candidate-24', 'Candidate-25-5p', 'Candidate-27-5p', 'Candidate-28', 
                     'Candidate-29-5p', 'Candidate-29-3p', 'Candidate-36-5p', 
                     'Candidate-37-5p', 'Candidate-40-3p', 'Candidate-40-5p',
                     'Candidate-42', 'Candidate-44', 'Candidate-45', 'Candidate-46',
                     'Candidate-47', 'Candidate-48', 'Candidate-50-3p', 
                     'Candidate-51-5p', 'Candidate-52-2', 'Candidate-53',
                     'Candidate-54-3-3p', 'Candidate-55', 'Candidate-57-5p', 
                     'Candidate-57-3p', 'Candidate-58-3p', 'Candidate-59-3p', 
                     'Candidate-59-5p']

feature_cols_h2o = ['miR-133b','miR-10b*','miR-7','miR-142-5p','miR-133a',
'miR-486-5p','let-7d*','miR-340','miR-532-3p','miR-328','miR-497*','miR-200b',
'miR-204','miR-1','miR-369-3p','miR-424','miR-155','miR-140-3p','Candidate-12-3p',
'Candidate-5','miR-143','miR-451','miR-18a*','miR-200a*','miR-212']

feature_cols_featuretools = ['let-7a', 'let-7a*', 'let-7b', 'let-7b*', 'let-7c', 'let-7c*', 'let-7d', 'let-7f', 'let-7g*', 'let-7i', 'miR-1',
 'miR-100', 'miR-100*', 'miR-101*', 'miR-103', 'miR-105', 'miR-105*', 'miR-106a', 'miR-106a*', 'miR-106b', 
 'miR-106b*', 'miR-107', 'miR-10a*', 'miR-10b', 'miR-10b*', 'miR-122', 'miR-122*', 'miR-1224-3p', 'miR-1224-5p', 
 'miR-1225-3p', 'miR-1225-5p', 'miR-1226', 'miR-1227', 'miR-1228', 'miR-1228*', 'miR-1229', 'miR-1233', 'miR-1236', 
 'miR-1237', 'miR-125a-3p', 'miR-125b-1*', 'miR-125b-2*', 'miR-1271', 'miR-128', 'miR-129-3p', 'miR-129-5p',
 'miR-129*', 'miR-130a', 'miR-130a*', 'miR-130b', 'miR-130b*', 'miR-133a', 'miR-134', 'miR-135a', 'miR-135a*',
 'miR-135b', 'miR-135b*', 'miR-136*', 'miR-137', 'miR-138', 'miR-138-1*', 'miR-139-3p', 'miR-141', 'miR-141*', 
 'miR-142-3p', 'miR-142-5p', 'miR-143*', 'miR-144', 'miR-144*', 'miR-145', 'miR-146a', 'miR-146a*', 'miR-146b-3p',
 'miR-148a*', 'miR-148b*', 'miR-149*', 'miR-150', 'miR-150*', 'miR-153', 'miR-154*', 'miR-155', 'miR-155*', 'miR-15b',
 'miR-17', 'miR-17*', 'miR-181a', 'miR-181a*', 'miR-181a-2*', 'miR-181c', 'miR-181c*', 'miR-182', 'miR-182*', 
 'miR-183', 'miR-183*', 'miR-184', 'miR-185*', 'miR-186*', 'miR-187*', 'miR-188-3p', 'miR-18b', 'miR-18b*',
 'miR-190', 'miR-191', 'miR-191*', 'miR-192', 'miR-192*', 'miR-193a-3p', 'miR-193b', 'miR-193b*', 'miR-195*',
 'miR-196a', 'miR-196a*', 'miR-197', 'miR-199b-3p', 'miR-19a*', 'miR-19b-1*', 'miR-19b-2*', 'miR-200c',
 'miR-200c*', 'miR-202', 'miR-202*', 'miR-204', 'miR-20b', 'miR-20b*', 'miR-21*', 'miR-211', 'miR-214', 
 'miR-215', 'miR-218-1*', 'miR-218-2*', 'miR-219-1-3p', 'miR-219-5p', 'miR-22', 'miR-22*', 'miR-221*',
 'miR-222*', 'miR-23a', 'miR-23a*', 'miR-24-1*', 'miR-24-2*', 'miR-27a*', 'miR-27b*', 'miR-296-3p', 'miR-299-3p',
 'miR-29b', 'miR-29c*', 'miR-302a*', 'miR-302b*', 'miR-302d', 'miR-30a', 'miR-30a*', 'miR-30b*', 'miR-30c-1*', 
 'miR-30c-2*', 'miR-30d*', 'miR-31', 'miR-320a', 'miR-323-3p', 'miR-324-3p', 'miR-328', 'miR-329', 'miR-335', 
 'miR-335*', 'miR-337-5p', 'miR-338-3p', 'miR-339-3p', 'miR-33a', 'miR-33b', 'miR-33b*', 'miR-340*', 'miR-342-3p', 
 'miR-342-5p', 'miR-345', 'miR-346', 'miR-362-3p', 'miR-363', 'miR-365', 'miR-367', 'miR-369-5p', 'miR-370',
 'miR-371-3p', 'miR-371-5p', 'miR-372', 'miR-373', 'miR-374b', 'miR-374b*', 'miR-376a', 'miR-376a*', 'miR-376b',
 'miR-377*', 'miR-378', 'miR-380*', 'miR-383', 'miR-409-3p', 'miR-409-5p', 'miR-423-3p', 'miR-424', 'miR-431',
 'miR-431*', 'miR-432*', 'miR-450b-3p', 'miR-451', 'miR-452*', 'miR-454*', 'miR-455-3p', 'miR-483-3p', 'miR-483-5p',
 'miR-485-3p', 'miR-485-5p', 'miR-486-3p', 'miR-486-5p', 'miR-487a', 'miR-488', 'miR-488*', 'miR-489', 'miR-490-3p', 
 'miR-490-5p', 'miR-491-3p', 'miR-491-5p', 'miR-495', 'miR-498', 'miR-499-3p', 'miR-501-5p', 'miR-502-3p', 'miR-504',
 'miR-505', 'miR-505*', 'miR-506', 'miR-507', 'miR-508-3p', 'miR-508-5p', 'miR-509-3p', 'miR-509-3-5p', 'miR-510',
 'miR-511', 'miR-512-3p', 'miR-512-5p', 'miR-513a-3p', 'miR-513a-5p', 'miR-513b', 'miR-513c', 'miR-516a-5p', 
 'miR-516b', 'miR-517*', 'miR-517b', 'miR-517c', 'miR-518a-3p', 'miR-518b', 'miR-518c', 'miR-518d-3p', 'miR-518e',
 'miR-518f*', 'miR-519b-3p', 'miR-519c-3p', 'miR-519e*', 'miR-520a-5p', 'miR-520c-3p', 'miR-524-5p', 'miR-525-5p',
 'miR-526b', 'miR-532-3p', 'miR-539', 'miR-541', 'miR-543', 'miR-544', 'miR-548a-5p', 'miR-548b-3p', 'miR-548b-5p',
 'miR-548d-5p', 'miR-549-5p', 'miR-550', 'miR-550*', 'miR-551a', 'miR-551b', 'miR-551b*', 'miR-556-3p', 'miR-556-5p',
 'miR-559', 'miR-570', 'miR-577', 'miR-580', 'miR-581', 'miR-582-3p', 'miR-582-5p', 'miR-584', 'miR-585', 'miR-589',
 'miR-589*', 'miR-597', 'miR-599', 'miR-600', 'miR-603', 'miR-605', 'miR-610', 'miR-615-3p', 'miR-615-5p', 'miR-616',
 'miR-617', 'miR-618', 'miR-624', 'miR-624*', 'miR-625', 'miR-625*', 'miR-628-3p', 'miR-628-5p', 'miR-629', 
 'miR-629*', 'miR-636', 'miR-641', 'miR-642', 'miR-643', 'miR-653', 'miR-654-5p', 'miR-658', 'miR-659', 'miR-665',
 'miR-668', 'miR-671-3p', 'miR-671-5p', 'miR-675', 'miR-7-2*', 'miR-760', 'miR-765', 'miR-767-5p', 'miR-768-3p',
 'miR-770-5p', 'miR-873', 'miR-874', 'miR-875-5p', 'miR-876-3p', 'miR-876-5p', 'miR-877*', 'miR-888', 'miR-890',
 'miR-892b', 'miR-923', 'miR-933', 'miR-934', 'miR-935', 'miR-938', 'miR-939', 'miR-99a', 'Candidate-1-3p',
 'Candidate-1-5p', 'Candidate-3', 'Candidate-4', 'Candidate-6', 'Candidate-7-5p', 'Candidate-8', 'Candidate-9-3p', 
 'Candidate-9-5p', 'Candidate-10', 'Candidate-11', 'Candidate-12-5p', 'Candidate-14', 'Candidate-15', 'Candidate-16',
 'Candidate-18-3p', 'Candidate-19', 'Candidate-20', 'Candidate-23-3p', 'Candidate-24', 'Candidate-25-3p', 
 'Candidate-26-3p', 'Candidate-26-5p', 'Candidate-27-3p', 'Candidate-28', 'Candidate-29-3p', 'Candidate-31', 
 'Candidate-33', 'Candidate-34-3p', 'Candidate-34-5p', 'Candidate-35', 'Candidate-36-5p', 'Candidate-37-5p',
 'Candidate-37-3p', 'Candidate-38', 'Candidate-39', 'Candidate-41', 'Candidate-42', 'Candidate-43', 'Candidate-44',
 'Candidate-45', 'Candidate-47', 'Candidate-49', 'Candidate-50-1-5p', 'Candidate-50-2-5p', 'Candidate-52-1',
 'Candidate-52-2', 'Candidate-53', 'Candidate-54', 'Candidate-55', 'Candidate-56', 'Candidate-57-5p', 
 'Candidate-58-3p', 'Candidate-58-5p', 'Candidate-60', 'Candidate-61', 'Candidate-62', 'Candidate-63', 
 'Candidate-64']

features_by_rank_4 = ['miR-1', 'miR-424', 'miR-100', 'miR-142-5p', 'miR-335', 'Candidate-5', 
                      'miR-451', 'miR-486-5p', 'miR-133a', 'let-7c', 'let-7e', 'let-7f', 
                      'let-7g', 'miR-10b', 'miR-125a-5p', 'miR-125b', 'miR-130a', 'miR-151-5p',
                      'miR-200c', 'miR-205', 'miR-26a', 'miR-31', 'miR-365', 'let-7b', 
                      'miR-155', 'miR-126*', 'miR-142-3p', 'miR-143', 'miR-200b', 'miR-203',
                      'miR-21', 'miR-21*', 'miR-23b', 'miR-26b', 'miR-450a', 'let-7i', 
                      'miR-144', 'miR-17*', 'miR-181a', 'miR-192', 'miR-28-3p', 'miR-483-5p',
                      'miR-99a', 'Candidate-14', 'let-7d', 'miR-106b*', 'miR-141', 'miR-199b-3p',
                      'miR-204', 'miR-29b', 'miR-30a', 'miR-338-3p', 'miR-374b', 'miR-378', 
                      'miR-7', 'miR-765', 'miR-328']

features_by_rank_3 = ['miR-1', 'miR-424', 'miR-100', 'miR-142-5p', 'miR-335', 
                      'Candidate-5', 'miR-451', 'miR-486-5p', 'miR-133a', 'let-7c', 
                      'let-7e', 'let-7f', 'let-7g', 'miR-10b', 'miR-125a-5p',
                      'miR-125b', 'miR-130a', 'miR-151-5p', 'miR-200c', 'miR-205',
                      'miR-26a', 'miR-31', 'miR-365', 'let-7b', 'miR-155', 'miR-126*',
                      'miR-142-3p', 'miR-143', 'miR-200b', 'miR-203', 'miR-21',
                      'miR-21*', 'miR-23b', 'miR-26b', 'miR-450a', 'let-7i',
                      'miR-144', 'miR-17*', 'miR-181a', 'miR-192', 'miR-28-3p', 
                      'miR-483-5p', 'miR-99a', 'Candidate-14', 'let-7d', 'miR-106b*',
                      'miR-141', 'miR-199b-3p', 'miR-204', 'miR-29b', 'miR-30a', 
                      'miR-338-3p', 'miR-374b', 'miR-378', 'miR-7', 'miR-765', 'miR-328',
                      'miR-126', 'miR-145*', 'miR-195', 'miR-223', 'miR-27b', 'miR-30a*',
                      'miR-375', 'miR-151-3p', 'miR-185', 'miR-191', 'miR-30c-1*',
                      'miR-320a', 'miR-423-5p', 'miR-425*', 'miR-92a', 'let-7a*', 
                      'let-7b*', 'miR-103', 'miR-107', 'miR-130b', 'miR-133b', 
                      'miR-135a', 'miR-135b', 'miR-146a', 'miR-17', 'miR-181c', 'miR-184',
                      'miR-193a-3p', 'miR-193b*', 'miR-194', 'miR-196a', 'miR-196b',
                      'miR-197', 'miR-200a*', 'miR-214', 'miR-215', 'miR-22', 'miR-23a',
                      'miR-24', 'miR-27a', 'miR-27a*', 'miR-27b*', 'miR-29a', 'miR-29c', 
                      'miR-30d', 'miR-342-3p', 'miR-455-3p', 'miR-484', 'miR-497', 'miR-505',
                      'miR-671-5p', 'miR-873', 'miR-876-5p', 'miR-944', 'Candidate-15',
                      'miR-509-3p', 'miR-524-5p', 'miR-513c', 'miR-10b*']




#2 piores
#X = mirna[feature_cols_lasso] # Features
X = mirna[feature_cols_infogain] # Features

#2 melhores
#X = mirna[feature_cols_h2o] # Features
#X = mirna[feature_cols_ridge] # Features


y = mirna.ID # Target variable

#print(len(features_by_rank_3))


param_grid = {
    'max_depth': [5,10,20,40,50,100],
    'min_samples_split': [2,3,4,5] 
}

model = GridSearchCV(estimator=DecisionTreeClassifier(),
                     param_grid=param_grid,
                     cv=LeaveOneOut(),
                     n_jobs=-1)

model.fit(X, y)
print(model.best_score_)
print(model.best_estimator_)

start_time = time.time()


#feature_cols_h2o
#0.96
#DecisionTreeClassifier(max_depth=20, min_samples_split=3)

#feature_cols_ridge
#0.9655172413793104
#DecisionTreeClassifier(max_depth=10, min_samples_split=5)

#feature_cols_lasso
#0.7068965517241379
#DecisionTreeClassifier(max_depth=20, min_samples_split=3)

#feature_cols_infogain
#0.7586206896551724
#DecisionTreeClassifier(max_depth=20)



#best_model = model.best_estimator_
best_model = DecisionTreeClassifier(max_depth=20)

predictions = cross_val_predict(best_model, X, y, cv=LeaveOneOut())
    
print("accuracy: " , metrics.accuracy_score(y, predictions))
print("precision: " , metrics.precision_score(y, predictions,average='macro'))

print(classification_report(y, predictions))


print("--- %s seconds ---" % round(time.time() - start_time,4))