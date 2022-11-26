import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML

import time

start_time = time.time()



#h2o.shutdown()

#https://docs.h2o.ai/h2o/latest-stable/h2o-docs/starting-h2o.html
#https://www.kaggle.com/sudalairajkumar/getting-started-with-h2o

"""
converter pra csv
XGBoost is not available on Windows machines
ERRR on field: _min_rows: The dataset size is too small to split for min_rows=100.0: must have at least 200.0 (weighted) rows, but have only 58.0.

"""





h2o.init(
    nthreads=-1,     # number of threads when launching a new H2O server
    max_mem_size=12  # in gigabytes
    )

mirna = h2o.import_file("./dataset/DataSet_Bioinfo1718.csv",header=1)

print(mirna.columns)


y = "ID"
x = mirna.columns
x.remove(y)

print("Iniciar o  H2OAutoML ")
aml = H2OAutoML(
    max_runtime_secs = 240,
    max_models=None,  # no limit
    )

print("Treinar o   H2OAutoML ")
aml.train(x = x, y = y, training_frame = mirna)

print("1 - leaderboard")
lb = aml.leaderboard
print(lb)
print(lb.head())
aml.leader  

print("2 - aml.leaderboard.as_data_frame")
lb2 = aml.leaderboard.as_data_frame()
print(lb2.head(10))

print("3 - get_model 0")
m = h2o.get_model(lb[0,"model_id"])
print(m.varimp(use_pandas=True).head(25))

print("4 - get_model 1")
m = h2o.get_model(lb[1,"model_id"])
print(m.varimp(use_pandas=True).head(25))

print("5 - get_model 2")
m = h2o.get_model(lb[2,"model_id"])
print(m.varimp(use_pandas=True).head(25))

print("6 - get_model 3")
m = h2o.get_model(lb[3,"model_id"])
print(m.varimp(use_pandas=True).head(25))

print("7 - get_model 4")
m = h2o.get_model(lb[4,"model_id"])
print(m.varimp(use_pandas=True).head(25))

print("--- %s seconds ---" % round(time.time() - start_time,4))




"""

2 - aml.leaderboard.as_data_frame
                                            model_id  ...       mse
0         GBM_grid__1_AutoML_20210125_212836_model_7  ...  0.076519
1        GBM_grid__1_AutoML_20210125_212836_model_27  ...  0.132300
2                       GBM_3_AutoML_20210125_212836  ...  0.079892
3        GBM_grid__1_AutoML_20210125_212836_model_16  ...  0.074871
4        GBM_grid__1_AutoML_20210125_212836_model_13  ...  0.079703
5  DeepLearning_grid__2_AutoML_20210125_212836_mo...  ...  0.094414
6         GBM_grid__1_AutoML_20210125_212836_model_9  ...  0.089435
7  StackedEnsemble_BestOfFamily_AutoML_20210125_2...  ...  0.088724
8                       GBM_2_AutoML_20210125_212836  ...  0.094534
9        GBM_grid__1_AutoML_20210125_212836_model_15  ...  0.078844

[10 rows x 7 columns]
3 - get_model 0
           variable  relative_importance  scaled_importance  percentage
0          miR-133b            14.517756           1.000000    0.200931
1          miR-10b*            11.857270           0.816743    0.164109
2             miR-7             7.333758           0.505158    0.101502
3        miR-142-5p             5.053856           0.348116    0.069947
4          miR-133a             4.884520           0.336451    0.067603
5        miR-486-5p             2.681470           0.184703    0.037112
6           let-7d*             2.669841           0.183902    0.036952
7           miR-340             1.353390           0.093223    0.018731
8        miR-532-3p             1.350000           0.092990    0.018684
9           miR-328             1.297299           0.089360    0.017955
10         miR-497*             1.285748           0.088564    0.017795
11         miR-200b             1.167771           0.080437    0.016162
12          miR-204             1.165623           0.080289    0.016133
13            miR-1             1.149136           0.079154    0.015904
14       miR-369-3p             1.076339           0.074139    0.014897
15          miR-424             1.012676           0.069754    0.014016
16          miR-155             0.975964           0.067226    0.013508
17       miR-140-3p             0.869566           0.059897    0.012035
18  Candidate-12-3p             0.815437           0.056168    0.011286
19      Candidate-5             0.721887           0.049724    0.009991
20          miR-143             0.632337           0.043556    0.008752
21          miR-451             0.590451           0.040671    0.008172
22         miR-18a*             0.584073           0.040232    0.008084
23        miR-200a*             0.580031           0.039953    0.008028
24          miR-212             0.561181           0.038655    0.007767
4 - get_model 1
           variable  relative_importance  scaled_importance  percentage
0          miR-10b*             8.743745           1.000000    0.203063
1          miR-133b             7.396786           0.845952    0.171781
2        miR-542-5p             5.757353           0.658454    0.133707
3          miR-497*             4.858366           0.555639    0.112830
4          miR-133a             3.865485           0.442086    0.089771
5        miR-142-5p             2.552083           0.291875    0.059269
6         miR-200a*             2.508258           0.286863    0.058251
7           miR-429             1.590615           0.181915    0.036940
8             miR-7             1.528473           0.174808    0.035497
9   Candidate-21-3p             1.145337           0.130989    0.026599
10  Candidate-12-3p             1.132014           0.129466    0.026290
11         miR-34b*             0.913468           0.104471    0.021214
12       miR-338-3p             0.525000           0.060043    0.012192
13        miR-200b*             0.451997           0.051694    0.010497
14          miR-30b             0.049959           0.005714    0.001160
15          miR-923             0.015894           0.001818    0.000369
16        miR-193b*             0.010118           0.001157    0.000235
17     Candidate-60             0.009390           0.001074    0.000218
18  Candidate-57-5p             0.004985           0.000570    0.000116
19           let-7a             0.000000           0.000000    0.000000
20          let-7a*             0.000000           0.000000    0.000000
21           let-7b             0.000000           0.000000    0.000000
22          let-7b*             0.000000           0.000000    0.000000
23           let-7c             0.000000           0.000000    0.000000
24          let-7c*             0.000000           0.000000    0.000000
5 - get_model 2
           variable  relative_importance  scaled_importance  percentage
0          miR-10b*            20.252783           1.000000    0.283349
1          miR-133b            19.653162           0.970393    0.274960
2          miR-147b             5.362687           0.264788    0.075027
3             miR-7             4.119483           0.203403    0.057634
4           miR-328             2.538880           0.125360    0.035521
5           miR-100             1.915922           0.094600    0.026805
6         miR-200a*             1.696460           0.083764    0.023735
7   Candidate-21-3p             1.685475           0.083222    0.023581
8          miR-497*             1.597794           0.078893    0.022354
9          miR-125b             1.181242           0.058325    0.016526
10          miR-204             1.127053           0.055649    0.015768
11         miR-133a             0.878707           0.043387    0.012294
12      Candidate-5             0.729634           0.036026    0.010208
13          miR-944             0.703644           0.034743    0.009844
14         miR-146a             0.630601           0.031136    0.008823
15          miR-425             0.555084           0.027408    0.007766
16          miR-155             0.518185           0.025586    0.007250
17  Candidate-12-3p             0.495359           0.024459    0.006930
18           miR-98             0.491669           0.024277    0.006879
19           miR-17             0.484309           0.023913    0.006776
20          miR-31*             0.475859           0.023496    0.006658
21          miR-192             0.462841           0.022853    0.006475
22         miR-183*             0.428937           0.021179    0.006001
23            miR-1             0.284040           0.014025    0.003974
24     Candidate-14             0.268950           0.013280    0.003763
6 - get_model 3
           variable  relative_importance  scaled_importance  percentage
0          miR-133b            15.282455           1.000000    0.210103
1          miR-10b*            10.538750           0.689598    0.144887
2          miR-125b             8.873059           0.580604    0.121987
3             miR-7             8.152581           0.533460    0.112082
4          miR-497*             6.130156           0.401124    0.084277
5           miR-205             6.111355           0.399894    0.084019
6          miR-147b             2.719726           0.177964    0.037391
7           miR-328             2.568985           0.168100    0.035318
8         miR-200a*             2.407430           0.157529    0.033097
9   Candidate-12-3p             2.157163           0.141153    0.029657
10         miR-133a             1.698257           0.111125    0.023348
11          miR-31*             1.626015           0.106398    0.022354
12       miR-139-3p             0.989046           0.064718    0.013597
13         miR-513c             0.822675           0.053831    0.011310
14          miR-451             0.412043           0.026962    0.005665
15          miR-514             0.386534           0.025293    0.005314
16     Candidate-48             0.376042           0.024606    0.005170
17      Candidate-5             0.315233           0.020627    0.004334
18          miR-329             0.296965           0.019432    0.004083
19          miR-192             0.249373           0.016318    0.003428
20        miR-200c*             0.187603           0.012276    0.002579
21          miR-765             0.173337           0.011342    0.002383
22          miR-21*             0.115705           0.007571    0.001591
23         miR-202*             0.071011           0.004647    0.000976
24       miR-142-5p             0.048558           0.003177    0.000668
7 - get_model 4
           variable  relative_importance  scaled_importance  percentage
0          miR-10b*            19.574392           1.000000    0.270488
1          miR-133b            13.857423           0.707936    0.191489
2             miR-7             8.218117           0.419840    0.113562
3          miR-147b             3.686368           0.188326    0.050940
4             miR-1             2.892467           0.147768    0.039970
5          miR-125b             2.577745           0.131690    0.035621
6         miR-200a*             2.551784           0.130363    0.035262
7   Candidate-12-3p             2.345116           0.119805    0.032406
8          miR-133a             2.274963           0.116221    0.031437
9           miR-31*             2.077471           0.106132    0.028707
10          miR-205             1.856504           0.094843    0.025654
11          miR-514             1.841697           0.094087    0.025449
12          miR-328             1.782037           0.091039    0.024625
13         miR-497*             1.585868           0.081017    0.021914
14       miR-139-3p             0.956453           0.048862    0.013217
15         miR-513c             0.752704           0.038453    0.010401
16         miR-202*             0.710305           0.036287    0.009815
17          miR-451             0.570240           0.029132    0.007880
18          miR-329             0.463966           0.023703    0.006411
19     Candidate-48             0.406529           0.020768    0.005618
20          miR-192             0.392457           0.020049    0.005423
21          miR-944             0.287316           0.014678    0.003970
22      Candidate-5             0.279192           0.014263    0.003858
23        miR-200c*             0.157122           0.008027    0.002171
24          miR-21*             0.138103           0.007055    0.001908

"""




