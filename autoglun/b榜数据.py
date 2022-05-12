import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
                                         importance  ...   p99_low
CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL          0.139000  ...  0.114023
CUR_MON_COR_DPS_MON_DAY_AVG_BAL            0.091000  ...  0.042042
CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR         0.026000  ... -0.026517
LAST_12_MON_COR_DPS_DAY_AVG_BAL            0.017000  ... -0.017855
ICO_CUR_MON_ACM_TRX_AMT                    0.011000  ... -0.006190
MON_12_AGV_TRX_CNT                         0.004667  ...  0.001358
CUR_YEAR_MID_BUS_INC                       0.004333  ... -0.004420
CUR_YEAR_MON_AGV_TRX_CNT                   0.004000  ... -0.011160
NB_RCT_3_MON_LGN_TMS_AGV                   0.003333  ... -0.011087
MON_12_ACM_ENTR_ACT_CNT                    0.003000  ... -0.016850
REG_DT                                     0.003000  ... -0.006925
LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL    0.002667  ... -0.013875
REG_CPT                                    0.002667  ... -0.000642
MON_12_TRX_AMT_MAX_AMT_PCTT                0.002667  ... -0.010566
ICO_CUR_MON_ACM_TRX_TM                     0.002667  ... -0.009262
MON_6_50_UP_ENTR_ACT_CNT                   0.002333  ... -0.012087
LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV     0.002333  ... -0.000975
PUB_TO_PRV_TRX_AMT_CUR_YEAR                0.002333  ... -0.010900
OPN_TM                                     0.002333  ... -0.009595
MON_12_AGV_ENTR_ACT_CNT                    0.002000  ... -0.003730
EMP_NBR                                    0.001667  ... -0.007086
MON_6_50_UP_LVE_ACT_CNT                    0.001667  ... -0.010262
AGN_CNT_RCT_12_MON                         0.001667  ... -0.004950
MON_12_EXT_SAM_TRSF_OUT_AMT                0.001667  ... -0.007086
COUNTER_CUR_YEAR_CNT_AMT                   0.001667  ... -0.007086
MON_12_ACM_LVE_ACT_CNT                     0.001333  ... -0.011900
AGN_CUR_YEAR_WAG_AMT                       0.001333  ... -0.005283
MON_12_ACT_OUT_50_UP_CNT_PTY_QTY           0.001333  ... -0.007420
LAST_12_MON_MON_AVG_TRX_AMT_NAV            0.001333  ... -0.016172

"""
imp_name = ["CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL",
            "CUR_MON_COR_DPS_MON_DAY_AVG_BAL",
            "CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR",
            "LAST_12_MON_COR_DPS_DAY_AVG_BAL",
            "ICO_CUR_MON_ACM_TRX_AMT",
            "MON_12_AGV_TRX_CNT",
            "CUR_YEAR_MID_BUS_INC ",
            "CUR_YEAR_MON_AGV_TRX_CNT",
            "NB_RCT_3_MON_LGN_TMS_AGV",
            "MON_12_ACM_ENTR_ACT_CNT",
            "REG_DT ",
            "LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL"   ,
            "REG_CPT",
            "MON_12_TRX_AMT_MAX_AMT_PCTT",
            "ICO_CUR_MON_ACM_TRX_TM",
            "MON_6_50_UP_ENTR_ACT_CNT",
            "LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV",
            "PUB_TO_PRV_TRX_AMT_CUR_YEAR",
            "OPN_TM",
            "MON_12_AGV_ENTR_ACT_CNT"
            ]


"""
train 数据集清洗 ？ 数据
"""
data = pd.read_csv('../data/zhaoshang/train.csv')
print(data.shape)
print(data.head())
print(data.dtypes)
colname = data.columns[2:]
strname = ['MON_12_CUST_CNT_PTY_ID',
           'AI_STAR_SCO',
           'WTHR_OPN_ONL_ICO',
           'SHH_BCK',
           'LGP_HLD_CARD_LVL',
           'NB_CTC_HLD_IDV_AIO_CARD_SITU']
for c in colname:
    if c not in strname:
        data[c] = data[c].replace(r'[?]', np.nan, regex=True).astype(float)
    else:
        data[c] = data[c].replace(r'[?]', np.nan, regex=True)

"""
test 数据集清洗 ？ 数据
"""
testdataB = pd.read_csv('test_B.csv')
print(testdataB.dtypes)
testcolname = testdataB.columns[1:]

for c in testcolname:
    if c not in strname:
        testdataB[c] = testdataB[c].replace(r'[?]', np.nan, regex=True).astype(float)
    else:
        testdataB[c] = testdataB[c].replace(r'[?]', np.nan, regex=True)

testdataB.to_csv('testB_data.csv',index=False)




import xgboost
import lightgbm
import sklearn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pylab

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('testB_data.csv')
testA_id = test_data["CUST_UID"]


train_data['AI_STAR_SCO'] = train_data['AI_STAR_SCO'].astype(object)
test_data['AI_STAR_SCO'] = test_data['AI_STAR_SCO'].astype(object)

print(train_data.dtypes)
print(test_data.dtypes)
strname = ['MON_12_CUST_CNT_PTY_ID',
           'AI_STAR_SCO',
           'WTHR_OPN_ONL_ICO',
           'SHH_BCK',
           'LGP_HLD_CARD_LVL',
           'NB_CTC_HLD_IDV_AIO_CARD_SITU']
# describe

# 把所有nan转变成2
for c in train_data.columns:
    if train_data[c].dtype == 'float64':
        train_data[c] = train_data[c].fillna(2.0)

for c in test_data.columns:
    if test_data[c].dtype == 'float64':
        test_data = test_data.fillna(2.0)


train_describe = train_data.describe()
test_describe = test_data.describe()

null_test_sum = test_data.isnull().sum()
dropname = test_data.columns[null_test_sum > len(test_data) * 0.5]
len(test_data.columns[null_test_sum < len(test_data) * 0.5])
dropname = dropname.drop('MON_12_CUST_CNT_PTY_ID')


# 对大数据取log 获得大数据的值
str_log = []
train_log = train_describe[3:4]
for c in train_log.columns:
    if train_log[c][0] < 0:
        str_log.append(c)

# str_log 有3个含有负数
# ['MON_12_EXT_SAM_AMT', 'CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR', 'CUR_YEAR_MID_BUS_INC']

str_mean = []
train_mean = train_describe[1:2]
for c in train_mean.columns:
    if train_mean[c][0] > 10:
        str_mean.append(c)
# str_mean 有41个


for d in str_mean:
    if d not in str_log:
        train_data[d] = np.log(train_data[d] + 1)
        test_data[d] = np.log(test_data[d] + 1)
    if d in str_log:
        for i in range(40000):
            if train_data.loc[i, d] < 0:
                train_data.loc[i, d] = -np.log(-train_data.loc[i, d] + 1)
            if train_data.loc[i, d] > 0:
                train_data.loc[i, d] = np.log(train_data.loc[i, d] + 1)
        for i in range(12000):
            if test_data.loc[i, d] < 0:
                test_data.loc[i, d] = -np.log(-test_data.loc[i, d] + 1)
            if test_data.loc[i, d] > 0:
                test_data.loc[i, d] = np.log(test_data.loc[i, d] + 1)


train_describe_2 = train_data.describe()
test_describe_2 = test_data.describe()



train_75 = train_describe[6:7]
str_2_name = []
for c in train_75.columns:
    if train_75[c][0] == 2:
        str_2_name.append(c)

# str_2_name
# ['COUNTER_CUR_YEAR_CNT_AMT', 'MON_12_EXT_SAM_TRSF_IN_AMT', 'MON_12_EXT_SAM_TRSF_OUT_AMT',
# 'MON_12_EXT_SAM_NM_TRSF_OUT_CNT', 'MON_12_EXT_SAM_AMT', 'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT',
# 'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT', 'CUR_YEAR_COUNTER_ENCASH_CNT', 'HLD_FGN_CCY_ACT_NBR']


train_data['AI_STAR_SCO'] = train_data['AI_STAR_SCO'].astype(object)
test_data['AI_STAR_SCO'] = test_data['AI_STAR_SCO'].astype(object)



# 去掉了一部分缺失值过多的float类
null_sum = train_data.isnull().sum()
dropname = train_data.columns[null_sum > len(train_data) * 0.5]
len(train_data.columns[null_sum < len(train_data) * 0.5])
dropname = dropname.drop('MON_12_CUST_CNT_PTY_ID')



colname = train_data.columns

#洗掉str2特征
for c in  str_2_name:
    colname = colname.drop(c)

colname = colname.drop(dropname)

# 训练集洗去drop特征
train_data = train_data[colname]


colnametest = test_data.columns

#洗掉dropname
colnametest = colnametest.drop(dropname)

#洗掉str2特征
for c in  str_2_name:
    colnametest = colnametest.drop(c)

test_data = test_data[colnametest]



# 去掉标签 构建训练特征
all_features = pd.get_dummies(train_data.iloc[:,1:], dummy_na=True)
all_features.dtypes
test_features = pd.get_dummies(test_data.iloc[:,1:], dummy_na=True)
test_features.dtypes


from autogluon.tabular import TabularDataset, TabularPredictor
train_data_auto = TabularDataset(all_features)
test_data_auto = TabularDataset(test_features)
id, label = 'CUST_UID', 'LABEL'

train_data_auto_desc = train_data_auto.describe()

predictor = TabularPredictor(label=label).fit(train_data_auto)

# predictor = TabularPredictor(label=label).fit(train_data_auto.drop(columns=[id]))
"""
Fitting model: WeightedEnsemble_L2 ...
	0.9124	 = Validation score   (accuracy)
	0.86s	 = Training   runtime
	0.01s	 = Validation runtime
AutoGluon training complete, total runtime = 118.46s ... Best model: "WeightedEnsemble_L2"
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220429_115841\")
"""
import collections

# preds = predictor.predict(test_data_auto.drop(columns=[id]))
preds = predictor.predict(test_data_auto)
print(preds.head(5))

# preds_proba = predictor.predict_proba(test_data_auto.drop(columns=[id]))
preds_proba = predictor.predict_proba(test_data_auto)
print(preds_proba.head(5))

import collections
print(collections.Counter(preds))


# testA_id = test_data_auto['CUST_UID']
testA_preds_proba = preds_proba[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

submission[1] = submission[1].map(lambda x:round(x, 13))
submission.to_csv('submission_6.txt', sep='\t', index=False, header=False)


















from autogluon.tabular import TabularDataset, TabularPredictor

train_data_auto = TabularDataset(train_data)
test_data_auto = TabularDataset(test_data)
id, label = 'CUST_UID', 'LABEL'

predictor = TabularPredictor(label=label).fit(train_data_auto.drop(columns=id))

# predictor = TabularPredictor(label=label).fit(train_data_auto.drop(columns=[id]))
"""
Fitting model: LightGBMXT ...
	Warning: Exception caused LightGBMXT to fail during training (ImportError)... Skipping this model.
		cannot import name 'log_evaluation' from 'lightgbm.callback' (D:\Anaconda\envs\automl\lib\site-packages\lightgbm\callback.py)
Fitting model: LightGBM ...
	Warning: Exception caused LightGBM to fail during training (ImportError)... Skipping this model.
		cannot import name 'log_evaluation' from 'lightgbm.callback' (D:\Anaconda\envs\automl\lib\site-packages\lightgbm\callback.py)

Fitting model: WeightedEnsemble_L2 ...
	0.9124	 = Validation score   (accuracy)
	0.86s	 = Training   runtime
	0.01s	 = Validation runtime
AutoGluon training complete, total runtime = 118.46s ... Best model: "WeightedEnsemble_L2"
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220429_115841\")

去掉一部分特征 ，取log
Fitting model: LightGBMXT ...
	0.908	 = Validation score   (accuracy)
	1.87s	 = Training   runtime
	0.02s	 = Validation runtime
Fitting model: LightGBM ...
	0.9104	 = Validation score   (accuracy)
	1.3s	 = Training   runtime
	0.02s	 = Validation runtime
Fitting model: WeightedEnsemble_L2 ...
	0.9124	 = Validation score   (accuracy)
	1.12s	 = Training   runtime
	0.01s	 = Validation runtime
	Counter({0: 9054, 1: 2946})
"""
import collections

# preds = predictor.predict(test_data_auto.drop(columns=[id]))
preds = predictor.predict(test_data_auto.drop(columns=[id]))
print(preds.head(5))

# preds_proba = predictor.predict_proba(test_data_auto.drop(columns=[id]))
preds_proba = predictor.predict_proba(test_data_auto.drop(columns=[id]))
print(preds_proba.head(5))

import collections

print(collections.Counter(preds))

# testA_id = test_data_auto['CUST_UID']
testA_preds_proba = preds_proba[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

submission[1] = submission[1].map(lambda x: round(x, 13))
submission.to_csv('submission_b.txt', sep='\t', index=False, header=False)

