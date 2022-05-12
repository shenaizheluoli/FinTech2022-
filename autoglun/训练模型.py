import xgboost
import lightgbm
import sklearn
import numpy as np
import pandas as pd
"""
                                         importance  ...   p99_low
"CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL"          0.139000  ...  0.114023
"CUR_MON_COR_DPS_MON_DAY_AVG_BAL"            0.091000  ...  0.042042
"CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR"         0.026000  ... -0.026517
"LAST_12_MON_COR_DPS_DAY_AVG_BAL"            0.017000  ... -0.017855
"ICO_CUR_MON_ACM_TRX_AMT"                    0.011000  ... -0.006190
"MON_12_AGV_TRX_CNT"                         0.004667  ...  0.001358
"CUR_YEAR_MID_BUS_INC "                      0.004333  ... -0.004420
"CUR_YEAR_MON_AGV_TRX_CNT"                   0.004000  ... -0.011160
"NB_RCT_3_MON_LGN_TMS_AGV"                   0.003333  ... -0.011087
"MON_12_ACM_ENTR_ACT_CNT"                    0.003000  ... -0.016850
"REG_DT"                                     0.003000  ... -0.006925

"CUR_YEAR_MON_AGV_TRX_CNT" "MON_12_AGV_TRX_CNT" "MON_12_ACM_ENTR_ACT_CNT"  相关性很强
"LAST_12_MON_COR_DPS_DAY_AVG_BAL"  "CUR_MON_COR_DPS_MON_DAY_AVG_BAL"  相关性很强
"""

imp_name = ["CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL",
            "CUR_MON_COR_DPS_MON_DAY_AVG_BAL"
            # "CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR",
            # "LAST_12_MON_COR_DPS_DAY_AVG_BAL",
            # "ICO_CUR_MON_ACM_TRX_AMT",
            # "MON_12_AGV_TRX_CNT",
            # "CUR_YEAR_MID_BUS_INC ",
            # "NB_RCT_3_MON_LGN_TMS_AGV"
            ]

"ICO_CUR_MON_ACM_TRX_AMT"

import matplotlib.pyplot as plt
import pylab

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('testB_data.csv')
testA_id = test_data["CUST_UID"]


train_data['AI_STAR_SCO'] = train_data['AI_STAR_SCO'].astype(object)
test_data['AI_STAR_SCO'] = test_data['AI_STAR_SCO'].astype(object)

print(train_data.dtypes)
print(test_data.dtypes)

# 剔除重要性变量
colname = train_data.columns

colname = colname.drop(imp_name)

# 剔除和其他变量相关性高于0.8的特征
corr_drop = ['AGN_CUR_YEAR_WAG_AMT', 'AGN_AGR_LATEST_AGN_AMT', 'MON_12_EXT_SAM_TRSF_IN_AMT', 'MON_12_EXT_SAM_TRSF_OUT_AMT',
             'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT', 'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT', 'MON_12_AGV_TRX_CNT', 'MON_12_ACM_ENTR_ACT_CNT',
             'MON_12_AGV_ENTR_ACT_CNT', 'MON_12_ACM_LVE_ACT_CNT', 'MON_12_AGV_LVE_ACT_CNT', 'MON_6_50_UP_LVE_ACT_CNT',
             'LAST_12_MON_COR_DPS_DAY_AVG_BAL', 'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
             'LAST_12_MON_MON_AVG_TRX_AMT_NAV', 'CUR_YEAR_MID_BUS_INC', 'EMP_NBR', 'HLD_DMS_CCY_ACT_NBR']
colname = train_data.columns
colname = colname.drop(corr_drop)




# 算变量之间的相关性
id, label = 'CUST_UID', 'LABEL'
train_data_corr = train_data.drop(columns=id).corr()
test_data_corr = test_data.drop(columns=id).corr()

strname = ['MON_12_CUST_CNT_PTY_ID',
           'AI_STAR_SCO',
           'WTHR_OPN_ONL_ICO',
           'SHH_BCK',
           'LGP_HLD_CARD_LVL',
           'NB_CTC_HLD_IDV_AIO_CARD_SITU']

# colname = train_data.columns
# for c in train_data.columns[2:]:
#     if c not in strname:
#         colname = colname.drop(c)
# train_data = train_data[colname]
# colname = colname.drop('LABEL')
# test_data = test_data[colname]
id = range(30,35)
# 30 - 51 的特征即可达到0.9的精度 To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_055158\")
# 30 - 40 的特征即可达到0.9048的精度To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_055626\")
# 30 - 35 的特征0.906的进度 To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_060031\") 取log 0.9044	 = Validation score   (accuracy)
# 30 - 34 的特征0.8404的进度
# 10 - 35 的特征 0.9108	 = Validation score   (accuracy) To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_061240\")
# 25 - 35 的特征 0.9068	 = Validation score   (accuracy) To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_061543\")
# 31 - 35 的特征 0.9048	 = Validation score   (accuracy) To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_062724\")
# 32 - 35 的特征 0.9016	 = Validation score   (accuracy) To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_063003\")
# 33 - 35 的特征 0.8996	 = Validation score   (accuracy) To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_063255\")
# 34 - 35 的特征 0.8652	 = Validation score   (accuracy) To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_063431\")
# 10 - 51 的特征 0.91	 = Validation score   (accuracy)
# 'CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL'单变量就有0.8652的进度 只去掉此变量 有0.8976	 = Validation score   (accuracy)
colname = train_data.columns
for i in range(51):
    if i >= 2 and i not in id:
        colname = colname.drop(train_data.columns[i])

# colname = colname.drop('CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL')
train_data = train_data[colname]
test_data = test_data[colname.drop('LABEL')]

# 测试特征精度
from autogluon.tabular import TabularDataset, TabularPredictor
train_data_auto = TabularDataset(train_data)
test_data_auto = TabularDataset(test_data)
id, label = 'CUST_UID', 'LABEL'
predictor = TabularPredictor(label=label, eval_metric='roc_auc').fit(train_data_auto.drop(columns=id))

# 特征重要性
imp_feature = predictor.feature_importance(train_data)


# describe

# 把所有nan转变成0
for c in train_data.columns:
    if train_data[c].dtype == 'float64':
        train_data[c] = train_data[c].fillna(0)

for c in test_data.columns:
    if test_data[c].dtype == 'float64':
        test_data = test_data.fillna(0)


train_describe = train_data.describe()
test_describe = test_data.describe()



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
dropname = train_data.columns[null_sum > len(train_data) * 0.3]
len(train_data.columns[null_sum < len(train_data) * 0.3])
dropname = dropname.drop('MON_12_CUST_CNT_PTY_ID')
dropname = dropname.drop('LGP_HLD_CARD_LVL')

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
train_data_auto2 = TabularDataset(all_features)
test_data_auto2 = TabularDataset(test_features)
id, label = 'CUST_UID', 'LABEL'

train_data_auto_desc = train_data_auto2.describe()

predictor2 = TabularPredictor(label=label).fit(train_data_auto2)

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
preds_all = predictor2.predict(test_data_auto2)
print(preds_all.head(5))

# preds_proba = predictor.predict_proba(test_data_auto.drop(columns=[id]))
preds_proba2 = predictor2.predict_proba(test_data_auto2)
print(preds_proba2.head(5))

import collections
print(collections.Counter(preds_all))
# Counter({0: 8849, 1: 3151}) Fitting model: WeightedEnsemble_L2 ...
# 	0.9112	 = Validation score   (accuracy)

# testA_id = test_data_auto['CUST_UID']
testA_preds_proba = preds_proba2[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

submission[1] = submission[1].map(lambda x:round(x, 13))
submission.to_csv('submission_b7.txt', sep='\t', index=False, header=False)

"""
去掉一系列相关变量，然后one-hot
Name: LABEL, dtype: int64
          0         1
0  0.958413  0.041587
1  0.976424  0.023576
2  0.817022  0.182978
3  0.986474  0.013526
4  0.426637  0.573363
Counter({0: 10685, 1: 1315})
submission.to_csv('submission_b7.txt', sep='\t', index=False, header=False)
"""












from autogluon.tabular import TabularDataset, TabularPredictor
train_data_auto = TabularDataset(train_data)
test_data_auto = TabularDataset(test_data)
id, label = 'CUST_UID', 'LABEL'


predictor = TabularPredictor(label=label, eval_metric='roc_auc').fit(train_data_auto.drop(columns=id))



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
# Counter({0: 8750, 1: 3250})
# 去掉 null 大于 0.3 的 结果 Counter({0: 8978, 1: 3022})
# 30 - 35  不取log 是Counter({0: 9431, 1: 2569}) 取log 是 Counter({0: 9500, 1: 2500}) b33 b34
# 特征重要性排序前11 Counter({0: 8786, 1: 3214}) b35 精度0.912
# 特征重要性排序前7 Counter({0: 8781, 1: 3219}) b36 精度 0.911
# 特征重要性排序前5 Counter({0: 8830, 1: 3170}) b37 0.9088	 = Validation score   (accuracy)
# 特征重要性前10个, 取log Counter({0: 8693, 1: 3307}) 计算出来的精度 0.9124	 = Validation score   (accuracy) To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220510_090506\") submissionb4
# 特征取7个，不取log 0.9104	 = Validation score   (accuracy) ， Counter({0: 8716, 1: 3284})  取log 0.9112	 = Validation score   (accuracy) Counter({0: 8701, 1: 3299}) submissionb41
# 特征取7个，取log，best训练 0.9047	 = Validation score   (accuracy) Counter({0: 8871, 1: 3129})
# 19个特征 0.9056	 = Validation score   (accuracy)
# testA_id = test_data_auto['CUST_UID'] Counter({0: 8813, 1: 3187})

# Counter({0: 9290, 1: 2710}) 删掉18个 增加了第二重要性变量 b8

# 去掉重要性最前的2个变量 0.8104	 = Validation score   (accuracy) Counter({0: 11072, 1: 928}) c1

# 5.11日去掉相关性大于0.9和两个重要性变量，精度0.874 Counter({0: 9292, 1: 2708}) c2

# 去掉相关性大于0.8和第一imp_feature auc0.9 submission_e.txt Counter({0: 10555, 1: 1445})
"""
# 删除了一系列相关性变量 Name: LABEL, dtype: int64
#           0         1
# 0  0.984717  0.015283
# 1  0.995491  0.004509
# 2  0.865904  0.134096
# 3  0.993760  0.006240
# 4  0.482457  0.517543
# Counter({0: 10817, 1: 1183}) 进度0.89， b6
"""
testA_preds_proba = preds_proba[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

submission[1] = submission[1].map(lambda x:round(x, 13))
submission.to_csv('submission_d.txt', sep='\t', index=False, header=False)


imp_feature.to_csv('imp_feature.csv')

# Fitting model: WeightedEnsemble_L2 ...
# 	0.9128	 = Validation score   (accuracy)
# 	1.08s	 = Training   runtime
# 	0.01s	 = Validation runtime
# AutoGluon training complete, total runtime = 139.04s ... Best model: "WeightedEnsemble_L2"












best = pd.read_csv('best.txt',sep='\t')

for i in range(12000):
    if best.iloc[i,0] >= 0.5 and best.iloc[i,0] + 0.005 <= 1:
        best.iloc[i, 0] += 0.005
    if best.iloc[i, 0] < 0.5 and best.iloc[i, 0] - 0.005 > 0:
        best.iloc[i, 0] -= 0.005
best.to_csv('best_1.txt', sep='\t', index=True, header=False)


id = []

for i in range(12000):
    if 0.55 >= best.iloc[i,0] >= 0.45 and  0.55 >= submission.iloc[i,1] >= 0.45:
        id.append(i)

idt = []
for i in range(12000):
    if best.iloc[i,0] >= 0.5 and  submission.iloc[i,1]< 0.5:
        idt.append(i)

idf = []
for i in range(12000):
    if best.iloc[i,0] < 0.5 and  submission.iloc[i,1] >=  0.5:
        idf.append(i)









d = 'MON_12_EXT_SAM_AMT'

np.log(train_data[d])

if train_data[d] < 0:
    train_data[d] = -np.log(-train_data[d])


train_data[train_data[d] < 0][d]

train_data[train_data[d] > 0][d]

train_data[train_data[d] > 0][d]

for i in range(40000):
    if train_data.loc[i, d] < 0:
        train_data.loc[i, d] = -np.log(-train_data.loc[i, d])
    if train_data.loc[i, d] > 0:
        train_data.loc[i, d] = np.log(train_data.loc[i, d])























# xgboost
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
for c in train_data.columns:
    if train_data[c].dtype == 'object':
        train_data[c] = lbl.fit_transform(train_data[c].astype(str))
print(train_data.dtypes)
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()


X = all_features.iloc[:,1:]
X = np.array(X)
y = all_features['LABEL']
y = list(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from pandas import MultiIndex, Int64Index
# 训练模型
## 导入XGBoost模型
from xgboost.sklearn import XGBClassifier
## 定义 XGBoost模型
clf =  xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective="binary:logistic")
# 在训练集上训练XGBoost模型
clf.fit(X_train, y_train)

# 对测试集进行预测
ans = clf.predict(X_test)

from sklearn import metrics

predict_prob_y = clf.predict_proba(X_test)
test_auc = metrics.roc_auc_score(list(y_test),predict_prob_y[:,1])#验证集上的auc值
print (test_auc)

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test.iloc[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))




clf =  xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective="binary:logistic")
# 在训练集上训练XGBoost模型
clf.fit(X, y)

# 对测试集进行预测
ans = clf.predict(X)
predict_prob_y = clf.predict_proba(X_test)
test_auc = metrics.roc_auc_score(list(y_test),predict_prob_y[:,1])#验证集上的auc值
print (test_auc)

cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y.iloc[i]:
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

preds_xgb_proba = clf.predict_proba(test_features)
preds_xgb_proba[:,1]
testA_id






from autogluon.tabular import TabularDataset, TabularPredictor
train_data_auto = TabularDataset(all_features)
test_data_auto = TabularDataset(test_features)
id, label = 'CUST_UID', 'LABEL'

predictor = TabularPredictor(label=label).fit(train_data_auto)
"""
Fitting model: WeightedEnsemble_L2 ...
	0.9124	 = Validation score   (accuracy)
	0.87s	 = Training   runtime
	0.01s	 = Validation runtime
AutoGluon training complete, total runtime = 116.07s ... Best model: "WeightedEnsemble_L2"
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220429_135742\")
"""
import collections

preds = predictor.predict(test_data_auto)
print(preds.head(5))
preds_proba = predictor.predict_proba(test_data_auto)
print(preds_proba.head(5))

import collections
print(collections.Counter(preds))

testA_id
testA_preds_proba = preds_proba[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

submission[1] = submission[1].map(lambda x:round(x, 13))
submission.to_csv('submission_4.txt', sep='\t', index=False, header=False)













a = 10
a = 's'



dict = {[1,2]:5}






# 判断特征是否有用

from autogluon.tabular import TabularDataset, TabularPredictor

name = train_data.columns
# name[2] name[50]
id = 17

train_data_auto = TabularDataset(pd.concat((train_data['LABEL'], train_data[name[id]]), axis=1))
test_data_auto = TabularDataset(test_data[name[id]])
id, label = 'CUST_UID', 'LABEL'

# pd.concat((train_data['LABEL'], train_data[name[id]]), axis=1)

predictor = TabularPredictor(label=label).fit(train_data_auto)

# predictor = TabularPredictor(label=label).fit(train_data_auto.drop(columns=[id]))
"""
2 :0.7504	 = Validation score   (accuracy)
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


"""
null_sum
CUST_UID                                       0
LABEL                                          0
AGN_CNT_RCT_12_MON                         21408
ICO_CUR_MON_ACM_TRX_TM                      3503
NB_RCT_3_MON_LGN_TMS_AGV                    3815
AGN_CUR_YEAR_AMT                           22541
AGN_CUR_YEAR_WAG_AMT                       27985
AGN_AGR_LATEST_AGN_AMT                     19593
ICO_CUR_MON_ACM_TRX_AMT                     3503
COUNTER_CUR_YEAR_CNT_AMT                     946
PUB_TO_PRV_TRX_AMT_CUR_YEAR                  946
MON_12_EXT_SAM_TRSF_IN_AMT                   254
MON_12_EXT_SAM_TRSF_OUT_AMT                  254
MON_12_EXT_SAM_NM_TRSF_OUT_CNT               254
MON_12_EXT_SAM_AMT                           254
CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT             946
CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT            946
MON_12_CUST_CNT_PTY_ID                     22941
MON_12_TRX_AMT_MAX_AMT_PCTT                 9600
CUR_YEAR_MON_AGV_TRX_CNT                     946
MON_12_AGV_TRX_CNT                           254
MON_12_ACM_ENTR_ACT_CNT                      254
MON_12_AGV_ENTR_ACT_CNT                      254
MON_12_ACM_LVE_ACT_CNT                       254
MON_12_AGV_LVE_ACT_CNT                       254
CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT            20136
MON_6_50_UP_ENTR_ACT_CNT                    6620
MON_6_50_UP_LVE_ACT_CNT                     6620
CUR_YEAR_COUNTER_ENCASH_CNT                  946
MON_12_ACT_OUT_50_UP_CNT_PTY_QTY            5860
MON_12_ACT_IN_50_UP_CNT_PTY_QTY             5860
LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL      114
LAST_12_MON_COR_DPS_DAY_AVG_BAL              114
CUR_MON_COR_DPS_MON_DAY_AVG_BAL              116
CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL            211
CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR           113
LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV       254
LAST_12_MON_MON_AVG_TRX_AMT_NAV              254
COR_KEY_PROD_HLD_NBR                         336
CUR_YEAR_MID_BUS_INC                          21
AI_STAR_SCO                                  413
WTHR_OPN_ONL_ICO                            1342
EMP_NBR                                      106
REG_CPT                                     3732
SHH_BCK                                      106
HLD_DMS_CCY_ACT_NBR                          114
REG_DT                                      1938
LGP_HLD_CARD_LVL                           16752
OPN_TM                                       113
NB_CTC_HLD_IDV_AIO_CARD_SITU                6833
HLD_FGN_CCY_ACT_NBR                          114
"""

# 30 - 35 b榜0.707

# 特征重要性
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
CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT            0.001000  ... -0.004730
MON_12_EXT_SAM_TRSF_IN_AMT                 0.001000  ... -0.004730
MON_12_AGV_LVE_ACT_CNT                     0.001000  ... -0.004730
HLD_FGN_CCY_ACT_NBR                        0.001000  ...  0.001000
MON_12_EXT_SAM_AMT                         0.001000  ... -0.010460
CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT          0.001000  ... -0.010460
CUR_YEAR_COUNTER_ENCASH_CNT                0.001000  ... -0.004730
MON_12_EXT_SAM_NM_TRSF_OUT_CNT             0.000667  ... -0.002642
CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT           0.000667  ... -0.008086
MON_12_CUST_CNT_PTY_ID                     0.000667  ... -0.005950
WTHR_OPN_ONL_ICO                           0.000667  ... -0.002642
MON_12_ACT_IN_50_UP_CNT_PTY_QTY            0.000667  ... -0.005950
HLD_DMS_CCY_ACT_NBR                        0.000667  ... -0.005950
LGP_HLD_CARD_LVL                           0.000667  ... -0.002642
COR_KEY_PROD_HLD_NBR                       0.000667  ... -0.002642
AI_STAR_SCO                                0.000333  ... -0.002975
SHH_BCK                                    0.000333  ... -0.002975
AGN_CUR_YEAR_AMT                           0.000333  ... -0.008420
NB_CTC_HLD_IDV_AIO_CARD_SITU               0.000333  ... -0.002975
AGN_AGR_LATEST_AGN_AMT                    -0.000333  ... -0.009086

"""