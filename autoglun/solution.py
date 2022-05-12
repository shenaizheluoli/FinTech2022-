import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
# 通过autogluon包调用各种机器学习模型 (python = 3.8)
# pip install autogluon
from autogluon.tabular import TabularDataset, TabularPredictor

"""
A榜
"""
"""
第一步 清洗数据
"""

"""
train 数据集清洗 ？ 数据
"""
data = pd.read_csv('train.csv')
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
testA 数据集清洗 ？ 数据
"""
testdata = pd.read_csv('test_A.csv')
print(testdata.dtypes)
testcolname = testdata.columns[1:]
for c in testcolname:
    if c not in strname:
        testdata[c] = testdata[c].replace(r'[?]', np.nan, regex=True).astype(float)
    else:
        testdata[c] = testdata[c].replace(r'[?]', np.nan, regex=True)

data.to_csv('train_data.csv',index=False)
testdata.to_csv('test_data.csv',index=False)
"""
testB 数据集清洗 ？ 数据
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

"""
第二步 读取数据
"""
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('testB_data.csv')
testA_id = test_data["CUST_UID"]

train_data['AI_STAR_SCO'] = train_data['AI_STAR_SCO'].astype(object)
test_data['AI_STAR_SCO'] = test_data['AI_STAR_SCO'].astype(object)


# 初步判断 所有特征的重要性
from autogluon.tabular import TabularDataset, TabularPredictor
train_data_auto = TabularDataset(train_data)
test_data_auto = TabularDataset(test_data)
id, label = 'CUST_UID', 'LABEL'
predictor = TabularPredictor(label=label, eval_metric='roc_auc').fit(train_data_auto.drop(columns=id))

# 特征重要性
imp_feature = predictor.feature_importance(train_data)
print(imp_feature[0:10])
"""
CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL 
CUR_MON_COR_DPS_MON_DAY_AVG_BAL
两个变量占据最多重要性，可能是强特征，也可能是毒特征
"""

"""
通过相关性选择特征
"""
# 计算变量的相关性
id, label = 'CUST_UID', 'LABEL'
train_data_corr = train_data.drop(columns=id).corr()
test_data_corr = test_data.drop(columns=id).corr()
corr_df=train_data_corr
# 热力图
# 剔除相关性系数高于threshold的corr_drop
threshold = 0.95
upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool))
corr_drop = []
corr_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
# 剔除两个脏变量 通过计算train数据的imp_feature和相关度，发现这两个变量强相关高重要性
corr_drop.append("CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL")
corr_drop.append("CUR_MON_COR_DPS_MON_DAY_AVG_BAL")


# 数据集drop特征
colname = train_data.columns
colname = colname.drop(corr_drop)

train_data = train_data[colname]
test_data = test_data[colname.drop('LABEL')]


# 测试特征精度 通过autogluon包调用模型 (python = 3.8)
# pip install autogluon
from autogluon.tabular import TabularDataset, TabularPredictor

train_data_auto = TabularDataset(train_data)
test_data_auto = TabularDataset(test_data)
id, label = 'CUST_UID', 'LABEL'
predictor = TabularPredictor(label=label, eval_metric='roc_auc').fit(train_data_auto.drop(columns=id))


# 预测01
# preds = predictor.predict(test_data_auto.drop(columns=[id]))
preds = predictor.predict(test_data_auto.drop(columns=[id]))
print(preds.head(5))

# 预测01 预测概率
# preds_proba = predictor.predict_proba(test_data_auto.drop(columns=[id]))
preds_proba = predictor.predict_proba(test_data_auto.drop(columns=[id]))
print(preds_proba.head(5))

import collections
print(collections.Counter(preds))


# 输出结果submission
testA_preds_proba = preds_proba[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

submission[1] = submission[1].map(lambda x:round(x, 13))
submission.to_csv('submission_z.txt', sep='\t', index=False, header=False)








