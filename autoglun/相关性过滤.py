from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import numpy as np
import pandas as pd

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('testB_data.csv')
testA_id = test_data["CUST_UID"]


train_data['AI_STAR_SCO'] = train_data['AI_STAR_SCO'].astype(object)
test_data['AI_STAR_SCO'] = test_data['AI_STAR_SCO'].astype(object)


colname = train_data.columns
strname = ['MON_12_CUST_CNT_PTY_ID',
           'AI_STAR_SCO',
           'WTHR_OPN_ONL_ICO',
           'SHH_BCK',
           'LGP_HLD_CARD_LVL',
           'NB_CTC_HLD_IDV_AIO_CARD_SITU']
colname = train_data.columns
colname = colname.drop(strname)
# 大于0.8的
corr_drop = ['AGN_CUR_YEAR_WAG_AMT', 'AGN_AGR_LATEST_AGN_AMT', 'MON_12_EXT_SAM_TRSF_IN_AMT', 'MON_12_EXT_SAM_TRSF_OUT_AMT',
             'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT', 'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT', 'MON_12_AGV_TRX_CNT', 'MON_12_ACM_ENTR_ACT_CNT',
             'MON_12_AGV_ENTR_ACT_CNT', 'MON_12_ACM_LVE_ACT_CNT', 'MON_12_AGV_LVE_ACT_CNT', 'MON_6_50_UP_LVE_ACT_CNT',
             'LAST_12_MON_COR_DPS_DAY_AVG_BAL', 'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
             'LAST_12_MON_MON_AVG_TRX_AMT_NAV', 'CUR_YEAR_MID_BUS_INC', 'EMP_NBR', 'HLD_DMS_CCY_ACT_NBR',
             # "CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL",
             "CUR_MON_COR_DPS_MON_DAY_AVG_BAL"
             ]
# 大于0.9的
corr_drop = ['AGN_CUR_YEAR_WAG_AMT',
             'MON_12_EXT_SAM_TRSF_OUT_AMT',
             'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT',
             'MON_12_AGV_TRX_CNT',
             'MON_12_ACM_ENTR_ACT_CNT',
             'MON_12_AGV_ENTR_ACT_CNT',
             'MON_12_ACM_LVE_ACT_CNT',
             'MON_12_AGV_LVE_ACT_CNT',
             'MON_6_50_UP_LVE_ACT_CNT',
             'CUR_MON_COR_DPS_MON_DAY_AVG_BAL',
             'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
             'CUR_YEAR_MID_BUS_INC', 'EMP_NBR',
             "CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL",
             "CUR_MON_COR_DPS_MON_DAY_AVG_BAL",
             ]

colname = train_data.columns
colname = colname.drop(corr_drop)

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

import collections

# preds = predictor.predict(test_data_auto.drop(columns=[id]))
preds = predictor.predict(test_data_auto.drop(columns=[id]))
print(preds.head(5))

# preds_proba = predictor.predict_proba(test_data_auto.drop(columns=[id]))
preds_proba = predictor.predict_proba(test_data_auto.drop(columns=[id]))
print(preds_proba.head(5))

import collections
print(collections.Counter(preds))

testA_preds_proba = preds_proba[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

submission[1] = submission[1].map(lambda x:round(x, 13))
submission.to_csv('submission_h.txt', sep='\t', index=False, header=False)



# 只去掉2个imp 0.734	 = Validation score   (roc_auc) submission_f.txt
# 去掉0.95特征 加两个重要性变量 submission_g.txt Counter({0: 9386, 1: 2614}) 得分0.83
# 去掉0.92相关性特征 去两个重要性变量 submission_h Counter({0: 9257, 1: 2743})

















# 计算变量的相关性
id, label = 'CUST_UID', 'LABEL'
train_data_corr = train_data.drop(columns=id).corr()
test_data_corr = test_data.drop(columns=id).corr()

train_data = train_data[colname]
colname = colname.drop('LABEL')
test_data = test_data[colname]



x = train_data[train_data.columns[2:]]

y = train_data[train_data.columns[1]]

df_norm = (x - x.min()) / (x.max() - x.min())


#假设在这里我一直我需要300个特征
X_fschi = SelectKBest(chi2, k=30).fit_transform(df_norm, y)
#SelectKBest(所依赖的统计量, k=选择前K个特征数量)
#X_fsvar中位数方差过滤过的了的数据
X_fschi.shape


cross_val_score(RFC(n_estimators=10,random_state=0),X_fschi,y,cv=5).mean()
#交叉验证后求均值


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
corr_drop.append("CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL")
corr_drop.append("CUR_MON_COR_DPS_MON_DAY_AVG_BAL")
for c in corr_drop:
    print(c,'\r\n')

drop = ['MON_12_AGV_TRX_CNT',
        'MON_12_ACM_ENTR_ACT_CNT',
        'MON_12_AGV_ENTR_ACT_CNT',
        'MON_12_ACM_LVE_ACT_CNT',
        'MON_12_AGV_LVE_ACT_CNT',
        'MON_6_50_UP_LVE_ACT_CNT',
        'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
        'EMP_NBR',
        'CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL',
        'CUR_MON_COR_DPS_MON_DAY_AVG_BAL']

"""
AGN_CUR_YEAR_WAG_AMT 
AGN_AGR_LATEST_AGN_AMT 
MON_12_EXT_SAM_TRSF_IN_AMT 
MON_12_EXT_SAM_TRSF_OUT_AMT 
CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT 
CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT 
MON_12_AGV_TRX_CNT 
MON_12_ACM_ENTR_ACT_CNT 
MON_12_AGV_ENTR_ACT_CNT 
MON_12_ACM_LVE_ACT_CNT 
MON_12_AGV_LVE_ACT_CNT 
MON_6_50_UP_LVE_ACT_CNT 
LAST_12_MON_COR_DPS_DAY_AVG_BAL 
CUR_MON_COR_DPS_MON_DAY_AVG_BAL 
LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV 
LAST_12_MON_MON_AVG_TRX_AMT_NAV 
CUR_YEAR_MID_BUS_INC 
EMP_NBR 
HLD_DMS_CCY_ACT_NBR 
"""