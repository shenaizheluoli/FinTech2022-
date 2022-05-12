import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
autogluon训练

AutoGluon training complete, total runtime = 146.84s ... Best model: "WeightedEnsemble_L2"
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220429_050531\")
"""
from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset(data)
test_data = TabularDataset(testdata)
id, label = 'CUST_UID', 'LABEL'

predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]), presets='best_quality')



preds = predictor.predict(test_data.drop(columns=[id]))
print(preds.head(5))
preds_proba = predictor.predict_proba(test_data.drop(columns=[id]))
print(preds_proba.head(5))

testA_id = test_data['CUST_UID']
testA_preds_proba = preds_proba[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

submission[1] = submission[1].map(lambda x:round(x, 13))
# submission.to_csv('submission_3.txt', sep='\t', index=False, header=False)
submission.to_csv('submission_3.csv',index=False, header=False)

import collections
print(collections.Counter(preds))
# Counter({0: 9063, 1: 2937})

model_summary = predictor.fit_summary(show_plot=True)
model_summary2 = predictor.fit_summary(show_plot=True)

import autogluon.tabular

from autogluon.tabular import TabularDataset, TabularPredictor
save_path = 'agModels-predictClass'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)



