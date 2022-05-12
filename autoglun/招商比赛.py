import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data/zhaoshang/train.csv')
print(data.shape)
print(data.head())
null_sum = data.isnull().sum()
null_sum2 = data.isin('?').sum()
print(data.columns[null_sum < len(data) * 0.2])

from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('train.csv')
id, label = 'CUST_UID', 'LABEL'
predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]))
"""
AutoGluon training complete, total runtime = 343.95s ... Best model: "WeightedEnsemble_L2"
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220429_025711\")
"""
import collections

test_data = TabularDataset('test_A.csv')
preds = predictor.predict(test_data.drop(columns=[id]))
print(preds.head(5))
preds_proba = predictor.predict_proba(test_data.drop(columns=[id]))
print(preds_proba.head(5))

testA_id = test_data['CUST_UID']
testA_preds_proba = preds_proba[1]
submission = pd.concat([testA_id, testA_preds_proba], axis=1)

# -*- coding: UTF-8 -*-
import sys
import json

sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
# 读取excel保存成txt格式
submission[1] = submission[1].map(lambda x:round(x, 10))

submission.to_csv('submission_1.txt', sep='\t', index=False, header=False)




collections.Counter(preds)

traindata = pd.read_csv('train.csv')
collections.Counter(traindata['LABEL'])
