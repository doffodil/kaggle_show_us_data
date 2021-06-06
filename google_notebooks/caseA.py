import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import warnings
import nltk
import wordcloud
import matplotlib.pyplot as plt
import re

def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()

pd.set_option('display.max_rows', 10, 'display.max_columns', 10, "display.max_colwidth", 30, 'display.width', 1000)

# 加载数据
test_id_list = os.listdir("../input/coleridgeinitiative-show-us-the-data/test")
train_label = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/train.csv")
sample_submission = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/sample_submission.csv")
train_id_list = os.listdir("../input/coleridgeinitiative-show-us-the-data/train")

# 整理训练数据
train_data = pd.DataFrame(columns=['id', 'section_title', 'text', 'dataset_label'])
for id in train_id_list[:200]:
    warnings.filterwarnings("ignore", 'This pattern has match groups')
    train_sample = pd.read_json('../input/coleridgeinitiative-show-us-the-data/train/' + id)
    dataset_label = clean_text(train_label[(train_label['Id'] + '.json') == id]['dataset_label'].values[0])
    cleaned_label = clean_text(train_label[(train_label['Id'] + '.json') == id]['cleaned_label'].values[0])
    index = [(dataset_label in clean_text(text_str)) for text_str in train_sample.loc[:,'text']]
    new_sample = train_sample[index]
    new_sample = pd.concat([new_sample.reset_index(drop=True), pd.DataFrame({'id': [id] * len(new_sample)})], axis=1)
    new_sample = pd.concat(
        [new_sample.reset_index(drop=True), pd.DataFrame({'dataset_label': [dataset_label] * len(new_sample)})], axis=1)
    # new_sample = pd.concat(
    #     [new_sample.reset_index(drop=True), pd.DataFrame({'cleaned_label': [cleaned_label] * len(new_sample)})], axis=1)
    train_data = pd.concat([train_data, new_sample], ignore_index=True)

# 整理测试数据
test_data = pd.DataFrame(columns=['id', 'section_title', 'text'])
for id in test_id_list:
    df = pd.read_json('../input/coleridgeinitiative-show-us-the-data/test/' + id)
    pd.DataFrame({'id': [id] * len(df)})
    new_test_data = pd.concat([pd.DataFrame({'id': [id] * len(df)}), df], axis=1)
    test_data = pd.concat([test_data, new_test_data], axis=0).reset_index(drop=True)

# 数据集名称出现在文本中,则认为该文章包含该数据集
sample_submission = sample_submission.fillna('')
label_set = set(train_label.loc[:,'cleaned_label'].values)
for _index in test_data.index.values:
    sub_text = clean_text(test_data["text"][_index])
    PredictionString = sample_submission.loc[sample_submission['Id']+'.json' == test_data.loc[_index,'id'],'PredictionString'].values
    for _label in label_set:
        if (_label in sub_text) and (_label not in PredictionString[0]):
            PredictionString = np.append(PredictionString,_label)
    sample_submission.loc[sample_submission['Id']+'.json' == test_data.loc[_index,'id'],'PredictionString'] = np.array('|'.join(PredictionString))

# 输出结果
for _index in sample_submission.index.values:
    sample_submission.loc[_index, 'PredictionString'] = sample_submission.loc[_index, 'PredictionString'][1:]
sample_submission.to_csv('submissionA.csv',index=False)
sample_submission.to_csv('submission.csv',index=False)