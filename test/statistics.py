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

pd.set_option('display.max_rows', 10, 'display.max_columns', 10, "display.max_colwidth", 30, 'display.width', 1000)
print('*' * 50 + '测试数据样例' + '*' * 50)
test_id_list = os.listdir("./input/coleridgeinitiative-show-us-the-data/test")
test_example = pd.read_json("./input/coleridgeinitiative-show-us-the-data/train/" + test_id_list[0])
print(test_example)
print()

print('*' * 50 + '读取训练集标签' + '*' * 50)
train_label = pd.read_csv("./input/coleridgeinitiative-show-us-the-data/train.csv")
print(train_label)
print()

print('*' * 50 + '读取提交示例' + '*' * 50)
sample_submission = pd.read_csv("./input/coleridgeinitiative-show-us-the-data/sample_submission.csv",index_col=0)
print(sample_submission)
print()

print('*' * 50 + '训练数据样例' + '*' * 50)
train_id_list = os.listdir("./input/coleridgeinitiative-show-us-the-data/train")
train_sample = pd.read_json("./input/coleridgeinitiative-show-us-the-data/train/" + train_id_list[0])
print(train_sample)
print()

print('*' * 50 + '读取训练数据' + '*' * 50)
print('将每个包含dataset_label的章节视为一个样本,重组训练数据')
train_data = pd.DataFrame(columns=['id', 'section_title', 'text', 'dataset_label'])
for id in tqdm(train_id_list[:100]):
    warnings.filterwarnings("ignore", 'This pattern has match groups')
    train_sample = pd.read_json('./input/coleridgeinitiative-show-us-the-data/train/' + id)
    dataset_label = train_label[(train_label['Id'] + '.json') == id]['dataset_label'].values[0]
    cleaned_label = train_label[(train_label['Id'] + '.json') == id]['cleaned_label'].values[0]
    new_sample = train_sample[train_sample['text'].str.contains(dataset_label)]
    new_sample = pd.concat([new_sample.reset_index(drop=True), pd.DataFrame({'id': [id] * len(new_sample)})], axis=1)
    new_sample = pd.concat(
        [new_sample.reset_index(drop=True), pd.DataFrame({'dataset_label': [dataset_label] * len(new_sample)})], axis=1)
    new_sample = pd.concat(
        [new_sample.reset_index(drop=True), pd.DataFrame({'cleaned_label': [cleaned_label] * len(new_sample)})], axis=1)
    train_data = pd.concat([train_data, new_sample], ignore_index=True)
print(train_data)
print()

print('*' * 50 + '训练数据简要分析' + '*' * 50)
print('数据缺失值统计:')
print(train_data.isnull().sum())
print()
print('章节长度')
train_statistics = pd.DataFrame({'text_len': [len(text) for text in train_data['text'].values]})
print(train_statistics)
print()
print('章节长度简要统计')
print(train_statistics['text_len'].describe())
print()

print('*' * 50 + '词云' + '*' * 50)
train_label_list = list(train_data['cleaned_label'].values)
datasets_label_list = list(train_data['dataset_label'].unique())
words = []
for _list in [word.split() for word in train_label_list]:
    words += _list
words_FreqDist = nltk.FreqDist(words)

print('Dataset_label中的常用词')
mostcommon = words_FreqDist.most_common(100)
dataset_label_word_cloud = wordcloud.WordCloud(width=1600, height=800, background_color='white').generate(
    str(mostcommon))
fig = plt.figure(figsize=(30, 10), facecolor='grey')
plt.imshow(dataset_label_word_cloud, interpolation="bilinear")
plt.axis('off')
plt.title('Top 100 Most Common Words in Data Label', fontsize=50)
plt.tight_layout(pad=0)
plt.show()
print()

print('常用词词频统计')
mostcommon_small = words_FreqDist.most_common(25)
x, y = zip(*mostcommon_small)
plt.figure(figsize=(50, 30))
plt.margins(0.02)
plt.bar(x, y)
plt.xlabel('Words', fontsize=50)
plt.ylabel('Frequency of Words', fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60, fontsize=40)
plt.title('Freq of 25 Most Common Words in Data-Label', fontsize=60)
plt.show()
print()

stopwords = ['in', 'to']
section_title_list = list(train_data['section_title'].values)
words2 = []
for _list in [str(word).split() for (word) in section_title_list]:
    _new_list = []
    for _word in _list:
        if _word not in stopwords:
            _new_list.append(_word)
    words2 += _new_list
words_FreqDist2 = nltk.FreqDist(words2)

print('Dataset_label中的常用词')
mostcommon2 = words_FreqDist2.most_common(100)
dataset_label_word_cloud = wordcloud.WordCloud(width=1600, height=800, background_color='white').generate(
    str(mostcommon2))
fig = plt.figure(figsize=(30, 10), facecolor='grey')
plt.imshow(dataset_label_word_cloud, interpolation="bilinear")
plt.axis('off')
plt.title('Top 100 Most Common Words in Data Label', fontsize=50)
plt.tight_layout(pad=0)
plt.show()
print()

print('常用词词频统计')
mostcommon_small2 = words_FreqDist2.most_common(25)
x, y = zip(*mostcommon_small2)
plt.figure(figsize=(50, 30))
plt.margins(0.02)
plt.bar(x, y)
plt.xlabel('Words', fontsize=50)
plt.ylabel('Frequency of Words', fontsize=50)
plt.yticks(fontsize=40)
plt.xticks(rotation=60, fontsize=40)
plt.title('Freq of 25 Most Common Words in Data-Label', fontsize=60)
plt.show()
print()

print('*' * 50 + '读取测试数据并整理' + '*' * 50)
print('将每个章节视为一个样本,重组测试数据')
test_data = pd.DataFrame(columns=['id', 'section_title', 'text'])
for id in test_id_list:
    df = pd.read_json('./input/coleridgeinitiative-show-us-the-data/test/' + id)
    pd.DataFrame({'id': [id] * len(df)})
    new_test_data = pd.concat([pd.DataFrame({'id': [id] * len(df)}), df], axis=1)
    test_data = pd.concat([test_data, new_test_data], axis=0).reset_index(drop=True)
print(test_data)
# sample_submission : 提交示例
print()


# 官方标准函数用于统一 dataset_label 格式
def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


print('*' * 50 + 'caseA' + '*' * 50)
print('如果训练集中的 dataset_label 在测试集文中出现, 就认为该文章使用了这一数据集')
# sample_submission : 提交示例
sample_submission = sample_submission.fillna('')
label_list = [clean_text(_label) for _label in train_label_list]

for _index in test_data.index.values:
    sub_text = clean_text(test_data["text"][_index])
    PredictionString = [sample_submission.loc[test_data['id'][_index][:-5],'PredictionString']]
    for _label in label_list:
        if (_label in sub_text) and (_label not in PredictionString):
            PredictionString.append(_label)
    sample_submission.loc[test_data['id'][_index][:-5],'PredictionString'] = '|'.join(PredictionString)
for _index in sample_submission.index.values:
    sample_submission.loc[_index, 'PredictionString'] = sample_submission.loc[_index, 'PredictionString'][1:]
print(sample_submission)
print()

print('*' * 50 + '写入casA_submission.csv' + '*' * 50)
sample_submission.to_csv('casA_submission.csv')
print('*' * 50 + '写完了' + '*' * 50)
print()