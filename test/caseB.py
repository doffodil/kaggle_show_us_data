import time
import random
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json, os, re
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda
from keras.models import Model


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_score(submission_file: str):
    pass


def data2qa(file_id: str):
    labels = train_label.loc[train_label.loc[:, 'Id'] == file_id[:-5], :].loc[:, 'cleaned_label'].values
    sample = pd.read_json('../input/coleridgeinitiative-show-us-the-data/train/' + file_id)
    text_list = []
    for _item in [text for text in sample.loc[:, 'text']]:
        text_list.extend(_item.split('.'))
    passages = []
    for passage in text_list:
        passage = clean_text(passage).strip()
        for label in labels:
            if (label in passage) and (len(passage) > 5):
                answer = label
                passages.append({'answer': answer, 'passage': passage})
            else:
                answer = ""
                if ('data' in passage) and (len(passage) > 5):
                    if random.randint(1, 100) < 15:
                        passages.append({'answer': answer, 'passage': passage})
    qa_sample = {'passages': passages, 'question': 'dataset', 'id': file_id[:-5]}
    return qa_sample


def test_data2qa(file_id: str):
    labels = train_label.loc[train_label.loc[:, 'Id'] == file_id[:-5], :].loc[:, 'cleaned_label'].values
    sample = pd.read_json('../input/coleridgeinitiative-show-us-the-data/test/' + file_id)
    text_list = []
    for _item in [text for text in sample.loc[:, 'text']]:
        text_list.extend(_item.split('.'))
    passages = []
    for passage in text_list:
        passage = clean_text(passage).strip()
        if len(passage) > 5:
            passages.append({'passage': passage})
    qa_sample = {'passages': passages, 'question': 'dataset', 'id': file_id[:-5]}
    return qa_sample


# 加载数据
train_label = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/train.csv")
test_id_list = os.listdir("../input/coleridgeinitiative-show-us-the-data/test")
sample_submission = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/sample_submission.csv")
train_id_list = os.listdir("../input/coleridgeinitiative-show-us-the-data/train")

# 训练参数
max_p_len = 256
max_q_len = 64
max_a_len = 32
batch_size = 1
epochs = 3

# bert配置
config_path = '../pretrain_model//uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../pretrain_model//uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../pretrain_model//uncased_L-12_H-768_A-12/vocab.txt'


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        """单条样本格式为
        输入：[CLS][MASK][MASK][SEP]问题[SEP]篇章[SEP]
        输出：答案
        """
        batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []
        for is_end, D in self.sample(random):
            question = D['question']
            answers = [p['answer'] for p in D['passages'] if p['answer']]
            passage = np.random.choice(D['passages'])['passage']
            passage = re.sub(u' |、|；|，', ',', passage)
            final_answer = ''
            for answer in answers:
                if all([
                    a in passage[:max_p_len - 2] for a in answer.split(' ')
                ]):
                    final_answer = answer.replace(' ', ',')
                    break
            a_token_ids, _ = tokenizer.encode(
                final_answer, maxlen=max_a_len + 1
            )
            q_token_ids, _ = tokenizer.encode(question, maxlen=max_q_len + 1)
            p_token_ids, _ = tokenizer.encode(passage, maxlen=max_p_len + 1)
            token_ids = [tokenizer._token_start_id]
            token_ids += ([tokenizer._token_mask_id] * max_a_len)
            token_ids += [tokenizer._token_end_id]
            token_ids += (q_token_ids[1:] + p_token_ids[1:])
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_a_token_ids.append(a_token_ids[1:])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_a_token_ids = sequence_padding(
                    batch_a_token_ids, max_a_len
                )
                yield [batch_token_ids, batch_segment_ids], batch_a_token_ids
                batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []


def masked_cross_entropy(y_true, y_pred):
    """交叉熵作为loss，并mask掉padding部分的预测
    """
    y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
    y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
    return cross_entropy


def get_ngram_set(x, n):
    """生成ngram合集，返回结果格式是:
    {(n-1)-gram: set([n-gram的第n个字集合])}
    """
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i:i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result


def gen_answer(question, passages):
    """由于是MLM模型，所以可以直接argmax解码。
    """
    all_p_token_ids, token_ids, segment_ids = [], [], []
    for passage in passages:
        passage = re.sub(u' |、|；|，', ',', passage)
        p_token_ids, _ = tokenizer.encode(passage, maxlen=max_p_len + 1)
        q_token_ids, _ = tokenizer.encode(question, maxlen=max_q_len + 1)
        all_p_token_ids.append(p_token_ids[1:])
        token_ids.append([tokenizer._token_start_id])
        token_ids[-1] += ([tokenizer._token_mask_id] * max_a_len)
        token_ids[-1] += [tokenizer._token_end_id]
        token_ids[-1] += (q_token_ids[1:] + p_token_ids[1:])
        segment_ids.append([0] * len(token_ids[-1]))
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    probas = model.predict([token_ids, segment_ids])
    results = {}
    for t, p in zip(all_p_token_ids, probas):
        a, score = tuple(), 0.
        for i in range(max_a_len):
            idxs = list(get_ngram_set(t, i + 1)[a])
            if tokenizer._token_end_id not in idxs:
                idxs.append(tokenizer._token_end_id)
            # pi是将passage以外的token的概率置零
            pi = np.zeros_like(p[i])
            pi[idxs] = p[i, idxs]
            a = a + (pi.argmax(),)
            score += pi.max()
            if a[-1] == tokenizer._token_end_id:
                break
        score = score / (i + 1)
        a = tokenizer.decode(a)
        if a:
            results[a] = results.get(a, []) + [score]
    results = {
        k: (np.array(v) ** 2).sum() / (sum(v) + 1)
        for k, v in results.items()
    }
    return results


def max_in_dict(d):
    if d:
        return sorted(d.items(), key=lambda s: -s[1])[0][0]


def predict_to_file(data, filename):
    """将预测结果输出到文件，方便评估
    """
    predicts = []
    for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
        q_text = d['question']
        p_texts = [p['passage'] for p in d['passages']]
        a = gen_answer(q_text, p_texts)
        a = max_in_dict(a)
        if a:
            s = {"id":d['id'], "answer":a}
            # s = u'%s\t%s\n' % (d['id'], a)
        else:
            # s = u'%s\t\n' % (d['id'])
            s = {"id": d['id'], "answer": ""}
        predicts.append(s)
    json.dump(predicts, open(filename, 'w', encoding='utf-8'), indent=4)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')


if __name__ == '__main__':
    # 将数据整理成QA的数据
    if not os.path.exists('../datasets/qa_show_us_data/' + 'qa_show_us_data_train.json'):
        time_s = time.time()
        data_list = []
        with Pool(cpu_count()) as pool:
            sample_list = pool.map(data2qa, train_id_list)
            data_list.extend(sample_list)
        json.dump(data_list, open(('../datasets/qa_show_us_data/' + 'qa_show_us_data_train.json'), 'w'), indent=4)
        webqa_data = json.load(open('../datasets/qa_show_us_data/qa_show_us_data_train.json', encoding='utf-8'))
        sogou_data = json.load(open('../datasets/qa_show_us_data/qa_show_us_data_train.json', encoding='utf-8'))
        print('整理训练数据耗时:', time.time() - time_s)
    else:
        webqa_data = json.load(open('../datasets/qa_show_us_data/qa_show_us_data_train.json', encoding='utf-8'))
        sogou_data = json.load(open('../datasets/qa_show_us_data/qa_show_us_data_train.json', encoding='utf-8'))

    # 保存一个随机序（供划分valid用）
    if not os.path.exists('random_order.json'):
        random_order = list(range(len(sogou_data)))
        np.random.shuffle(random_order)
        json.dump(random_order, open('random_order.json', 'w'), indent=4)
    else:
        random_order = json.load(open('random_order.json'))

    # 划分valid
    train_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 != 0]
    valid_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 == 0]
    train_data.extend(train_data)
    train_data.extend(webqa_data)  # 将SogouQA和WebQA按2:1的比例混合

    # 加载并精简词表，建立分词器
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)

    model = build_transformer_model(
        config_path,
        checkpoint_path,
        with_mlm=True,
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )
    output = Lambda(lambda x: x[:, 1:max_a_len + 1])(model.output)
    model = Model(model.input, output)
    model.summary()

    model.compile(loss=masked_cross_entropy, optimizer=Adam(1e-5))



    # 训练模型
    if not os.path.exists('../model_weight/best_model2.weights'):
        time_s = time.time()
        evaluator = Evaluator()
        train_generator = data_generator(train_data, batch_size)

        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
        print('模型训练耗时:', time.time() - time_s)

    # 整理测试文件
    if not os.path.exists('../datasets/qa_show_us_data/' + 'qa_show_us_data_test.json'):
        time_s = time.time()
        data_list = []
        # 将数据整理成QA的数据
        with Pool(cpu_count()) as pool:
            sample_list = pool.map(test_data2qa, test_id_list)
            data_list.extend(sample_list)
        json.dump(data_list, open(('../datasets/qa_show_us_data/' + 'qa_show_us_data_test.json'), 'w'), indent=4)
        print('整理测试数据耗时:', time.time() - time_s)

    # 预测
    time_s = time.time()
    if os.path.exists('../model_weight/best_model2.weights'):
        model.load_weights('best_model2.weights')
    data_dir = '../datasets/qa_show_us_data/qa_show_us_data_test.json'
    dataset = json.load(open(data_dir, encoding='utf-8'))
    predict_to_file(dataset, '../datasets/qa_show_us_data/output.json')
    print('模型预测耗时:', time.time() - time_s)

    # 整理测试结果
    time_s = time.time()
    result = pd.read_json('../datasets/qa_show_us_data/output.json')
    for _index in result.index.values:
        answer = result.loc[_index, 'answer']
        answer = ' '.join([item.strip() for item in answer.split(',')])
        result.loc[_index, 'answer'] = answer
    result.columns = ['Id', 'PredictionString']
    result.to_csv('submission.csv', index=False)
    print('提交结果耗时:', time.time() - time_s)

    # 计算score
    calculate_score('submission.csv')
