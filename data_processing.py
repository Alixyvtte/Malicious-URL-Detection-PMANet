from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np

def dataPreprocess(filename, input_ids, input_types, input_masks, label, urltype):
    pad_size = 200  # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)
    bert_path = "/hy-tmp/charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器
    with open(filename, encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f)):
            x1 = l.strip()
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1 + ["[SEP]"]

            # 得到input_id, seg_id, att_mask
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)

            # 短则补齐，长则切断
            if len(ids) < pad_size:
                types = types + [1] * (pad_size - len(ids))  # mask部分 segment置为1
                masks = masks + [0] * (pad_size - len(ids))
                ids = ids + [0] * (pad_size - len(ids))
            else:
                types = types[:pad_size]
                masks = masks[:pad_size]
                ids = ids[:pad_size]
            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)
            #         print(len(ids), len(masks), len(types))
            assert len(ids) == len(masks) == len(types) == pad_size
            if urltype == 1:
                label.append([1])
            elif urltype == 0:
                label.append([0])


def dataPreprocessFromCSV(filename, input_ids, input_types, input_masks, label):
    pad_size = 200
    bert_path = "/hy-tmp/charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器

    data = pd.read_csv(filename, encoding='utf-8')
    for i, row in tqdm(data.iterrows(), total=len(data)):
        x1 = row['url']  # 请替换为你CSV文件中文本数据所在的列名
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

        # 得到input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(ids))
        masks = [1] * len(ids)

        # 短则补齐，长则切断
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))  # mask部分 segment置为1
            masks = masks + [0] * (pad_size - len(ids))
            ids = ids + [0] * (pad_size - len(ids))
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)
        assert len(ids) == len(masks) == len(types) == pad_size

        y = row['label']
        if y == 'malicious':
            label.append([1])
        elif y == 'benign':
            label.append([0])

def spiltDatast(input_ids, input_types, input_masks, label):
    # 随机打乱索引
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)  # 固定种子
    np.random.shuffle(random_order)
    print(random_order[:10])

    # 4:1 划分训练集和测试集
    input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    print("input_ids_train.shape:" + str(input_ids_train.shape))
    print("input_types_train.shape:" + str(input_types_train.shape))
    print("input_masks_train.shape:" + str(input_masks_train.shape))
    print("y_train.shape:" + str(y_train.shape))

    input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
    input_types_test = np.array([input_types[i] for i in random_order[int(len(input_ids) * 0.8):]])
    input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids) * 0.8):]])
    y_test = np.array([label[i] for i in random_order[int(len(input_ids) * 0.8):]])
    print("input_ids_test.shape:" + str(input_ids_test.shape))
    print("input_types_test.shape:" + str(input_types_test.shape))
    print("input_masks_test.shape:" + str(input_masks_test.shape))
    print("y_test.shape:" + str(y_test.shape))

    return input_ids_train, input_types_train, input_masks_train, y_train, input_ids_test, input_types_test, input_masks_test, y_test

