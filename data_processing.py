from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np

def dataPreprocess(filename, input_ids, input_types, input_masks, label, urltype):
    """
    Preprocess data from a file containing URLs and labels.

    :param filename: The path to the data file.
    :param input_ids: List to store input char IDs.
    :param input_types: List to store segment IDs.
    :param input_masks: List to store attention masks.
    :param label: List to store labels.
    :param urltype: The type of URL (0 for benign, 1 for malware).
    :return: None
    """
    pad_size = 200
    # Also known as max_len (Based on preliminary analysis,
    # the maximum text length is 38, taking 32 to cover 99%)
    bert_path = "charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # Initialize the tokenizer
    with open(filename, encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f)):
            x1 = l.strip()
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1 + ["[SEP]"]

            # Get input_id, seg_id, att_mask
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)

            # Pad if short, truncate if long
            if len(ids) < pad_size:
                types = types + [1] * (pad_size - len(ids))  # Set segment to 1 for the masked part
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
    """
    Preprocess data from a CSV file containing URLs and labels.

    :param filename: The path to the CSV data file.
    :param input_ids: List to store input char IDs.
    :param input_types: List to store segment IDs.
    :param input_masks: List to store attention masks.
    :param label: List to store labels.
    :return: None
    """
    pad_size = 200
    bert_path = "/hy-tmp/charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # Initialize the tokenizer

    data = pd.read_csv(filename, encoding='utf-8')
    for i, row in tqdm(data.iterrows(), total=len(data)):
        x1 = row['url']  # Replace with the column name in your CSV file where the text data is located
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

        # Get input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(ids))
        masks = [1] * len(ids)

        # Pad if short, truncate if long
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))  # Set segment to 1 for the masked part
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
    """
    Split the dataset into training and testing sets.

    :param input_ids: List of input character IDs.
    :param input_types: List of segment IDs.
    :param input_masks: List of attention masks.
    :param label: List of labels.
    :return: Split datasets for training and testing.
    """
    # Randomly shuffle the indices
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)  # Fix the seed
    np.random.shuffle(random_order)
    print(random_order[:10])

    # Split the dataset into 80% training and 20% testing
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

