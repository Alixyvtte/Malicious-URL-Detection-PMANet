import os

import numpy as np
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
import tensorflow


def GetDataAndLabelsFromFiles(file_path, max_sequence_length=None, convert_to_array=True, label_phish=1, label_legit=0,
                              label_defacement=2, label_phishing=3,
                              split_char='\t'):
    samples = []
    labels = []
    with open(file_path, encoding="utf8", errors='ignore') as fp:
        line = fp.readline()
        le = LabelEncoder()
        while line:
            if line.__contains__('\n'):
                line = line.replace('\n', '')
                line = line.replace(' ', '')

            parts = line.split(split_char)
            label = int(parts[0].replace('"', ''))
            if label == -1:
                label = label_legit
            elif label == 1:
                label = label_phish
            #elif label == 2:
                #label = label_defacement
            #elif label == 3:
                #label = label_phishing

            sample = parts[1]

            if len(sample) < 8:
                line = fp.readline()
                continue

            if max_sequence_length is not None:
                sample = sample[:max_sequence_length]

            # sample = sample[1:]   # Remove first char from url because it is "
            # sample = sample[:-1]  # Remove last  char from url because it is "
            # sample = sample.lower()
            samples.append(sample)
            labels.append(label)

            line = fp.readline()

    if convert_to_array:
        labels = np.asarray(labels, np.int32)
    # labels = le.fit_transform(labels)
    # labels = tensorflow.keras.utils.to_categorical(labels)
    return samples, labels


def GetSplitedDataAndLabelsFromFiles(file_path):
    netlocs = []
    paths = []
    labels = []
    with open(file_path, encoding="utf8", errors='ignore') as fp:
        line = fp.readline()
        while line:
            if line.__contains__('\n'):
                line = line.replace('\n', '')
                line = line.replace(' ', '')

            parts = line.split(",")
            label = int(parts[0].replace('"', '')) - 1
            sample = parts[1]
            sample = sample[1:]  # Remove first char from url because it is "
            sample = sample[:-1]  # Remove last  char from url because it is "
            # sample = sample.lower()
            url_info = urlparse(sample)
            netlocs.append(url_info.netloc)
            paths.append(url_info.path)
            labels.append(label)
            line = fp.readline()

    labels = np.asarray(labels, np.int32)
    return netlocs, paths, labels


def CreateModelFileNameFromArgs(opt):
    model_name = "model"
    model_name = model_name + "_data_" + str(opt.dataset.value).replace('/', '-')
    model_name = model_name + "_n1_" + str(opt.ngram_1)
    model_name = model_name + "_n2_" + str(opt.ngram_2)
    model_name = model_name + "_n3_" + str(opt.ngram_3)
    model_name = model_name + "_mf_" + str(opt.max_features)
    model_name = model_name + "_maxdf_" + str(opt.max_df)
    model_name = model_name + "_mindf_" + str(opt.min_df)
    model_name = model_name + "_msl_" + str(opt.max_seq_len)
    model_name = model_name + "_dim_" + str(opt.embed_dim)
    model_name = model_name + "_attn_" + str(opt.attn_width)
    model_name = model_name + "_rnn_" + str(opt.rnn_cell_size)
    model_name = model_name + "_bs_" + str(opt.batch_size)
    model_name = model_name + "_ep_" + str(opt.epochs)
    model_name = model_name + "_is_" + str(opt.case_insensitive)
    return model_name


'''
def SaveResults(best_epoch_index, best_train_accu, best_train_loss, best_valid_accu, best_valid_loss,
                best_tp, best_fp, best_tn, best_fn, best_tpr, best_fpr, best_precision, best_recall, best_auc,
                opt, elapsed_time):
    # Saving results on CSV File
    csv_filename = "outputs/training/results.csv"
    # Check if file exists
    # dataset   ngram_1 ngram_2	ngram_3	max_features	max_df	min_df	max_seq_len	embed_dim	attn_width	rnn_cell_size	batch_size	epochs	best_epoch_index	best_train_acc	best_train_loss	best_val_acc	best_val_loss
    csv_content = list()
    csv_content.append(opt.dataset.value)  # Since it is an enum, we need to get its value
    csv_content.append(opt.ngram_1)
    csv_content.append(opt.ngram_2)
    csv_content.append(opt.ngram_3)
    csv_content.append(opt.max_features)
    csv_content.append(opt.max_df)
    csv_content.append(opt.min_df)
    csv_content.append(opt.max_seq_len)
    csv_content.append(opt.embed_dim)
    csv_content.append(opt.attn_width)
    csv_content.append(opt.rnn_cell_size)
    csv_content.append(opt.batch_size)
    csv_content.append(opt.warm_start)
    # Since it is an enum, we need to get its value
    csv_content.append(opt.warm_mode.value if opt.warm_start else 'None')
    csv_content.append(opt.epochs)
    csv_content.append(best_epoch_index)

    csv_content.append(opt.case_insensitive)

    csv_content.append(best_train_accu)
    csv_content.append(best_valid_accu)
    csv_content.append(best_precision)
    csv_content.append(best_recall)
    csv_content.append(best_auc)

    csv_content.append(elapsed_time)

    import csv
    with open(csv_filename, 'a+', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_content)
'''


def SaveMulticlassResults(
        best_epoch_index,
        best_train_loss,
        best_valid_loss,
        best_multiclass_auc,
        best_f1_score,
        opt,
        elapsed_time
):
    # Saving results to a CSV file
    csv_filename = "outputs/training/multiclass_results.csv"

    # Prepare the content to be saved in the CSV file
    csv_content = [
        opt.dataset.value,  # Since it is an enum, we need to get its value
        opt.ngram_1,
        opt.ngram_2,
        opt.ngram_3,
        opt.max_features,
        opt.max_df,
        opt.min_df,
        opt.max_seq_len,
        opt.embed_dim,
        opt.attn_width,
        opt.rnn_cell_size,
        opt.batch_size,
        opt.warm_start,
        opt.warm_mode.value if opt.warm_start else 'None',
        opt.epochs,
        best_epoch_index,
        opt.case_insensitive,
        best_train_loss,
        best_valid_loss,
        best_multiclass_auc,
        best_f1_score,
        elapsed_time
    ]

    import csv
    # Write to the CSV file
    with open(csv_filename, 'a+', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_content)


# Adding Boolean argument helper ref:https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


from enum import Enum


class DatasetOptions(Enum):
    grambeddings = 'grambeddings'
    grambeddings_adv = 'grambeddings_Adversarial'
    ebubekir = 'ebubekir'
    kd01 = 'kaggle/01-sid321axn'
    kd02 = 'kaggle/02-akshaya1508'
    kd03 = 'kaggle/03-siddharthkumar25'
    pdrcnn = 'PDRCNN'
    phishstorm = 'PhishStorm'
    udbjm = 'UDBJM'
    grambeddings_augmMode_not_trained = 'grambeddings_augmMode_not_trained'
    grambeddings_train_orig_valid_adv1 = 'grambeddings_train_orig_valid_adv1'
    grambeddings_train_orig_valid_adv2 = 'grambeddings_train_orig_valid_adv2'
    grambeddings_train_adv4_valid_adv1 = 'grambeddings_train_adv4_valid_adv1'
    grambeddings_train_adv4_valid_adv2 = 'grambeddings_train_adv4_valid_adv2'
    ISCXURL2016 = 'ISCXURL2016'


# check if dir exist if not create it
def check_dir(file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
