import argparse
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_utils import GetDataAndLabelsFromFiles, DatasetOptions, check_dir
from tld import get_tld
import random
import csv

LABEL_PHISH = 1
LABEL_LEGIT = 0


def get_args():
    parser = argparse.ArgumentParser(
        """Extracting Top-K Selected NGrams according tp selected scoring method.""")

    parser.add_argument("-d", "--dataset", type=DatasetOptions, default=DatasetOptions.grambeddings,
                        choices=list(DatasetOptions), help="input dataset name")

    parser.add_argument("-c", "--attack_char", type=str, default='-', help="character to add between tokens")

    return parser.parse_args()


def parseDomainInfo(sample):
    try:
        res = get_tld(sample, as_object=True)
        return res
    except:
        return None


def generate_adversarials(tokenizer, legitimate_samples, attack_char):
    adversarial_legitimates = list()

    for sample in legitimate_samples:

        if 'ARVADAPRESS' in sample:
            print("Debug me")

        domain_info = parseDomainInfo(sample)
        # If we could not be able to parse domain information of URL, then skip it.
        if domain_info is None:

            continue

        tokens_per_sample = tokenizer.tokenize(domain_info.domain)

        token_count = len(tokens_per_sample)
        # Since any token have a chance to contain '_' at the beginning character, remove the first letter.
        for index in range(0, token_count):
            token = tokens_per_sample[index]
            if token[0] == '▁':
                token = token[1:]
                tokens_per_sample[index] = token

        adversarial_domain = ""
        if token_count < 2:
            continue
        elif token_count > 2:
            # Randomly select where to add this hyphen symbol
            insertion_index = random.randint(1, token_count - 1)
            for i in range(0, token_count):
                if i != insertion_index:
                    adversarial_domain = adversarial_domain + tokens_per_sample[i]
                else:
                    adversarial_domain = adversarial_domain + attack_char + tokens_per_sample[i]
        else:
            adversarial_domain = attack_char.join(tokens_per_sample)

        # If the adversarial version is same with the original domain then skip it
        if domain_info.domain == adversarial_domain:
            continue

        if attack_char not in adversarial_domain:
            continue

        if '--' in adversarial_domain and '--' not in domain_info.domain:
            print("Double hyphen is found after adding hyphen into domain and replaced with single hyphen")
            adversarial_domain = adversarial_domain.replace('--', '-')
            continue

        adversarial_url = sample.replace(domain_info.domain, adversarial_domain)

        if attack_char not in adversarial_url:
            continue

        if sample == adversarial_url:
            continue

        adversarial_legitimates.append(adversarial_url)

        progress = len(adversarial_legitimates)
        if (progress % 100) == 0:
            progress_percent = (progress / len(legitimate_samples)) * 100
            print("Progress : ", progress_percent)

    return adversarial_legitimates


def Process(args):
    print("###### Adversarial Attack Generation is Started with parameters given below #####")
    print(args)
    print('####################################### Loading Dataset  #######################################')
    train_file = 'data/' + args.dataset.value + '/train.csv'
    val_file = 'data/' + args.dataset.value + '/test.csv'
    train_samples, train_labels = GetDataAndLabelsFromFiles(train_file, convert_to_array=False, label_legit=LABEL_LEGIT,
                                                            label_phish=LABEL_PHISH)
    val_samples, val_labels = GetDataAndLabelsFromFiles(val_file, convert_to_array=False, label_legit=LABEL_LEGIT,
                                                        label_phish=LABEL_PHISH)

    output_directory = 'data/' + args.dataset.value + '_Adversarial_Full/'
    print("##### Note that all results will be saved under ", output_directory, " directory ######")
    check_dir(output_directory)

    print("##### Creating Training Phish File from inputs #####")
    indexes_phish = [idx for idx, element in enumerate(train_labels) if element == LABEL_PHISH]
    samples_phish = [train_samples[i] for i in indexes_phish]
    train_phish_file = output_directory + 'TRP.csv'
    with open(train_phish_file, 'w', encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for url in samples_phish:
            line = [1, url]
            writer.writerow(line)
    print("Done")

    print("##### Creating Training Legitimate File from inputs #####")
    indexes_legit = [idx for idx, element in enumerate(train_labels) if element == LABEL_LEGIT]
    samples_legit = [train_samples[i] for i in indexes_legit]
    train_legit_file = output_directory + 'TRL.csv'
    with open(train_legit_file, 'w', encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for url in samples_legit:
            line = [0, url]
            writer.writerow(line)
    print("Done")

    print("##### Creating Test Phish File from inputs #####")
    indexes_phish_val = [idx for idx, element in enumerate(val_labels) if element == LABEL_PHISH]
    samples_phish_val = [val_samples[i] for i in indexes_phish_val]
    test_phish_file = output_directory + 'TEP.csv'
    with open(test_phish_file, 'w', encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for url in samples_phish_val:
            line = [1, url]
            writer.writerow(line)
    print("Done")

    print("##### Creating Test Legitimate File from inputs #####")
    indexes_legit = [idx for idx, element in enumerate(val_labels) if element == LABEL_LEGIT]
    samples_legit = [val_samples[i] for i in indexes_legit]
    test_legit_file = output_directory + 'TEL.csv'
    with open(test_legit_file, 'w', encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for url in samples_legit:
            line = [0, url]
            writer.writerow(line)
    print("Done")

    print("##### Reading Legitimate Samples from TEP.csv #####")
    ph_samples, ph_labels = GetDataAndLabelsFromFiles(test_phish_file, convert_to_array=False,
                                                      label_legit=LABEL_LEGIT, label_phish=LABEL_PHISH)
    tel_samples, tel_labels = GetDataAndLabelsFromFiles(test_legit_file, convert_to_array=False,
                                                        label_legit=LABEL_LEGIT, label_phish=LABEL_PHISH)
    print("Done")
    print('##### Initializing AutoTransformer to Tokenize Each Legitimate URL in Dataset')
    tokenizer = AutoTokenizer.from_pretrained(r'charbert-bert-wiki')
    print("Done")
    adversarial_phish = list()
    print('##### Starting to generate adversarial versions of legitimate samples')
    adversarial_phish = generate_adversarials(tokenizer, ph_samples, args.attack_char)
    print("##### Adversarial attacks are generated, well done... #####")
    test_converted_phish_file = output_directory + 'TEP_converted.csv'
    print("##### Saving this converted urls to ", test_converted_phish_file, " file")
    with open(test_converted_phish_file, 'w', encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for url in adversarial_phish:
            line = [1, url]
            writer.writerow(line)
    print("Done")

    test_augmented_mode_1_file = output_directory + 'test_aug_mode1.csv'
    print("##### Saving full augmented test data with labels to ", test_augmented_mode_1_file, " file")
    with open(test_augmented_mode_1_file, 'w', encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        '''
        Used to create an enhanced validation dataset: 
        which includes legitimate samples and adversarially processed legitimate samples
        '''
        aug_samples = tel_samples + adversarial_phish
        aug_labels = [0] * len(tel_samples) + [1] * len(adversarial_phish)
        for label, url in zip(aug_labels, aug_samples):
            line = [label, url]
            writer.writerow(line)
    print("Done")

    test_augmented_mode_2_file = output_directory + 'test_aug_mode2.csv'
    print("##### Saving full augmented test data with labels to ", test_augmented_mode_2_file, " file")
    with open(test_augmented_mode_2_file, 'w', encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        '''
        Used to create enhanced verification data sets: 
        legitimate samples, adversarially processed legitimate samples, and a subset of malicious samples
        '''
        half_p = int(len(samples_phish_val) / 2)
        aug_samples = tel_samples + adversarial_phish + samples_phish_val[:half_p]
        aug_labels = [0] * len(tel_samples) + [1] * len(adversarial_phish) + [1] * len(
            samples_phish_val[:half_p])
        for label, url in zip(aug_labels, aug_samples):
            line = [label, url]
            writer.writerow(line)
    print("Done")

    print("##### Reading Legitimate Samples from TRL_converted.csv #####")
    trl_samples, trl_labels = GetDataAndLabelsFromFiles(train_legit_file, convert_to_array=False,
                                                        label_legit=LABEL_LEGIT, label_phish=LABEL_PHISH)
    trp_samples, trp_labels = GetDataAndLabelsFromFiles(train_phish_file, convert_to_array=False,
                                                        label_legit=LABEL_LEGIT, label_phish=LABEL_PHISH)
    print("Done")
    adversarial_phish = list()
    print('##### Starting to generate adversarial versions of legitimate samples')
    adversarial_phish = generate_adversarials(tokenizer, trp_samples, args.attack_char)
    print("##### Adversarial attacks are generated, well done... #####")
    train_converted_phish_file = output_directory + 'TRP_converted.csv'
    print("##### Saving this converted urls to ", train_converted_phish_file, " file")
    with open(train_converted_phish_file, 'w', encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file)
        for url in adversarial_phish:
            line = [1, url]
            writer.writerow(line)
    print("Done")

    train_augmented_mode_file = output_directory + 'train_aug_mode.csv'
    print("##### Saving full augmented training data with labels to ", test_augmented_mode_2_file, " file")
    with open(train_augmented_mode_file, 'w', encoding='utf-8', newline="") as csv_file:
        '''
        Create an enhanced training data set
        This line of code merges the original training sample list train_samples and the adversarial sample 
        list adversarial_legitimates into an enhanced sample list aug_samples
        '''
        writer = csv.writer(csv_file, delimiter='\t')
        aug_samples = train_samples + adversarial_phish

        train_labels[train_labels == LABEL_LEGIT] = 0
        aug_labels = train_labels + [1] * len(adversarial_phish)

        c = list(zip(aug_samples, aug_labels))
        random.shuffle(c)
        adv_samples, adv_labels = zip(*c)
        aug_samples = list(adv_samples)
        import numpy as np
        aug_labels = np.array(adv_labels)
        for label, url in zip(aug_labels, aug_samples):
            line = [label, url]
            writer.writerow(line)
    print("Done")


if __name__ == "__main__":
    opt = get_args()
    Process(opt)
