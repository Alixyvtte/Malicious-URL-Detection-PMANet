# Malicious-URL-Detection-PMANet
This is an implementation of the [paper](https://arxiv.org/abs/2311.12372) - "Malicious URL Detection via Pretrained Language Model-Guided Multi-Level Feature Attention Network"

## Overview

The proposed framework, which is based on a character-aware pretrained language model and includes a novel feature concatenation method, a channel attention mechanism with pyramid pooling, demonstrates good performance in the detection of malicious URLs.

## Model Designs

![Graphforgithub.png](https://s2.loli.net/2023/11/08/fF3KSqErtCzQVk5.png)

## Directory Guide

| Folder/File Name         | Purpose                                                      |
| ------------------------ | ------------------------------------------------------------ |
| /Data                    | Where the dataset files are stored under their unique dataset names. Contains all the dataset we used in our paper. |
| /character_bert_wiki     | Where the pre-trained model CharBERT based on BERT is stored. You can also download using  [this](https://drive.google.com/file/d/1rF5_LbA2qIHuehnNepGmjz4Mu6OqEzYT/view?usp=sharing) link. |
| attention.py             | Contains the proposed channel attention model.               |
| data_processing.py       | Contains functions for preprocessing the dataset, including converting URL strings into BERT's input format and splitting it into training and validation sets. |
| Model_PMA.py             | Where the proposed Malicious URL Detection PMANet architecture is stored. |
| Model_CharBERT.py        | This file contains the original CharBERT model part.         |
| Train.py                 | Trains the model according to given parameters, You need to modify the "*pre-trained model path*" and "*dataset path*". |
| Test_binary.py           | Tests the pre-trained model in the binary classification experiments according to given parameters. |
| Test_Multiple.py         | Tests the pre-trained model in the multiple classification experiment according to given parameters. |
| bert_utils.py            | Some basic source codes of original bert which are used in the CharBERT structure. |
| adverserial_generator.py | Where the functions to  generate adversarial examples by using specified samples are stored. |

## Requirements

```
Python 3.8
Pytorch 2.0.0
```

## Usage

In all datasets for training or testing, each line includes the label and the URL text string following the template:

`<URL String><label>`

**Example:**

The URL label is either +1 (Malicious) or 0  (Benign).

```
http://www.exampleforbenign.com/urlpath/ 0
http://www.exampleformalicious.com/urlpath/ 1
```

You can adjust the hyper parameters for your specific computing device.  Also can further tuning the  `batch_size`, `learning_rate` and `train_epoch.`  To train the model, use the '`Train.py`' file, and you can find descriptions of the training parameters that influence the training phase below.

| **Parameter** | Description                                                  | **Default**      |
| :------------ | :----------------------------------------------------------- | :--------------- |
| dataset       | An enumeration value specifies the dataset used for training the model. | malware_urls.txt |
| model_path    | Output folder to save the model                              | /model.pth       |
| batch_size    | Number of URLs in each training batch                        | 64               |
| num_epoch     | Number of training epochs                                    | 3                |
| lr            | Learning rate                                                | 2e-5             |
| pad_size      | The maximum number of characters in a URL. The URL is either truncted or padded with a `<PADDING>` character to reach this length. (Based on preliminary analysis,  the maximum text length is 38, taking 32 to cover 99%) | 200              |

