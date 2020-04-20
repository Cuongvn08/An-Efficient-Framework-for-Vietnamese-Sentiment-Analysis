# nohup python -u main_bertpho.py >> main_bertpho.out &

import os
import sys
import sklearn
import numpy as np
import pandas as pd

import re

import xgboost as xgb
import tensorflow as tf
import torch

from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras import layers

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset
from utils.utils import class_eval, AverageMeter
from utils.utils import check_fields, print_progress, AverageMeter, class_eval, save_checkpoint, \
    load_checkpoint

from transformers import RobertaConfig
from transformers import RobertaModel
from transformers import BertTokenizer
from transformers import BertPreTrainedModel
from transformers import BertForSequenceClassification

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import pytz
from datetime import datetime
tz = pytz.timezone('Asia/Saigon')
#date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S ')

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import string
string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

import fairseq
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

from vncorenlp import VnCoreNLP

import argparse

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def strToBool(str):
    return str.lower() in ('true', 'yes', 'on', 't', '1')

parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)

parser.add_argument('--PATH_STAR_RATING_TRAIN', default='dataset/aivivn/train.csv')
parser.add_argument('--PATH_STAR_RATING_TEST', default='dataset/aivivn/test.csv')
parser.add_argument('--TEXT_COLUMN', default='discriptions')
parser.add_argument('--LABEL_COLUMN', default='mapped_rating')
parser.add_argument('--MAX_LEN', type=int, default=128)
parser.add_argument('--VNCORENLP', default='/data/cuong/data/nlp/embedding/vncorenlp/VnCoreNLP-1.1.1.jar')

parser.add_argument('--BERT_VOCAB', default='/data/cuong/data/nlp/embedding/PhoBERT_base_transformers/dict.txt')
parser.add_argument('--BERT_MODEL', default='/data/cuong/data/nlp/embedding/PhoBERT_base_transformers/model.bin')
parser.add_argument('--BERT_CONFIG', default='/data/cuong/data/nlp/embedding/PhoBERT_base_transformers/config.json')
parser.add_argument('--bpe-codes', default='/data/cuong/data/nlp/embedding/PhoBERT_base_transformers/bpe.codes')

parser.add_argument('--OUTPUT_DIR', default='output/aivivn/models/bertpho/')
parser.add_argument('--EPOCHS', type=int, default=10)

param = parser.parse_args()


def printBoth(message):
    date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S ')
    print(date_time + message)


def preprocessing(text):
    # remove duplicate characters such as đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

    # remove punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    # remove '_'
    text = text.replace('_', ' ')

    # remove numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # lower word
    text = text.lower()

    # replace special words
    replace_list = {
        'ô kêi': ' ok ', 'o kê': ' ok ',
        'kh ':' không ', 'kô ':' không ', 'hok ':' không ',
        'kp ': ' không phải ', 'kô ': ' không ', 'ko ': ' không ', 'khong ': ' không ', 'hok ': ' không ',
    }
    for k, v in replace_list.items():
        text = text.replace(k, v)

    # split texts
    texts = text.split()

    if len(texts) < 5:
        text = None

    return text


def segment(text):
    words = []
    for w in rdrsegmenter.tokenize(text):
        words = words + w
    text = ' '.join([i for i in words])
    return text


def bert_encode(text):
    text = segment(text)
    #text = '<s> ' + bpe.encode(text) + ' </s>'
    text = '<s> ' + text + ' </s>'
    text_ids = vocab.encode_line(text, append_eos=False, add_if_not_exist=False).long().tolist()

    return text_ids


def rpad(array, n=param.MAX_LEN):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n]
    extra = n - current_len
    return array + ([0] * extra)


class DatasetReview(Dataset):
    def __init__(self, reviews, ratings):
        self.dataset = [
            (
                rpad(bert_encode(reviews[i]), n=param.MAX_LEN),
                ratings[i],
            )
            for i in range(len(reviews))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, y = self.dataset[index]
        X = torch.tensor(X)
        y = torch.tensor(y)
        return X, y


def train_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    for batch, labels in generator:
        batch, labels = batch.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss, logits = model(input_ids=batch, labels=labels)
        err = lossfn(logits, labels)
        loss.backward()
        optimizer.step()

        acc, precision, recall, fscore, auc_score = class_eval(logits.data.cpu(), labels, pred_type=None)
        accuracies.update(acc, labels.size(0))
        precisions.update(precision, labels.size(0))
        recalls.update(recall, labels.size(0))
        fscores.update(fscore, labels.size(0))
        auc_scores.update(auc_score, labels.size(0))
        losses.update(loss.item(), labels.size(0))

        #print('batch', batch)
        #print('labels', labels)
        #print('logits', logits)

    return losses.avg, accuracies.avg, precisions.avg, recalls.avg, fscores.avg, auc_scores.avg


def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    with torch.no_grad():
        for batch, labels in generator:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            logits = model(input_ids=batch)[0]
            error = lossfn(logits, labels)

            acc, precision, recall, fscore, auc_score = class_eval(logits.data.cpu(), labels, pred_type=None)
            accuracies.update(acc, labels.size(0))
            precisions.update(precision, labels.size(0))
            recalls.update(recall, labels.size(0))
            fscores.update(fscore, labels.size(0))
            auc_scores.update(auc_score, labels.size(0))
            losses.update(error.item(), labels.size(0))

    return losses.avg, accuracies.avg, precisions.avg, recalls.avg, fscores.avg, auc_scores.avg


# load segmenter
printBoth('Loading segmenter ...')
rdrsegmenter = VnCoreNLP(param.VNCORENLP, annotators="wseg", max_heap_size='-Xmx500m')


# load BPE encoder
printBoth('\nLoading BPE encoder ...')
'''
parser_BPE = argparse.ArgumentParser()
parser_BPE.add_argument('--bpe-codes',
    default="/data/cuong/data/nlp/embedding/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args = parser_BPE.parse_args()
'''
bpe = fastBPE(param)
print('bpe', bpe)


# Load the dictionary
vocab = Dictionary()
vocab.add_from_file(param.BERT_VOCAB)
print('vocab', len(vocab))


# load data
printBoth('Reading csv files ...')
df_train = pd.read_csv(param.PATH_STAR_RATING_TRAIN, usecols=[param.TEXT_COLUMN, param.LABEL_COLUMN], sep=',', nrows=None)
df_test  = pd.read_csv(param.PATH_STAR_RATING_TEST, usecols=[param.TEXT_COLUMN, param.LABEL_COLUMN], sep=',', nrows=None)

printBoth('df_train={}; counts={}'.format(df_train.shape, df_train[param.LABEL_COLUMN].value_counts()))
printBoth('df_test={};  counts={}'.format(df_test.shape,  df_test[param.LABEL_COLUMN].value_counts()))

df_train.dropna(subset=[param.TEXT_COLUMN], inplace=True)
df_test.dropna(subset=[param.TEXT_COLUMN], inplace=True)

df_train[param.TEXT_COLUMN] = df_train[param.TEXT_COLUMN].apply(lambda x:preprocessing(x))
df_test[param.TEXT_COLUMN]  = df_test[param.TEXT_COLUMN].apply(lambda x:preprocessing(x))

df_train.drop_duplicates(subset=param.TEXT_COLUMN, keep = 'first', inplace = True)
df_train.dropna(subset=[param.TEXT_COLUMN], inplace=True)
df_train = df_train.reset_index()

df_test.drop_duplicates(subset=param.TEXT_COLUMN, keep = 'first', inplace = True)
df_test.dropna(subset=[param.TEXT_COLUMN], inplace=True)
df_test = df_test.reset_index()

# prepare training data
printBoth('Preparing bert data for training ...')
reviews = list(df_train[param.TEXT_COLUMN])
ratings = df_train[param.LABEL_COLUMN].values

train_reviews, val_reviews, train_ratings, val_ratings =  \
                sklearn.model_selection.train_test_split(reviews, ratings, test_size=0.2, random_state=42, shuffle=True, stratify=ratings)
test_reviews = list(df_test[param.TEXT_COLUMN])
test_ratings = df_test[param.LABEL_COLUMN].values

printBoth('Preparing DatasetReview ...')
dataset_train = DatasetReview(train_reviews, train_ratings)
dataset_val   = DatasetReview(val_reviews, val_ratings)
dataset_test  = DatasetReview(test_reviews, test_ratings)

# build model
printBoth('Building BERT model ...')
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = RobertaModel.from_pretrained(Config.BERT_MODEL, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(input_ids)

        '''
        outputs 2
        outputs torch.Size([32, 64, 768])
        outputs torch.Size([32, 768])
        '''

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


printBoth('Loading RobertaConfig ...')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = RobertaConfig.from_pretrained(param.BERT_CONFIG)
config.num_labels = 2
model = BertForSequenceClassification(config=config)

model = model.to(DEVICE)
lossfn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

if not os.path.exists(param.OUTPUT_DIR):
    os.makedirs(param.OUTPUT_DIR)

best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_auc = 0
for epoch in range(param.EPOCHS):
    print('\n')

    # train
    losses, accuracies, precisions, recalls, fscores, auc_scores = \
                    train_one_epoch(model, lossfn, optimizer, dataset_train, batch_size=32)
    printBoth('[TRAIN] epoch={}; losses={:0.5}; accuracies={:0.5}; precisions={:0.5}; recalls={:0.5}; fscores={:0.5}; auc_scores={:0.5}'.\
                format(epoch, losses, accuracies, precisions, recalls, fscores, auc_scores))

    # valid
    losses, accuracies, precisions, recalls, fscores, auc_scores = \
                    evaluate_one_epoch(model, lossfn, optimizer, dataset_val, batch_size=32)
    printBoth('[VALID] epoch={}; losses={:0.5}; accuracies={:0.5}; precisions={:0.5}; recalls={:0.5}; fscores={:0.5}; auc_scores={:0.5}'.\
                format(epoch, losses, accuracies, precisions, recalls, fscores, auc_scores))

    is_best = (auc_scores > best_auc) if config.num_labels == 2 else (accuracies > best_acc)
    best_acc1 = max(accuracies, best_acc)
    best_precision = max(precisions, best_precision)
    best_recall = max(recalls, best_recall)
    best_f1 = max(fscores, best_f1)
    best_auc = max(auc_scores, best_auc)

    # test
    losses, accuracies, precisions, recalls, fscores, auc_scores = \
                    evaluate_one_epoch(model, lossfn, optimizer, dataset_test, batch_size=32)
    printBoth('[TEST] epoch={}; losses={:0.5}; accuracies={:0.5}; precisions={:0.5}; recalls={:0.5}; fscores={:0.5}; auc_scores={:0.5}'.\
                format(epoch, losses, accuracies, precisions, recalls, fscores, auc_scores))

    # save the best model if any
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict()
    }, is_best, param.OUTPUT_DIR)


printBoth('The end.')
