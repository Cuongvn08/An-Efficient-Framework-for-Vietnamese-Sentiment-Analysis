import re
import pickle

import string
import codecs
import pandas as pd
import numpy as np
from torchtext import data
from torchtext.data import Dataset, Example
from sklearn.model_selection import train_test_split
from utils.utils import str2list
from pyvi import ViTokenizer
import torch


def preprocessing(text, embeddings_index):
    text_ori = text

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
    texts = [t for t in texts if embeddings_index.get(t) is not None]
    text = u' '.join(texts)

    if len(texts) < 5:
        text = None

    return text

def load_word_embedding(embedding_file):
    #print('Loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(embedding_file, encoding='utf-8')
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def filter_glove_emb(word_dict, embedding_index):
    # filter embeddings
    dim = 300
    scale = np.sqrt(3.0 / dim)
    vectors = np.random.uniform(-scale, scale, [len(word_dict), dim])

    for word in word_dict.keys():
        embedding_vector = embedding_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            index = list(word_dict.keys()).index(word)
            vectors[index] = embedding_vector
        #else:
        #    print(word)
    return vectors

class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""

    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex

'''
def tokenizer(text):
    text = " ".join(re.findall("[a-zA-Z]+", text))
    return text.split(' ')
'''

def load_data(type, session, **kwargs):
    if type == 'csv':
        return load_csv_data(session, kwargs)


def load_csv_data(session, kwargs):
    print('session', session)

    is_train = kwargs['is_train']
    if is_train:
        # get setting
        train_file = session['train_file']
        validate_file = session.get('validate_file', None)
        test_file = session['test_file']
        text_column = str(session['text_column'])
        label_column = str(session['label_column'])
        use_cols = [text_column, label_column]
        batch_size = int(session['batch_size'])
        val_ratio = float(session.get('val_ratio', 0.2))
        fix_length = session.get('fix_length', None)
        if fix_length:
            fix_length = int(fix_length)
        pretrained_embedding = session.get('pretrained_embedding', '/data/cuong/data/nlp/embedding/cc.vi.300.vec')
        sep = session.get('sep', ',')
        nrows = None if str(session['nrows']) == 'None' else int(session['nrows'])

        # load pretrained embedding
        embeddings_index = load_word_embedding(pretrained_embedding)

        # load data
        train_df = pd.read_csv(train_file, usecols=use_cols, sep=sep, nrows=nrows)
        test_df  = pd.read_csv(test_file, usecols=use_cols, sep=sep, nrows=nrows)

        print('train_df={}; counts={}'.format(train_df.shape, train_df[label_column].value_counts()))
        print('test_df={};  counts={}'.format(test_df.shape,  test_df[label_column].value_counts()))

        train_df.dropna(subset=[text_column], inplace=True)
        test_df.dropna(subset=[text_column], inplace=True)


        if validate_file:
            valid_df = pd.read_csv(validate_file, usecols=use_cols, sep=sep)
        else:
            train_df, valid_df = train_test_split(train_df, test_size=val_ratio, random_state=42, shuffle=True, stratify=train_df[label_column])

        train_df[text_column] = train_df[text_column].apply(lambda x:preprocessing(x, embeddings_index))
        valid_df[text_column] = valid_df[text_column].apply(lambda x:preprocessing(x, embeddings_index))
        test_df[text_column]  = test_df[text_column].apply(lambda x:preprocessing(x, embeddings_index))

        train_df.drop_duplicates(subset=text_column, keep = 'first', inplace = True)
        train_df.dropna(subset=[text_column], inplace=True)
        train_df = train_df.reset_index()

        valid_df.drop_duplicates(subset=text_column, keep = 'first', inplace = True)
        valid_df.dropna(subset=[text_column], inplace=True)
        valid_df = valid_df.reset_index()

        test_df.drop_duplicates(subset=text_column, keep = 'first', inplace = True)
        test_df.dropna(subset=[text_column], inplace=True)
        test_df = test_df.reset_index()


        # build vocab
        TEXT = data.Field(sequential=True, lower=True, include_lengths=True, fix_length=fix_length)
        LABEL = data.LabelField(sequential=False, use_vocab=False)

        fields = {}
        fields[text_column] = TEXT
        fields[label_column] = LABEL

        train_dataset = DataFrameDataset(train_df, fields=fields)
        valid_dataset = DataFrameDataset(valid_df, fields=fields)
        test_dataset = DataFrameDataset(test_df, fields=fields)

        TEXT.build_vocab(train_dataset, valid_dataset)
        print('TEXT.vocab', len(TEXT.vocab))
        print(TEXT.vocab.freqs.most_common(20))

        # set vocab vectors
        vectors = filter_glove_emb(TEXT.vocab.stoi, embeddings_index)
        TEXT.vocab.set_vectors(TEXT.vocab.stoi, torch.from_numpy(vectors), 300)
        print('TEXT.vocab.Vectors', len(TEXT.vocab.vectors))
        print(TEXT.vocab.vectors)

        train_iter = data.BucketIterator(train_dataset,
                                         shuffle=True,
                                         batch_size=batch_size,
                                         repeat=False)
        valid_iter = data.BucketIterator(valid_dataset,
                                        shuffle=False,
                                        batch_size=batch_size,
                                        repeat=False)
        test_iter = data.BucketIterator(test_dataset,
                                        shuffle=False,
                                        batch_size=batch_size,
                                        repeat=False)

        del embeddings_index
        del train_df, test_df

        return train_iter, valid_iter, test_iter, TEXT
    else:
        decode_file = session['decode_file']
        vocab_file = session['vocab_file']
        text_column = str2list(session['text_column'])
        if len(text_column) != 1:
            raise Exception('only 1 text column needed, found %d: %s'% (len(text_column), ','.join(text_column)))
        batch_size = int(session['batch_size'])
        fix_length = session.get('fix_length', None)
        if fix_length:
            fix_length = int(fix_length)

        vocab = pickle.load(open(vocab_file, 'rb'))
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, fix_length=fix_length)
        TEXT.vocab = vocab
        fields = {}
        for column in text_column:
            fields[column] = TEXT

        decode_df = pd.read_csv(decode_file, usecols=text_column)
        decode_dataset = DataFrameDataset(decode_df, fields=fields)
        decode_iter = data.BucketIterator(decode_dataset,
                                         shuffle=False,
                                         batch_size=batch_size,
                                         repeat=False)
        return decode_iter, TEXT
