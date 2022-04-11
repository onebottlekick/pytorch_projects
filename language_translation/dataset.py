from functools import partial
import pickle

import spacy
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

from utils import tokenize
from configs import DEVICE, BATCH_SIZE

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

SRC = Field(
    tokenize=partial(tokenize, nlp=spacy_de, reverse=True),
    init_token='<sos>',
    eos_token='<eos>',
    lower=True
)

TRG = Field(
    tokenize=partial(tokenize, nlp=spacy_en),
    init_token='<sos>',
    eos_token='<eos>',
    lower=True
)

train_data, val_data, test_data = Multi30k.splits(root='datasets', exts=('.de', '.en'), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    datasets=(train_data, val_data, test_data),
    batch_size=BATCH_SIZE,
    device=DEVICE
)