# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 13:00:37 2018

@author: Shubham
"""

import re
import logging
import en_core_web_sm
import numpy as np
import pandas as pd
import spacy
import torch
from torchtext import data
import time
from torchtext.vocab import GloVe, CharNGram 
import pickle

NLP = en_core_web_sm.load()
MAX_CHARS = 20000
VAL_RATIO = 0.2
LOGGER = logging.getLogger("toxic_dataset")


def tokenizer(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [x.text for x in NLP.tokenizer(comment) if x.text != " "]


def prepare_csv(seed=999):
    df_train = pd.read_csv(r"C:\Users\Shubham\Desktop\data\quora\train.csv")
    df_train["question_text"] = df_train.question_text.str.replace("\n", " ")
    idx = np.arange(df_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * VAL_RATIO)
    df_train.iloc[idx[val_size:], :].to_csv(
        r"C:\Users\Shubham\Desktop\data\quora\dataset_train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv(
        r"C:\Users\Shubham\Desktop\data\quora\dataset_val.csv", index=False)
    df_test = pd.read_csv(r"C:\Users\Shubham\Desktop\data\quora\test.csv")
    df_test["question_text"] = df_test.question_text.str.replace("\n", " ")
    df_test.to_csv(r"C:\Users\Shubham\Desktop\data\quora\dataset_test.csv", index=False)


def get_dataset(fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only supports all lower cases
        lower = True
    LOGGER.debug("Preparing CSV files...")
    prepare_csv()
    question = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        lower=lower
    )
    LOGGER.debug("Reading train csv file...")
    train, val = data.TabularDataset.splits(
        path=r"C:\Users\Shubham\Desktop\data\quora", 
        format='csv', skip_header=True,
        train='dataset_train.csv', validation='dataset_val.csv',
        fields=[
            ('qid', None),
            ('question_text', question),
            ('target', data.Field(
            use_vocab=False, sequential=False))
        ])
    LOGGER.debug("Reading test csv file...")
    test = data.TabularDataset(
        path=r"C:\Users\Shubham\Desktop\data\quora\dataset_test.csv", 
        format='csv', skip_header=True,
        fields=[
            ('id', None),
            ('question_text', question),
            ('target', data.Field(
            use_vocab=False, sequential=False))
        ])
    LOGGER.debug("Building vocabulary...")
    question.build_vocab(
        train, val, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    LOGGER.debug("Done preparing the datasets")
    return train, val, test


def get_iterator(dataset, batch_size, train=True, shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=0,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False)
    return dataset_iter

# return processed train, test and validation datasets
def load_dataset(train, val, test, batch_size):
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)
    
    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    
    train_iter = get_iterator(train, batch_size)
    val_iter = get_iterator(val, batch_size)
    test_iter = get_iterator(test, batch_size, train=False)
    
    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, val_iter, test_iter

if __name__ == "__main__":
    st = time.time()
    train, val, test = get_dataset()
    end = time.time()
    total_time = end - st
    print("%0.2f seconds" %total_time)
    
    TEXT, vocab_size, word_embeddings, train_iter, val_iter, test_iter = load_dataset(train, val, test, 32)
    
#    print ("Iterating over train set")
#    for examples in get_iterator(train, 32, train=True,
#            shuffle=True, repeat=False):
#        x = examples.question_text # (fix_length, batch_size) Tensor
#        y = torch.stack([
#            examples.target
#        ], dim=1)
#    
#    print("done")