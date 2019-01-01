import torch
import torchtext
import spacy
import time
from torchtext import *
import torch.nn as nn

nlp = spacy.load('en')

def tokenizer(text):
    return [tok.text for tok in nlp.tokenizer(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

st = time.time()

# splitting the data into train, test and validation
train, test = data.TabularDataset.splits(
        path='C:/Users/Shubham/Desktop/data/quora', train='train.csv',
        test='test.csv', format='csv',
        fields=[('Text', TEXT), ('Label', LABEL)])

# using pretrained embeddings
TEXT.build_vocab(train, vectors="glove.6B.100d")

train_iter, test_iter = data.Iterator.splits(
        (train, test), sort_key=lambda x: len(x.Text),
        batch_sizes=(32, 256, 256), device=-1)

vocab = TEXT.vocab

embed = nn.Embedding(len(vocab), 100)
embed.weight.data.copy_(vocab.vectors)

end = time.time()
total_time = end - st

print ("Total time taken %s seconds" %total_time)