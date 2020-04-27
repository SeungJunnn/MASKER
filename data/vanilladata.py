import os
from os import listdir
from os.path import isfile, join
import random
import csv
import numpy as np
import torch
import json
from utils import *
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

def NewsDataload(tokenizer, sample_classes=[i for i in range(20)], max_len=512):
    if isfile('train_news_tokens.npy') and isfile('test_news_tokens.npy') and isfile('train_news_labels.npy') and isfile('test_news_labels.npy'):
        train_news_tokens = np.load('train_news_tokens.npy')
        train_news_labels = np.load('train_news_labels.npy')
        test_news_tokens = np.load('test_news_tokens.npy')
        test_news_labels = np.load('test_news_labels.npy')
    else:
        train_data=[]
        test_data=[]
        train_label=[]
        test_label=[]

        f = open('train.csv', encoding='utf-8')
        g = open('test.csv', encoding='utf-8')
        for line in f:
            line = line.split(',')
            newsfile = open(line[0], encoding='utf-8', errors='ignore')
            next(newsfile)
            text = newsfile.read()
            tokens = tokenization(tokenizer, text, max_len)
            label = int(line[1])
            train_data.append(tokens)
            train_label.append(label)
        for line in g:
            line = line.split(',')
            newsfile = open(line[0], encoding='utf-8', errors='ignore')
            next(newsfile)
            text = newsfile.read()
            tokens = tokenization(tokenizer, text, max_len)
            label = int(line[1])
            test_data.append(tokens)
            test_label.append(label)

        assert len(train_data) == len(train_label)
        assert len(test_data) == len(test_label)

        train_news_tokens = torch.zeros((len(train_data),train_data[0].shape[0]), dtype=torch.long)
        train_news_labels = torch.zeros((len(train_label),1), dtype=torch.long)
        test_news_tokens = torch.zeros((len(test_data),test_data[0].shape[0]), dtype=torch.long)
        test_news_labels = torch.zeros((len(test_label),1), dtype=torch.long)

        for i in range(len(train_data)):
            train_news_tokens[i]=train_data[i]
            train_news_labels[i]=train_label[i]
        for i in range(len(test_data)):
            test_news_tokens[i]=test_data[i]
            test_news_labels[i]=test_label[i]

        np.save('train_news_tokens', train_news_tokens.numpy())
        np.save('train_news_labels', train_news_labels.numpy())
        np.save('test_news_tokens', test_news_tokens.numpy())
        np.save('test_news_labels', test_news_labels.numpy())

    train_news_tokens = torch.from_numpy(np.load('train_news_tokens.npy'))
    train_news_labels = torch.from_numpy(np.load('train_news_labels.npy'))
    test_news_tokens = torch.from_numpy(np.load('test_news_tokens.npy'))
    test_news_labels = torch.from_numpy(np.load('test_news_labels.npy'))

    train_data=[]
    test_data=[]

    for i in range(len(train_news_labels)):
        if train_news_labels[i].item() in sample_classes:
            train_data.append([train_news_tokens[i], train_news_labels[i]])
    for i in range(len(test_news_labels)):
        if test_news_labels[i].item() in sample_classes:
            test_data.append([test_news_tokens[i], test_news_labels[i]])


    return train_data, test_data

def ReviewDataload(tokenizer, sample_classes=[i for i in range(50)], max_len=512):
    if isfile('train_review_tokens.npy') and isfile('test_review_tokens.npy') and isfile('train_review_labels.npy') and isfile('test_review_labels.npy'):
        train_review_tokens = np.load('train_review_tokens.npy')
        train_review_labels = np.load('train_review_labels.npy')
        test_review_tokens = np.load('test_review_tokens.npy')
        test_review_labels = np.load('test_review_labels.npy')
        train_idx = np.load('train_idx.npy')
    else:
        fn = '50EleReviews.json'
        with open(fn, 'r') as infile:
                docs = json.load(infile)
        X = docs['X']
        y = np.asarray(docs['y'])
        train_data=[]
        test_data=[]
        train_label=[]
        test_label=[]

        train_idx, test_idx = splitindex(X,y,[i for i in range(50)])
        for idx in train_idx:
            tokens = tokenization(tokenizer,X[idx],max_len)
            label = int(y[idx])
            train_data.append(tokens)
            train_label.append(label)
        for idx in test_idx:
            tokens = tokenization(tokenizer,X[idx],max_len)
            label = int(y[idx])
            test_data.append(tokens)
            test_label.append(label)

        assert len(train_data) == len(train_label)
        assert len(test_data) == len(test_label)

        train_review_tokens = torch.zeros((len(train_data),train_data[0].shape[0]), dtype=torch.long)
        train_review_labels = torch.zeros((len(train_label),1), dtype=torch.long)
        test_review_tokens = torch.zeros((len(test_data),test_data[0].shape[0]), dtype=torch.long)
        test_review_labels = torch.zeros((len(test_label),1), dtype=torch.long)

        for i in range(len(train_data)):
            train_review_tokens[i]=train_data[i]
            train_review_labels[i]=train_label[i]
        for i in range(len(test_data)):
            test_review_tokens[i]=test_data[i]
            test_review_labels[i]=test_label[i]

        np.save('train_review_tokens', train_review_tokens.numpy())
        np.save('train_review_labels', train_review_labels.numpy())
        np.save('test_review_tokens', test_review_tokens.numpy())
        np.save('test_review_labels', test_review_labels.numpy())
        np.save('train_idx', np.asarray(train_idx))

    train_review_tokens = torch.from_numpy(np.load('train_review_tokens.npy'))
    train_review_labels = torch.from_numpy(np.load('train_review_labels.npy'))
    test_review_tokens = torch.from_numpy(np.load('test_review_tokens.npy'))
    test_review_labels = torch.from_numpy(np.load('test_review_labels.npy'))

    train_data=[]
    test_data=[]

    for i in range(len(train_review_labels)):
        if train_review_labels[i].item() in sample_classes:
            train_data.append([train_review_tokens[i], train_review_labels[i]])
    for i in range(len(test_review_labels)):
        if test_review_labels[i].item() in sample_classes:
            test_data.append([test_review_tokens[i], test_review_labels[i]])

    return train_data, test_data

def IMDBDataload(tokenizer, sample_classes=[0,1], max_len=512):
    if isfile('train_imdb_tokens.npy') and isfile('test_imdb_tokens.npy') and isfile('train_imdb_labels.npy') and isfile('test_imdb_labels.npy'):
        train_imdb_tokens = np.load('train_imdb_tokens.npy')
        train_imdb_labels = np.load('train_imdb_labels.npy')
        test_imdb_tokens = np.load('test_imdb_tokens.npy')
        test_imdb_labels = np.load('test_imdb_labels.npy')
    else:
        train_data=[]
        test_data=[]
        train_label=[]
        test_label=[]

        f = open('imdb.txt', encoding='utf-8')
        for line in f:
            line = line.split('\t')
            text=line[2]
            if text[0]=='"':
                text=text[1:]
            tokens = tokenization(tokenizer, text, max_len)
            if line[-2] == 'pos':
                label=1
            elif line[-2] == 'neg':
                label=0
            elif line[-2] == 'unsup':
                break
            else:
                print(line[-2])
                raise ValueError

            if line[1]=='train':
                train_data.append(tokens)
                train_label.append(label)
            elif line[1]=='test':
                test_data.append(tokens)
                test_label.append(label)
            else:
                raise ValueError

        assert len(train_data) == len(train_label)
        assert len(test_data) == len(test_label)

        train_imdb_tokens = torch.zeros((len(train_data),train_data[0].shape[0]), dtype=torch.long)
        train_imdb_labels = torch.zeros((len(train_label),1), dtype=torch.long)
        test_imdb_tokens = torch.zeros((len(test_data),test_data[0].shape[0]), dtype=torch.long)
        test_imdb_labels = torch.zeros((len(test_label),1), dtype=torch.long)

        for i in range(len(train_data)):
            train_imdb_tokens[i]=train_data[i]
            train_imdb_labels[i]=train_label[i]
        for i in range(len(test_data)):
            test_imdb_tokens[i]=test_data[i]
            test_imdb_labels[i]=test_label[i]

        np.save('train_imdb_tokens', train_imdb_tokens.numpy())
        np.save('train_imdb_labels', train_imdb_labels.numpy())
        np.save('test_imdb_tokens', test_imdb_tokens.numpy())
        np.save('test_imdb_labels', test_imdb_labels.numpy())

    train_imdb_tokens = torch.from_numpy(np.load('train_imdb_tokens.npy'))
    train_imdb_labels = torch.from_numpy(np.load('train_imdb_labels.npy'))
    test_imdb_tokens = torch.from_numpy(np.load('test_imdb_tokens.npy'))
    test_imdb_labels = torch.from_numpy(np.load('test_imdb_labels.npy'))

    train_data=[]
    test_data=[]

    for i in range(len(train_imdb_labels)):
        if train_imdb_labels[i].item() in sample_classes:
            train_data.append([train_imdb_tokens[i], train_imdb_labels[i]])
    for i in range(len(test_imdb_labels)):
        if test_imdb_labels[i].item() in sample_classes:
            test_data.append([test_imdb_tokens[i], test_imdb_labels[i]])


    return train_data, test_data

def AmazonDataload(tokenizer, sample_classes=[0,1], max_len=512):    
    if isfile('train_amazon_tokens.npy') and isfile('test_amazon_tokens.npy') and isfile('train_amazon_labels.npy') and isfile('test_amazon_labels.npy'):
        train_amazon_tokens = np.load('train_amazon_tokens.npy')
        train_amazon_labels = np.load('train_amazon_labels.npy')
        test_amazon_tokens = np.load('test_amazon_tokens.npy')
        test_amazon_labels = np.load('test_amazon_labels.npy')
    else:
        train_data=[]
        test_data=[]
        train_label=[]
        test_label=[]

        import csv

        with open('amazon.csv', encoding='utf-8') as csvfile:
            rdr = csv.reader(csvfile, delimiter =',')
            for row in rdr:
                label = row[14]
                text = row[16]
                if not label.isdigit():
                    continue
                if int(label) in [1,5]:
                    if int(label) == 1:
                        label=0
                    else:
                        label=1
                    tokens = tokenization(tokenizer, text, max_len)
                    if random.random()<0.7:
                        train_data.append(tokens)
                        train_label.append(label)
                    else:
                        test_data.append(tokens)
                        test_label.append(label)                

        assert len(train_data) == len(train_label)
        assert len(test_data) == len(test_label)

        train_amazon_tokens = torch.zeros((len(train_data),train_data[0].shape[0]), dtype=torch.long)
        train_amazon_labels = torch.zeros((len(train_label),1), dtype=torch.long)
        test_amazon_tokens = torch.zeros((len(test_data),test_data[0].shape[0]), dtype=torch.long)
        test_amazon_labels = torch.zeros((len(test_label),1), dtype=torch.long)

        for i in range(len(train_data)):
            train_amazon_tokens[i]=train_data[i]
            train_amazon_labels[i]=train_label[i]
        for i in range(len(test_data)):
            test_amazon_tokens[i]=test_data[i]
            test_amazon_labels[i]=test_label[i]

        np.save('train_amazon_tokens', train_amazon_tokens.numpy())
        np.save('train_amazon_labels', train_amazon_labels.numpy())
        np.save('test_amazon_tokens', test_amazon_tokens.numpy())
        np.save('test_amazon_labels', test_amazon_labels.numpy())

    train_amazon_tokens = torch.from_numpy(np.load('train_amazon_tokens.npy'))
    train_amazon_labels = torch.from_numpy(np.load('train_amazon_labels.npy'))
    test_amazon_tokens = torch.from_numpy(np.load('test_amazon_tokens.npy'))
    test_amazon_labels = torch.from_numpy(np.load('test_amazon_labels.npy'))

    train_data=[]
    test_data=[]

    for i in range(len(train_amazon_labels)):
        if train_amazon_labels[i].item() in sample_classes:
            train_data.append([train_amazon_tokens[i], train_amazon_labels[i]])
    for i in range(len(test_amazon_labels)):
        if test_amazon_labels[i].item() in sample_classes:
            test_data.append([test_amazon_tokens[i], test_amazon_labels[i]])


    return train_data, test_data

def FoodDataload(tokenizer, sample_classes=[0,1], max_len=512):    
    if isfile('train_food_tokens.npy') and isfile('test_food_tokens.npy') and isfile('train_food_labels.npy') and isfile('test_food_labels.npy'):
        train_food_tokens = np.load('train_food_tokens.npy')
        train_food_labels = np.load('train_food_labels.npy')
        test_food_tokens = np.load('test_food_tokens.npy')
        test_food_labels = np.load('test_food_labels.npy')
    else:
        train_data=[]
        test_data=[]
        train_label=[]
        test_label=[]

        

        f= open('foods_train.txt', encoding='utf-8', errors='ignore')
        for line in f:
            line = line.split(':')
            label = line[1]
            text = line[0]
            if int(label) in [1,5]:
                if int(label) == 1:
                    label=0
                else:
                    label=1
                tokens = tokenization(tokenizer, text, max_len)
                train_data.append(tokens)
                train_label.append(label)

        f= open('foods_test.txt', encoding='utf-8', errors='ignore')
        for line in f:
            line = line.split(':')
            label = line[1]
            text = line[0]
            if int(label) in [1,5]:
                if int(label) == 1:
                    label=0
                else:
                    label=1
                tokens = tokenization(tokenizer, text, max_len)
                test_data.append(tokens)
                test_label.append(label)                

        assert len(train_data) == len(train_label)
        assert len(test_data) == len(test_label)

        train_food_tokens = torch.zeros((len(train_data),train_data[0].shape[0]), dtype=torch.long)
        train_food_labels = torch.zeros((len(train_label),1), dtype=torch.long)
        test_food_tokens = torch.zeros((len(test_data),test_data[0].shape[0]), dtype=torch.long)
        test_food_labels = torch.zeros((len(test_label),1), dtype=torch.long)

        for i in range(len(train_data)):
            train_food_tokens[i]=train_data[i]
            train_food_labels[i]=train_label[i]
        for i in range(len(test_data)):
            test_food_tokens[i]=test_data[i]
            test_food_labels[i]=test_label[i]

        np.save('train_food_tokens', train_food_tokens.numpy())
        np.save('train_food_labels', train_food_labels.numpy())
        np.save('test_food_tokens', test_food_tokens.numpy())
        np.save('test_food_labels', test_food_labels.numpy())

    train_food_tokens = torch.from_numpy(np.load('train_food_tokens.npy'))
    train_food_labels = torch.from_numpy(np.load('train_food_labels.npy'))
    test_food_tokens = torch.from_numpy(np.load('test_food_tokens.npy'))
    test_food_labels = torch.from_numpy(np.load('test_food_labels.npy'))

    train_data=[]
    test_data=[]

    for i in range(len(train_food_labels)):
        if train_food_labels[i].item() in sample_classes:
            train_data.append([train_food_tokens[i], train_food_labels[i]])
    for i in range(len(test_food_labels)):
        if test_food_labels[i].item() in sample_classes:
            test_data.append([test_food_tokens[i], test_food_labels[i]])


    return train_data, test_data

def SST2Dataload(tokenizer, sample_classes=[0,1],max_len=512):
    if isfile('train_sst2_tokens.npy') and isfile('test_sst2_tokens.npy') and isfile('train_sst2_labels.npy') and isfile('test_sst2_labels.npy'):
        train_sst2_tokens = np.load('train_sst2_tokens.npy')
        train_sst2_labels = np.load('train_sst2_labels.npy')
        test_sst2_tokens = np.load('test_sst2_tokens.npy')
        test_sst2_labels = np.load('test_sst2_labels.npy')
    else:
        train_data=[]
        test_data=[]
        train_label=[]
        test_label=[]

        f = open('sst2_train.tsv', encoding='utf-8')
        g = open('sst2_dev.tsv', encoding='utf-8')
        for line in f:
            line = line.split('\t')
            text = line[0]
            tokens = tokenization(tokenizer, text, max_len)
            label = int(line[1])
            train_data.append(tokens)
            train_label.append(label)
        for line in g:
            line = line.split('\t')
            text = line[0]
            tokens = tokenization(tokenizer, text, max_len)
            label = int(line[1])
            test_data.append(tokens)
            test_label.append(label)

        assert len(train_data) == len(train_label)
        assert len(test_data) == len(test_label)

        train_sst2_tokens = torch.zeros((len(train_data),train_data[0].shape[0]), dtype=torch.long)
        train_sst2_labels = torch.zeros((len(train_label),1), dtype=torch.long)
        test_sst2_tokens = torch.zeros((len(test_data),test_data[0].shape[0]), dtype=torch.long)
        test_sst2_labels = torch.zeros((len(test_label),1), dtype=torch.long)

        for i in range(len(train_data)):
            train_sst2_tokens[i]=train_data[i]
            train_sst2_labels[i]=train_label[i]
        for i in range(len(test_data)):
            test_sst2_tokens[i]=test_data[i]
            test_sst2_labels[i]=test_label[i]

        np.save('train_sst2_tokens', train_sst2_tokens.numpy())
        np.save('train_sst2_labels', train_sst2_labels.numpy())
        np.save('test_sst2_tokens', test_sst2_tokens.numpy())
        np.save('test_sst2_labels', test_sst2_labels.numpy())

    train_sst2_tokens = torch.from_numpy(np.load('train_sst2_tokens.npy'))
    train_sst2_labels = torch.from_numpy(np.load('train_sst2_labels.npy'))
    test_sst2_tokens = torch.from_numpy(np.load('test_sst2_tokens.npy'))
    test_sst2_labels = torch.from_numpy(np.load('test_sst2_labels.npy'))

    train_data=[]
    test_data=[]

    for i in range(len(train_sst2_labels)):
        if train_sst2_labels[i].item() in sample_classes:
            train_data.append([train_sst2_tokens[i], train_sst2_labels[i]])
    for i in range(len(test_sst2_labels)):
        if test_sst2_labels[i].item() in sample_classes:
            test_data.append([test_sst2_tokens[i], test_sst2_labels[i]])


    return train_data, test_data

def ReutersDataload(tokenizer, sample_classes=[0,1],max_len=512):
    label=torch.LongTensor([51])
    train_data=None
    files = os.listdir('reuters_test/')
    for file in files:
        newsfile = open('reuters_test/'+file, encoding='utf-8', errors='ignore')
        text=newsfile.read()
        tokens = tokenization(tokenizer, text, max_len)
        test_data.append([tokens, label])
    return train_data, test_data

def VanillaDataload(datatype, tokenizer, sample_classes, max_len=512):
    if datatype=='news':
        train_data, test_data = NewsDataload(tokenizer, sample_classes, max_len)
    elif datatype=='review':
        train_data, test_data = ReviewDataload(tokenizer, sample_classes, max_len)
    elif datatype=='imdb':
        train_data, test_data = IMDBDataload(tokenizer, sample_classes, max_len)
    elif datatype=='amazon':
        train_data, test_data = AmazonDataload(tokenizer, sample_classes, max_len)
    elif datatype=='food':
        train_data, test_data = FoodDataload(tokenizer, sample_classes, max_len)
    elif datatype=='sst2':
        train_data, test_data = SST2Dataload(tokenizer, sample_classes, max_len)
    elif datatype=='reuters':
        train_data, test_data = ReutersDataload(tokenizer, sample_classes, max_len)

    return train_data, test_data

def main():

    MODELS = {'bert' : (RobertaModel, RobertaTokenizer, 'roberta-base')}
    model_class, tokenizer_class, pretrained_weights = MODELS['bert']
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampling_rates=[0.1,0.25,0.5,0.75,1.0]
    datatypes = ['news', 'review', 'imdb', 'food', 'sst2']

    for rate in sampling_rates:
        for datatype in datatypes:
            sample_classes=sample(num_total_classes[datatype],rate)
            VanillaDataload(datatype,tokenizer, sample_classes)

if __name__ == "__main__":
    main()

