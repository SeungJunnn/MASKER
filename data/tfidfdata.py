from os import listdir
from os.path import isfile, join
import random
import csv
import numpy as np
import torch
import json
from transformers import BertConfig, BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import *
from dataload import *

def Newstfidfkeytokens(keyword_size, sample_classes, tokenizer):
    if isfile('keytokens_news_'+str(len(sample_classes))+'.npy', np.array(keytokens)):
        keytokens=torch.from_numpy(np.load('keytokens_news_'+str(len(sample_classes))+'.npy')).tolist()
    else:
        doc = open('train.csv', encoding='utf-8')
        documents = ['' for i in range(len(sample_classes))]
        for line in doc:
            line = line.split(',')
            newsfile = open(line[0], encoding='utf-8', errors='ignore')
            text = newsfile.read()
            label = int(line[1])
            if label < len(sample_classes):
                documents[label]+=text
            else:
                break
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        tX = vectorizer.fit_transform(documents)
        tX=tX.todense()
        tX=np.squeeze(np.asarray(tX))

        for i in range(tX.shape[1]):
            for j in range(tX.shape[0]):
                if tX[j][i]==0:
                    break
                elif tX[j][i]!=0 and j==tX.shape[0]-1:
                    for k in range(tX.shape[0]):
                        tX[k][i]=0

        keytokens = []
        for i in range(len(sample_classes)):
            c = 0
            idxx=tX[i].argsort()[::-1]
            j=0
            while c < keyword_size:
                keyword = vectorizer.get_feature_names()[idxx[j]]
                keytoken = tokenizer.encode(keyword)
                if len(keytoken)==1:
                    if keytoken[0] not in keytokens:
                        c+=1
                        keytokens.append(keytoken[0])
                j+=1

        np.save('keytokens_news_'+str(len(sample_classes)), np.array(keytokens))
        print('keytokens saved!')
    return keytokens

def Reviewtfidfkeytokens(keyword_size, sample_classes, tokenizer):
    if isfile('keytokens_review_'+str(len(sample_classes))+'.npy', np.array(keytokens)):
        keytokens=torch.from_numpy(np.load('keytokens_review_'+str(len(sample_classes))+'.npy')).tolist()
    else:
        fn = '50EleReviews.json'
        with open(fn, 'r') as infile:
                docs = json.load(infile)
        X = docs['X']
        train_idx=np.load('train_idx.npy')
        documents = ['' for i in range(len(sample_classes))]
        for i in range(len(train_idx)):
            label = train_idx[i]//1000
            if label >= len(sample_classes):
                break
            documents[label]+=X[train_idx[i]]
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        tX = vectorizer.fit_transform(documents)
        tX=tX.todense()
        tX=np.squeeze(np.asarray(tX))

        for i in range(tX.shape[1]):
            for j in range(tX.shape[0]):
                if tX[j][i]==0:
                    break
                elif tX[j][i]!=0 and j==tX.shape[0]-1:
                    for k in range(tX.shape[0]):
                        tX[k][i]=0

        keytokens = []
        for i in range(len(sample_classes)):
            c = 0
            idxx=tX[i].argsort()[::-1]
            j=0
            while c < keyword_size:
                keyword = vectorizer.get_feature_names()[idxx[j]]
                keytoken = tokenizer.encode(keyword)
                if len(keytoken)==1:
                    if keytoken[0] not in keytokens:
                        keytokens.append(keytoken[0])
                        c+=1
                j+=1

        np.save('keytokens_review_'+str(len(sample_classes)), np.array(keytokens))
        print('keytokens saved!')
    return keytokens

def IMDBtfidfkeytokens(keyword_size, sample_classes, tokenizer):
    if isfile('keytokens_imdb_'+str(len(sample_classes))+'.npy', np.array(keytokens)):
        keytokens=torch.from_numpy(np.load('keytokens_imdb_'+str(len(sample_classes))+'.npy')).tolist()
    else:
        documents = ['' for i in range(2)]
        f = open('imdb.txt', encoding='utf-8')
        for line in f:
            line = line.split('\t')
            text=line[2]
            if text[0]=='"':
                text=text[1:]
            if line[-2] == 'pos':
                label=1
            elif line[-2] == 'neg':
                label=0
            elif line[-2] == 'unsup':
                break
            documents[label]+=text
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        tX = vectorizer.fit_transform(documents)
        tX=tX.todense()
        tX=np.squeeze(np.asarray(tX))

        for i in range(tX.shape[1]):
            for j in range(tX.shape[0]):
                if tX[j][i]==0:
                    break
                elif tX[j][i]!=0 and j==tX.shape[0]-1:
                    for k in range(tX.shape[0]):
                        tX[k][i]=0

        keytokens = []
        for i in range(2):
            c = 0
            idxx=tX[i].argsort()[::-1]
            j=0
            while c < keyword_size:
                keyword = vectorizer.get_feature_names()[idxx[j]]
                keytoken = tokenizer.encode(keyword, add_special_tokens=False)
                if len(keytoken)==1:
                    if keytoken[0] not in keytokens:
                        keytokens.append(keytoken[0])
                        c+=1
                j+=1

        np.save('keytokens_imdb_'+str(len(sample_classes)), np.array(keytokens))
        print('keytokens saved!')
    return keytokens

def Foodtfidfkeytokens(keyword_size, sample_classes, tokenizer):
    if isfile('keytokens_food_'+str(len(sample_classes))+'.npy', np.array(keytokens)):
        keytokens=torch.from_numpy(np.load('keytokens_food_'+str(len(sample_classes))+'.npy')).tolist()
    else:
        documents = ['' for i in range(2)]
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
                documents[label]+=text
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        tX = vectorizer.fit_transform(documents)
        tX=tX.todense()
        tX=np.squeeze(np.asarray(tX))

        for i in range(tX.shape[1]):
            for j in range(tX.shape[0]):
                if tX[j][i]==0:
                    break
                elif tX[j][i]!=0 and j==tX.shape[0]-1:
                    for k in range(tX.shape[0]):
                        tX[k][i]=0

        keytokens = []
        for i in range(2):
            c = 0
            idxx=tX[i].argsort()[::-1]
            j=0
            while c < keyword_size:
                keyword = vectorizer.get_feature_names()[idxx[j]]
                keytoken = tokenizer.encode(keyword, add_special_tokens=False)
                if len(keytoken)==1:
                    if keytoken[0] not in keytokens:
                        c+=1
                        keytokens.append(keytoken[0])
                j+=1

        np.save('keytokens_food_'+str(len(sample_classes)), np.array(keytokens))
        print('keytokens saved!')
    return keytokens

def SST2tfidfkeytokens(keyword_size, sample_classes, tokenizer):
    if isfile('keytokens_sst2_'+str(len(sample_classes))+'.npy', np.array(keytokens)):
        keytokens=torch.from_numpy(np.load('keytokens_sst2_'+str(len(sample_classes))+'.npy')).tolist()
    else:
        doc = open('sst2_train.tsv', encoding='utf-8')
        documents = ['' for i in range(len(sample_classes))]
        for line in doc:
            line = line.split('\t')
            text = line[0]
            label = int(line[1])
            documents[label]+=text
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        tX = vectorizer.fit_transform(documents)
        tX=tX.todense()
        tX=np.squeeze(np.asarray(tX))

        for i in range(tX.shape[1]):
            for j in range(tX.shape[0]):
                if tX[j][i]==0:
                    break
                elif tX[j][i]!=0 and j==tX.shape[0]-1:
                    for k in range(tX.shape[0]):
                        tX[k][i]=0

        keytokens = []
        for i in range(2):
            c = 0
            idxx=tX[i].argsort()[::-1]
            j=0
            while c < keyword_size:
                keyword = vectorizer.get_feature_names()[idxx[j]]
                keytoken = tokenizer.encode(keyword, add_special_tokens=False)
                if len(keytoken)==1:
                    if keytoken[0] not in keytokens:
                        c+=1
                        keytokens.append(keytoken[0])
                j+=1

        np.save('keytokens_sst2_'+str(len(sample_classes)), np.array(keytokens))
        print('keytokens saved!')
    return keytokens

def TFIDFkeytokens(datatype, keyword_size, sample_classes, tokenizer):
    if datatype=='news':
        keytokens = Newstfidfkeytokens(keyword_size, sample_classes, tokenizer)
    elif datatype=='review':
        keytokens = Reviewtfidfkeytokens(keyword_size, sample_classes, tokenizer)
    elif datatype=='imdb':
        keytokens = IMDBtfidfkeytokens(keyword_size, sample_classes, tokenizer)
    elif datatype=='amazon':
        keytokens = Amazontfidfkeytokens(keyword_size, sample_classes, tokenizer)
    elif datatype=='food':
        keytokens = Foodtfidfkeytokens(keyword_size, sample_classes, tokenizer)
    elif datatype=='sst2':
        keytokens = SST2tfidfkeytokens(keyword_size, sample_classes, tokenizer)
    return keytokens