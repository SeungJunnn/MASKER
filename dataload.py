from os import listdir
from os.path import isfile, join
import random
import csv
import numpy as np
import torch
import json
from data.utils import *
from data.vanilladata import *
from data.tfidfdata import *
from data.attentiondata import *

def KeywordMaskedDataload(wordtype, datatype, keyword_size, tokenizer, sample_classes, max_len):
    train_data=[]
    test_data=[]
    train_label=[]
    test_label=[]

    if wordtype in ['tfidf', 'attention']:
        if wordtype=='tfidf':
            dataname='keytokens_'+datatype+'_'
        elif wordtype=='attention':
            dataname='attentiontokens_'+datatype+'_'
        datalist=['train_tokens','train_labels','test_tokens','test_labels']

        for component in datalist:
            if not isfile(dataname+str(len(sample_classes))+'_'+component+'.npy'):
                reset=True
                break
            else:
                reset=False

        if not reset:
            print('Use existing one.')

        else:
            _,_ = VanillaDataload(datatype, tokenizer, sample_classes, max_len)

            train_tokens = np.load('train_'+datatype+'_tokens.npy')
            train_labels = np.load('train_'+datatype+'_labels.npy')
            test_tokens = np.load('test_'+datatype+'_tokens.npy')
            test_labels = np.load('test_'+datatype+'_labels.npy')

            if wordtype=='tfidf':
                keytokens = TFIDFkeytokens(datatype, keyword_size, sample_classes, tokenizer)
            elif wordtype=='attention':
                keytokens = Attentiontokens(datatype, keyword_size, sample_classes, tokenizer)

            for i in range(len(train_tokens)):
                tokens = torch.from_numpy(train_tokens[i])
                label = torch.from_numpy(train_labels[i])
                if int(label) in sample_classes:
                    m_tokens, labels = convert_keyword_to_mask(tokens, keytokens, label)
                    tokens = tokens.unsqueeze(0)
                    m_tokens = m_tokens.unsqueeze(0)
                    tokens = torch.cat((tokens, m_tokens))
                    train_data.append(tokens)
                    train_label.append(labels)

            for i in range(len(test_tokens)):
                tokens = torch.from_numpy(test_tokens[i])
                label = torch.from_numpy(test_labels[i])
                if int(label) in sample_classes:
                    m_tokens, labels = convert_keyword_to_mask(tokens, keytokens, label)
                    tokens = tokens.unsqueeze(0)
                    m_tokens= m_tokens.unsqueeze(0)
                    tokens = torch.cat((tokens, m_tokens))
                    test_data.append(tokens)
                    test_label.append(labels)

            assert len(train_data) == len(train_label)
            assert len(test_data) == len(test_label)

            train_tokens = torch.zeros((len(train_data),train_data[0].shape[0], train_data[0].shape[1]), dtype=torch.long)
            train_labels = torch.zeros((len(train_label),train_label[0].shape[0]), dtype=torch.long)
            test_tokens = torch.zeros((len(test_data),test_data[0].shape[0], train_data[0].shape[1]), dtype=torch.long)
            test_labels = torch.zeros((len(test_label),test_label[0].shape[0]), dtype=torch.long)

            for i in range(len(train_data)):
                train_tokens[i]=train_data[i]
                train_labels[i]=train_label[i]
            for i in range(len(test_data)):
                test_tokens[i]=test_data[i]
                test_labels[i]=test_label[i]

            np.save(dataname+str(len(sample_classes))+'_'+'train_tokens', train_tokens.numpy())
            np.save(dataname+str(len(sample_classes))+'_'+'train_labels', train_labels.numpy())
            np.save(dataname+str(len(sample_classes))+'_'+'test_tokens', test_tokens.numpy())
            np.save(dataname+str(len(sample_classes))+'_'+'test_labels', test_labels.numpy())

        train_tokens = torch.from_numpy(np.load(dataname+str(len(sample_classes))+'_'+'train_tokens.npy'))
        train_labels = torch.from_numpy(np.load(dataname+str(len(sample_classes))+'_'+'train_labels.npy'))
        test_tokens = torch.from_numpy(np.load(dataname+str(len(sample_classes))+'_'+'test_tokens.npy'))
        test_labels = torch.from_numpy(np.load(dataname+str(len(sample_classes))+'_'+'test_labels.npy'))
        keytokens = torch.from_numpy(np.load(dataname+str(len(sample_classes))+'.npy'))

        train_data=[]
        test_data=[]

        for i in range(len(train_labels)):
            if train_labels[i][-1].item() in sample_classes:
                train_data.append([train_tokens[i], train_labels[i]])
        for i in range(len(test_labels)):
            if test_labels[i][-1].item() in sample_classes:
                test_data.append([test_tokens[i], test_labels[i]])
        return train_data, test_data, keytokens
    elif wordtype == 'random':
        keytokens=None
        train_data=[]
        test_data=[]
        _,_ = VanillaDataload(datatype, tokenizer, sample_classes, max_len)
        train_tokens = np.load('train_'+datatype+'_tokens.npy')
        train_labels = np.load('train_'+datatype+'_labels.npy')
        test_tokens = np.load('test_'+datatype+'_tokens.npy')
        test_labels = np.load('test_'+datatype+'_labels.npy')
        for i in range(len(train_tokens)):
            tokens = torch.from_numpy(train_tokens[i])
            label = torch.from_numpy(train_labels[i])
            if int(label) in sample_classes:
                m_tokens, labels = convert_random_to_mask(tokens, label)
                tokens = tokens.unsqueeze(0)
                m_tokens= m_tokens.unsqueeze(0)
                tokens = torch.cat((tokens, m_tokens))
                train_data.append([tokens,labels])

        for i in range(len(test_tokens)):
            tokens = torch.from_numpy(test_tokens[i])
            label = torch.from_numpy(test_labels[i])
            if int(label) in sample_classes:
                m_tokens, labels = convert_random_to_mask(tokens, label)
                tokens = tokens.unsqueeze(0)
                m_tokens= m_tokens.unsqueeze(0)
                tokens = torch.cat((tokens, m_tokens))
                test_data.append([tokens,labels])
        return train_data, test_data, keytokens

def WindowMaskedDataload(wordtype, datatype, keyword_size, tokenizer, sample_classes, max_len, for_infer=False):
    if wordtype in ['tfidf','attention']:
        o_train_data, o_test_data, keytokens = KeywordMaskedDataload(wordtype, datatype, keyword_size, tokenizer, sample_classes, max_len)
    elif wordtype == 'random':
        o_train_data, o_test_data = RandomMaskedDataload(datatype,tokenizer,sample_classes,max_len)
        keytokens = None
    else:
        raise ValueError
    train_data=[]
    test_data=[]

    for data in o_train_data:
        window_masked_token=torch.empty(data[0][0].shape, dtype=torch.long)
        window_masked_token.data = data[0][0].clone()
        label = data[1]
        if for_infer == False:
            for i in range(len(label)-1):
                if data[0][1][i] == 0:
                    break
                if label[i] == -1:
                    if random.random() < 0.9 :
                        window_masked_token[i] =103 #MASK with probability 90
        tokens = data[0][0].unsqueeze(0)
        m_tokens= data[0][1].unsqueeze(0)
        w_tokens= window_masked_token.unsqueeze(0)
        tokens = torch.cat((tokens, m_tokens,w_tokens))
        train_data.append([tokens, label])

    for data in o_test_data:
        window_masked_token=torch.empty(data[0][0].shape, dtype=torch.long)
        window_masked_token.data = data[0][0].clone()
        label = data[1]
        tokens = data[0][0].unsqueeze(0)
        m_tokens= data[0][1].unsqueeze(0)
        w_tokens= window_masked_token.unsqueeze(0)
        tokens = torch.cat((tokens, m_tokens, w_tokens))
        test_data.append([tokens, label])

    return train_data, test_data, keytokens

