import os
import numpy as np
import torch
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataload import *
from data.utils import *
from models import ExposureNet, MaskNet, FineTuningNet
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

MODELS = {'bert' : (RobertaModel, RobertaTokenizer, 'roberta-base')}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_class, tokenizer_class, pretrained_weights = MODELS['bert']

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert = model_class.from_pretrained(pretrained_weights)

#### Set here ####

datatype = 'review'
num_id_class = 50
task = 'None'
model_type='softmax'
use_outer_ood = 'review'

#### Set here ####

path = './models'
model_list = os.listdir(path)
data_list = ['reuters','news','review','imdb','sst2','food']
model_list = ['vanilla_softmax_news_20','exposure_sigmoid_news_20','vanilla_softmax_review_50','exposure_sigmoid_review_50']

for model_name in model_list:
    print('***** '+model_name+' *****')
    os.chdir("./models")
    model = torch.load(model_name)
    task,model_type,datatype,num_id_class=model_name.split('_')
    num_id_class=int(num_id_class)
    os.chdir("../data")
    _, id_data = VanillaDataload(datatype, tokenizer, [i for i in range(num_id_class)], max_len=512)
    if num_total_classes[datatype]!=int(num_id_class): #split
        _,ood_data=VanillaDataload(datatype, tokenizer, [i for i in range(num_id_class,num_total_classes[datatype])], max_len=512)

        test_data = id_data+ood_data
        for i in range(len(ood_data)):
            test_data[-i][1]=torch.LongTensor([51]) #ood data
        final_test_data=[]
        for data in test_data:
            tokens, label=data
            final_test_data.append(convert_to_features(tokens,task,label))
        testdataset = Dataset(filename=final_test_data)
        test_loader = DataLoader(dataset=testdataset, batch_size=64, shuffle=False, num_workers=2)
        l_logits = []
        id_ood = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                if task=='vanilla':
                    logits = model(inputs)
                elif task=='mklm':
                    _,logits = model(inputs)
                else:
                    _,logits,_ = model(inputs)
                
                for i in range(len(logits)):
                    l_logits.append(logits[i])
                    id_ood.append(labels[i][-1].item())

        logits = torch.zeros((len(final_test_data),num_id_class), dtype=torch.float32)

        for i in range(len(final_test_data)):
            logits[i] = l_logits[i]
        logits = logits.numpy()
        id_oods = np.asarray(id_ood)
        os.chdir("../outputs")
        np.save('logits_'+model_name+'_'+datatype, logits)
        np.save('label_'+model_name+'_'+datatype, id_oods)

        logits = torch.from_numpy(np.load('logits_'+model_name+'_'+datatype+'.npy'))
        labels = torch.from_numpy(np.load('label_'+model_name+'_'+datatype+'.npy'))

        auroc(logits, labels, num_id_class, model_type)
        os.chdir("../")

    else:
        for ood in data_list:
            if ood != datatype:
                _, ood_data=VanillaDataload(ood, tokenizer, [i for i in range(num_total_classes[ood])], max_len=512)
            else:
                continue
            test_data = id_data+ood_data
            print('***** OOD datatype : '+ood)
            for i in range(len(ood_data)):
                test_data[-i][1]=torch.LongTensor([51]) #ood data
            final_test_data=[]
            for data in test_data:
                tokens, label=data
                final_test_data.append(convert_to_features(tokens,task,label))
            testdataset = Dataset(filename=final_test_data)
            test_loader = DataLoader(dataset=testdataset, batch_size=64, shuffle=False, num_workers=2)
            l_logits = []
            id_ood = []
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    if task=='vanilla':
                        logits = model(inputs)
                    elif task=='mklm':
                        _,logits = model(inputs)
                    else:
                        _,logits,_ = model(inputs)
                    
                    for i in range(len(logits)):
                        l_logits.append(logits[i])
                        id_ood.append(labels[i][-1].item())

            logits = torch.zeros((len(final_test_data),num_id_class), dtype=torch.float32)

            for i in range(len(final_test_data)):
                logits[i] = l_logits[i]
            logits = logits.numpy()
            id_oods = np.asarray(id_ood)
            os.chdir("../outputs")
            np.save('logits_'+model_name+'_'+ood, logits)
            np.save('label_'+model_name+'_'+ood, id_oods)

            logits = torch.from_numpy(np.load('logits_'+model_name+'_'+ood+'.npy'))
            labels = torch.from_numpy(np.load('label_'+model_name+'_'+ood+'.npy'))

            auroc(logits, labels, num_id_class, model_type)
            print('**********')
            os.chdir("../data")
        os.chdir('../')
