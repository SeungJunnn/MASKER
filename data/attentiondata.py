from os import listdir
from os.path import isfile, join
import random
import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
from train import FineTuningNet, Dataset
from dataload import *
from utils import *

def Attentiontokens(datatype, keyword_size, sample_classes, tokenizer):
    dataname='attentiontokens_'+datatype+'_'
    if isfile(dataname+str(len(sample_classes))+'.npy'):
        attentiontokens=torch.from_numpy(dataname+str(len(sample_classes)+'.npy')).tolist()
    else:
        total_model=torch.load('../models/vanilla_softmax_'+datatype+'_'+str(len(sample_classes)))
        model=total_model.module.bert
        pretrained_weight=model.state_dict()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_version = 'roberta-base'
        model = RobertaModel.from_pretrained(model_version, output_attentions=True).to(device)
        model.load_state_dict(pretrained_weight)

        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        tokenizer = RobertaTokenizer.from_pretrained(model_version, do_lower_case=True)

        train_data,_ = VanillaDataload(datatype, tokenizer, sample_classes)
        traindataset = Dataset(filename=train_data)
        train_loader = DataLoader(dataset=traindataset, batch_size=16, shuffle=False, num_workers=2)

        attention_scores = np.zeros(model.module.config.vocab_size)
        attention_freq = np.zeros(model.module.config.vocab_size)
        attention_average = np.zeros(model.module.config.vocab_size)

        with torch.no_grad():
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                attention = model(inputs)[-1]
                for i in range(len(attention[11])):
                    for j in range(512):
                        score = attention[11][i][0][0][j]
                        token = inputs[i][j].item()
                        attention_scores[token] += score.item()
                        attention_freq[token] += 1

        for i in range(model.module.config.vocab_size):
            score = attention_scores[i]
            freq = attention_freq[i]
            if freq == 0:
                average = 0
            else:
                average = score/freq
            attention_average[i]=average

        key_num = keyword_size*len(sample_classes)

        attentiontokens = attention_average.argsort()[-key_num:][::-1].tolist()
        np.save(dataname+str(len(sample_classes)),attentiontokens)
    return attentiontokens


