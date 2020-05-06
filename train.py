import os
import numpy as np
import torch
import json
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

def main():
    os.chdir("./data")
    w = 0.01
    w1=0.001
    w2=0.0001

    MAX_EPOCH=10
    savePATH='models'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODELS = {'bert' : (RobertaModel, RobertaTokenizer, 'roberta-base')}

    model_class, tokenizer_class, pretrained_weights = MODELS['bert']

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert = model_class.from_pretrained(pretrained_weights)
    bert = bert.to(device)
    total_vocab_size = bert.config.vocab_size
    parser = argparse.ArgumentParser()

    parser.add_argument("--datatype", default=None, type=str, required=True) #news, review, imdb....
    parser.add_argument("--sampling_rate", default=None, type=float, required=True)
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--task", default=None, type=str) #either vanilla, mklm
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--keyword_size", default=10, type=int)
    parser.add_argument("--keyword_type", default='attention', type=str) #either tfidf, attention, random
    parser.add_argument("--use_outlier_exposure", default=False, type=bool)
    args = parser.parse_args()

    print('datatype : ', args.datatype)
    print('sampling_rate : ', args.sampling_rate)
    print('model_type : ', args.model_type)
    print('task : ', args.task)

    if args.task == 'mklm' and args.keyword_type == 'random':
        batch_size=4
    elif args.task == 'mklm':
        batch_size=16
    elif args.task == 'vanilla':
        batch_size=32
    else:
        raise ValueError

    if args.model_type not in ['softmax', 'sigmoid']:
        print("Unknown model type")
        raise ValueError

    sample_classes=sample(num_total_classes[args.datatype],args.sampling_rate)
    n_classes=len(sample_classes)

    if args.task == 'vanilla':
        train_data, test_data = VanillaDataload(args.datatype, tokenizer, sample_classes, args.max_len)
        model=FineTuningNet(bert, n_classes)
        optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
    elif args.task == 'mklm':
        if not args.use_outlier_exposure:
            train_data, test_data, keytokens = KeywordMaskedDataload(args.keyword_type, args.datatype, args.keyword_size, tokenizer, sample_classes, args.max_len)
            model=MaskNet(bert, len(keytokens), n_classes)
        else:
            train_data, test_data, keytokens = WindowMaskedDataload(args.keyword_type, args.datatype, args.keyword_size, tokenizer, sample_classes, args.max_len, for_infer=False)
            model=ExposureNet(bert, len(keytokens), n_classes)
        #optimizer = optim.AdamW([{'params': model.bert.parameters(), 'lr' : 1e-5} ], lr=2e-5, eps=1e-8)
        optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

    traindataset = Dataset(filename=train_data)
    testdataset = Dataset(filename=test_data)
    train_loader = DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=testdataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model.to(device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    
    best = 0.5

    print('weight for self-supervision : ',w1)
    print('weight for outlier exposure : ',w2)

    for epoch in range(MAX_EPOCH):
        print('*** Epoch : %d ***' % epoch)
        
        tr_steps = 0
        running_loss = 0.0
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            if args.task == 'vanilla':
                labels = labels.squeeze(1)
            logits = model(inputs)
            optimizer.zero_grad()
            ss_loss, cls_loss, uni_loss = compute_loss(logits, labels, n_classes, args.model_type, args.task, args.use_outlier_exposure, device)
            loss = cls_loss+w1*ss_loss+w2*uni_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss1 += ss_loss.item()
            loss2 += cls_loss.item()
            loss3 += uni_loss.item()
            tr_steps+=1

        print('Total Loss : %f' % (running_loss/tr_steps))
        print('Classification Loss : %f' % (loss2/tr_steps))
        print('Self-Supervised Loss : %f' % (loss1/tr_steps))
        print('Outlier Exposure Loss : %f' %(loss3/tr_steps))
    
        correct=np.zeros(n_classes)
        total=np.zeros(n_classes)

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                if args.task == 'mklm':
                    c_labels = labels[:,-1]
                    if args.use_outlier_exposure == False:
                        _, logits = model(inputs)
                    else:
                        _, logits, _ = model(inputs)

                else:
                    c_labels = labels.squeeze(1)
                    logits = model(inputs)

                _, predicted = torch.max(logits.data, 1)
                for j in range(len(c_labels)):
                    total[int(c_labels[j].item())]+=1
                    if predicted[j].item() == c_labels[j].item() :
                        correct[int(c_labels[j].item())]+=1

            print('Accuracy : %f' %(sum(correct)/sum(total)))
            if (sum(correct)/sum(total)) > best:
                best = sum(correct)/sum(total)
                if args.use_outlier_exposure == True:
                    task_name='exposure'
                elif args.task=='mklm':
                    task_name=args.keyword_type
                else:
                    task_name=args.task #vanilla

                os.chdir('../models')
                torch.save(model, task_name+'_'+args.model_type+'_'+args.datatype+'_'+str(n_classes))

    print('Best Accuarcy : ',best)

if __name__ == "__main__":
    main()
