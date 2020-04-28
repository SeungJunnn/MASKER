import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

num_total_classes={'news':20, 'review':50, 'imdb':2, 'food':2, 'sst2':2,'reuters':2}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_TOKEN=50264 # or 104
CLS_TOKEN=0 #or 101
PAD_TOKEN=1 #or 0

def tokenization(tokenizer, raw_text, max_len):
    input_ids = torch.tensor(tokenizer.encode(raw_text, add_special_tokens=True))
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    elif len(input_ids) < max_len:
        pad_tensor=torch.zeros((max_len-len(input_ids)), dtype=torch.long)
        pad_tensor.fill_(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
        input_ids = torch.cat((input_ids,pad_tensor))
    return input_ids

def one_hot(labels, n_classes):
    one_hot=torch.zeros((len(labels), n_classes), dtype=torch.float32)
    for i in range(len(labels)):
        one_hot[i][labels[i]]=1.0
    return one_hot

def uniform_labels(labels, n_classes, model_type):
    one_hot=torch.zeros((len(labels), n_classes), dtype=torch.float32)
    if model_type=='softmax':
        one_hot.fill_(1/n_classes)
    return one_hot

def sample(n_classes, sampling_rate):
    classes = [i for i in range(n_classes)]
    sample_number = n_classes*sampling_rate
    sample_classes = [i for i in range(int(sample_number))]
    return sample_classes

def splitindex(X,y,sample_classes,ratio=0.7):
    train_idx = []
    test_idx = []
    for i in sample_classes:
        shuffle_idx = np.random.permutation(1000)
        shuffle_idx+= i*1000
        for i in range(int(1000*ratio)):
            train_idx.append(shuffle_idx[i])
        for j in range(int(1000*ratio),1000):
            test_idx.append(shuffle_idx[j])
    return train_idx, test_idx

def convert_random_to_mask(input_ids, label):
    labels = torch.empty(len(input_ids)+1, dtype=torch.long)
    labels.fill_(-1)
    m_input_ids = torch.empty(input_ids.shape, dtype=torch.long)
    m_input_ids.data = input_ids.clone()
    for i,idx in enumerate(input_ids):
        if random.random() < 0.15 and idx != CLS_TOKEN:
            labels[i] = m_input_ids[i]
            m_input_ids[i] = MASK_TOKEN #[MASK] token
    labels[-1] = int(label)
    return m_input_ids, labels

def convert_keyword_to_mask(input_ids, keytokens, label):
    labels = torch.empty(len(input_ids)+1, dtype=torch.long)
    labels.fill_(-1)
    m_input_ids = torch.empty(input_ids.shape, dtype=torch.long)
    m_input_ids.data = input_ids.clone()

    for i,idx in enumerate(input_ids):
        if random.random()<0.5:
            if idx in keytokens and idx != CLS_TOKEN:
                labels[i] = keytokens.index(idx)
                m_input_ids[i] = MASK_TOKEN #[MASK] token
            elif idx==PAD_TOKEN:
                break
    labels[-1] = int(label)
    return m_input_ids, labels

def convert_to_features(tokens,task,label):
    if task == 'vanilla':
        return [tokens, label]
    elif task == 'mklm':
        m_tokens = tokens.unsqueeze(0)
        f_tokens = torch.cat((m_tokens, m_tokens))
        labels = torch.zeros(len(tokens)+1, dtype=torch.long)
        labels.fill_(-1)
        labels[-1] = int(label)
        return [f_tokens, labels]
    elif task =='exposure':
        m_tokens = tokens.unsqueeze(0)
        f_tokens = torch.cat((m_tokens, m_tokens, m_tokens))
        labels = torch.zeros(len(tokens)+1, dtype=torch.long)
        labels.fill_(-1)
        labels[-1] = int(label)
        return [f_tokens, labels]

class Dataset(Dataset):
    def __init__(self, filename):
        self.data = filename
        self.len = len(filename)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return self.len

def mask_attention(x):
    attention_masks=[]
    for seq in x[:,0]:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = torch.FloatTensor(attention_masks)
    return attention_masks

def compute_loss(logits, labels, n_classes, model_type, task, uniform, device):
    c1 = nn.CrossEntropyLoss(ignore_index=-1)
    c2 = nn.BCEWithLogitsLoss()
    kl = nn.KLDivLoss()
    if uniform==False:
        ss_loss=torch.FloatTensor([0]).to(device)
        uni_loss=torch.FloatTensor([0]).to(device)
        if task=='vanilla':
            cls_logits=logits
        elif task == 'mklm':
            ss_logits, cls_logits = logits
            ss_logits = ss_logits.permute(0,2,1)
            ss_loss = c1(ss_logits, labels[:,:-1])
            labels=labels[:,-1]

    elif uniform==True: #task is always mklm for here
        ss_logits, cls_logits, uni_logits = logits
        ss_logits = ss_logits.permute(0,2,1)
        ss_loss = c1(ss_logits, labels[:,:-1])
        uni_loss = kl(F.log_softmax(uni_logits,dim=1), F.softmax(uniform_labels(labels[:,-1],n_classes, model_type),dim=1).to(device))
        labels=labels[:,-1]

    if model_type =='softmax':
            cls_loss = c1(cls_logits, labels)
    elif model_type == 'sigmoid':
        cls_loss = c2(cls_logits, one_hot(labels, n_classes).to(device))
    else:
        raise ValueError

    return ss_loss, cls_loss, uni_loss

def auroc(logits, labels, num_id_class, model_type):
    def ths_list(lb, ub, num):
        thresholds = []
        for i in range(num):
            thresholds.append(lb + i*(ub-lb)/(num-1))
        return thresholds
    def point(outputs, threshold, id_ood):
        fp = 0
        tp = 0
        tn = 0
        fn=0
        for i in range(len(outputs)):
            if torch.max(outputs[i]) > threshold and id_ood[i] < num_id_class:
                tn+=1
            elif torch.max(outputs[i]) < threshold and id_ood[i] >= num_id_class:
                tp+=1
            elif torch.max(outputs[i]) > threshold and id_ood[i] >= num_id_class:
                fn+=1
            else:
                fp+=1
        if tn+fp == 0 :
            fpr=0
        else:
            fpr = fp/(tn+fp)
        if tp+fn == 0:
            tpr=0
        else:
            tpr = tp/(tp+fn)
        pos=tp+fn
        neg=fp+tn
        return fpr, tpr, pos, neg

    thresholds=ths_list(0,1,10000)

    softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()

    total=len(logits)

    if model_type == 'softmax':
        outputs = softmax(logits)
    else:
        outputs = sigmoid(logits)
    fpr_list = []
    tpr_list = []
    e=0
    f=0
    maxd=0

    for i,threshold in enumerate(thresholds):
        fpr, tpr,pos,neg = point(outputs, threshold, labels)
        dr=((1-fpr)*neg+tpr*pos)/total
        if dr > maxd:
            maxd=dr
        if 1-fpr < tpr and e==0 :
            print('Equal Error Rate : ', fpr)
            e=1

        if tpr > 0.8 and f == 0:
            print('TNR at TPR 0.8 : ', 1-fpr)
            f=1

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        if fpr == 1.0 and tpr == 1.0:
            break
    print('detection rate : ',maxd)
    area = 0
    for i in range(len(fpr_list)-1):
        area += (fpr_list[i+1]-fpr_list[i]) * (tpr_list[i]+tpr_list[i+1]) / 2
    print('AUROC : ', area)
