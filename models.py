import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data.utils import *

class ExposureNet(nn.Module):
    def __init__(self, bert, vocab_size, n_classes):
        super(ExposureNet, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(768,n_classes) #layer for classification

    def forward(self, x): #x contains 2 input_ids : [non-masked, masked]
        attention_masks=[]
        for seq in x[:,0]:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        attention_masks = torch.FloatTensor(attention_masks).to(device)

        _, o_out = self.bert(x[:,0], attention_masks) #pooled output
        m_out,_ = self.bert(x[:,1], attention_masks)
        _, w_out = self.bert(x[:,2], attention_masks)
        #o_out = o_out[:,0,:] #only for non-pooled output
        #w_out = w_out[:,0,:] #only for non-pooled output

        x = F.relu(self.fc1(m_out))
        x = self.dropout(x)
        logits1 = self.fc2(x) #for self-supervised task 
        logits2 = self.fc3(o_out) #For classification
        logits3 = self.fc3(w_out) #For outlier
        return logits1, logits2, logits3

class MaskNet(nn.Module): #layer for MLM, MKLM
    def __init__(self, bert, vocab_size, n_classes):
        super(MaskNet, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(768,n_classes) #layer for classification

    def forward(self, x): #x contains 2 input_ids : [non-masked, masked]
        attention_masks=[]
        for seq in x[:,0]:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        attention_masks = torch.FloatTensor(attention_masks).to(device)

        o_out, _ = self.bert(x[:,0], attention_masks) #pooled output
        m_out, _ = self.bert(x[:,1], attention_masks)
        #out = o_out[:,0,:]

        x = F.relu(self.fc1(m_out))
        x = self.dropout(x)
        logits1 = self.fc2(x) #for self-supervised task 
        logits2 = self.fc3(out) #For classification
        return logits1, logits2

class FineTuningNet(nn.Module): #Layer for normal fine-tuning
    def __init__(self, bert, n_classes):
        super(FineTuningNet, self).__init__()
        self.bert = bert
        self.fc = nn.Linear(768, n_classes)
    def forward(self, x):
        attention_masks=[]
        for seq in x:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        attention_masks = torch.FloatTensor(attention_masks).to(device)
        _, x = self.bert(x, attention_masks) #pooled output
        x = self.fc(x)
        return x