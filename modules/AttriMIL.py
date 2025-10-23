import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class AttriMIL(nn.Module): 
    '''
    Multi-Branch ABMIL with constraints
    '''
    def __init__(self, n_classes=2, dim=512):
        super().__init__()
        self.adaptor = nn.Sequential(nn.Linear(dim, dim//2),
                                     nn.ReLU(),
                                     nn.Linear(dim // 2 , dim))
        
        attention = []
        classifer = [nn.Linear(dim, 1) for i in range(n_classes)]
        for i in range(n_classes):
            attention.append(Attn_Net_Gated(L = dim, D = dim // 2,))
        self.attention_nets = nn.ModuleList(attention)
        self.classifiers = nn.ModuleList(classifer)
        self.n_classes = n_classes
        self.bias = nn.Parameter(torch.zeros(n_classes), requires_grad=True)
    
    def forward(self, h):
        h = h + self.adaptor(h)
        A_raw = torch.empty(self.n_classes, h.size(0), ) # N x 1
        instance_score = torch.empty(1, self.n_classes, h.size(0)).float().to(h.device)
        for c in range(self.n_classes):
            A, h = self.attention_nets[c](h)
            A = torch.transpose(A, 1, 0)  # 1 x N
            A_raw[c] = A
            instance_score[0, c] = self.classifiers[c](h)[:, 0]
        attribute_score = torch.empty(1, self.n_classes, h.size(0)).float().to(h.device)
        for c in range(self.n_classes):
            attribute_score[0, c] = instance_score[0, c] * torch.exp(A_raw[c])
            
        logits = torch.empty(1, self.n_classes).float().to(h.device)
        for c in range(self.n_classes):
            logits[0, c] = torch.sum(attribute_score[0, c], keepdim=True, dim=-1) / torch.sum(torch.exp(A_raw[c]), dim=-1) + self.bias[c]
            
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {}
        return logits, Y_prob, Y_hat, attribute_score, results_dict
    
    

def spatial_constraint(A, n_classes, nearest, ks=3):
    loss_spatial = torch.tensor(0.0).to(device)
    # N = A.shape[-1]
    for c in range(1, n_classes):
        score = A[:, c] # N
        nearest_score = score[nearest] # N ks^2-1
        abs_nearest = torch.abs(nearest_score)
        max_indices = torch.argmax(abs_nearest, dim=1)
        local_prototype = nearest_score.gather(1, max_indices.view(-1, 1)).squeeze()
        # print(local_prototype[:10])
        loss_spatial += torch.mean(torch.abs(torch.tanh(score - local_prototype)))
    return loss_spatial


def rank_constraint(data, label, model, A, n_classes, label_positive_list, label_negative_list):
    loss_rank = torch.tensor(0.0).to(device)
    for c in range(n_classes):
        if label == c:
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1] # top feature
            if label_positive_list[c].full():
                _ = label_positive_list[c].get()
            label_positive_list[c].put(h)
            if label_negative_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(device)
            else:
                h = label_negative_list[c].get()
                label_negative_list[c].put(h)
                _, _, _, Ah, _ = model(h.detach())
                if c != 0:
                    loss_rank = loss_rank + torch.clamp(torch.mean(Ah[0, c] - value), min=0.0) + torch.clamp(torch.mean(-value), min=0.0) + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
                else:
                    loss_rank = loss_rank + torch.clamp(torch.mean(-value), min=0.0) + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
        else:
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1] # top feature
            if label_negative_list[c].full():
                _ = label_negative_list[c].get()
            label_negative_list[c].put(h)
            if label_positive_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(device)
            else:
                h = label_positive_list[c].get()
                label_positive_list[c].put(h)
                _, _, _, Ah, _ = model(h.detach())
                if c != 0:
                    loss_rank = loss_rank + torch.clamp(torch.mean(value - Ah[0, c]), min=0.0) + torch.clamp(torch.mean(value), min=0.0)
                else:
                    loss_rank = loss_rank + torch.clamp(torch.mean(value), min=0.0) + torch.clamp(torch.mean(-Ah[0, c]), min=0.0)
    loss_rank = loss_rank / n_classes
    return loss_rank, label_positive_list, label_negative_list
