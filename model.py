# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:48:33 2022

@author: alienware
"""

import numpy as np
import pandas as pd
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from torch.optim import lr_scheduler

# GRUCell
class GRUCell(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(GRUCell, self).__init__()
        self.r = nn.Linear(x_dim + h_dim, h_dim, True)
        self.z = nn.Linear(x_dim + h_dim, h_dim, True)

        self.c = nn.Linear(x_dim, h_dim, True)
        self.u = nn.Linear(h_dim, h_dim, True)

    def forward(self, x, h):
        rz_input = torch.cat((x, h), -1)

        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))

        u = torch.tanh(self.c(x) + r * self.u(h))

        new_h = z * h + (1 - z) * u
        return new_h

# GRU
class GRU(nn.Module):
    def __init__(self, gene_emb, gene_hid,  edge_dim, g_hid):
        '''
            dru_emb: 811
            dis_emb: 935
            tar_emb: 953

            dru_hid: 811
            dis_hid: 935
            tar_hid: 953

            edge_dim: 50

            g_hid: 1000
        '''
        super(GRU, self).__init__()

        # dru_gru: 881 + 881 + (322 + 50) + (1437 + 50) + 1000, 811
        self.gene_gru = GRUCell(gene_emb + gene_hid + g_hid, gene_hid)

        #   g_gru: 881 + 322 + 1437, 1000
        self.g_gru = GRUCell(gene_hid, g_hid)

    def forward(self, i, h, g, mask):
        '''
            dru_i: [811, 881 + 811 + (322 + 50) + (1437 + 50)]
            dis_i: [935, 322 + 322 + (881 + 50)]
            tar_i: [953, 1437 + 1437 + (881 + 50)]

            dru_h: [811, 881]
            dis_h: [935, 322]
            tar_h: [953, 1437]

            dru_m: [811, 811]
            dis_m: [935, 935]
            tar_m: [953, 953]

                g: [1000]
        '''
        gene_i = i
        gene_h = h
        gene_m= mask

        # g_expand_dru: [811, 1000]
        g_expand_gene = g.unsqueeze(0).expand(gene_h.size(0), g.size(-1))
        # x: [811, 881 + 811 + (322 + 50) + (1437 + 50) + 1000]
        x = torch.cat((gene_i, g_expand_gene), -1)
        # new_dru_h: [811, 881]
        new_gene_h = self.gene_gru(x, gene_h)
        # [881]
        gene_h_mean = new_gene_h.sum(0) / new_gene_h.size(0)
        # [881 + 322 + 1437]
        mean = gene_h_mean
        new_g = self.g_gru(mean, g)

        return new_gene_h,new_g

# GRN
class GRNGOB(nn.Module):
    def __init__(self, gene_emb,gene_hid, edge_dim, g_hid, dp=0.1, layer=2,
                 agg='gate', dru_agg='edge', device='cpu'):
        super(GRNGOB, self).__init__()
        self.layer = layer
        self.dp = dp

        self.slstm = GRU(gene_emb, gene_hid, edge_dim, g_hid)

        self.gene_hid = gene_hid
        self.g_hid = g_hid

        self.agg = agg
        self.dru_agg = dru_agg
        self.device = device

        # dru-dru, dru-dis, dru-tar, dis-dis, tar-tar
        self.edgeemb = nn.Embedding(2, edge_dim)

        # dru-dis
        self.gate1 = nn.Linear(gene_hid + gene_hid + edge_dim, gene_hid + edge_dim)
        self.gate2 = nn.Linear(gene_hid + gene_hid + edge_dim, gene_hid + edge_dim)
        # dru_agg param
        if self.dru_agg != 'edge':
            self.weight = nn.Linear(gene_emb, gene_emb)

    def mean(self, x, m, smooth=0):
        mean = torch.matmul(m, x)
        return mean / (m.sum(2, True) + smooth)

    def sum(self, x, m):
        return torch.matmul(m, x)

    def forward(self, gene_emb, mat):
        '''
            dru_emb: [811, 881]
            dis_emb: [935, 322]
            tar_emb: [953, 1437]


            mat: dru_dru_mat, dis_dis_mat, tar_tar_mat, dru_dis_mat, dru_tar_mat
                value:
                    0: none
                    1: dru-dru
                    2: dis-dis
                    3: tar-tar
                    4: dru-dis
                    5: dru-tar
        '''

        gene_size = gene_emb.size(0)

        gene_gene_mat= mat

        # [811, 811]
        gene_gene_mask = (gene_gene_mat != 0).float()
        gene_gene_mask_t = gene_gene_mask.transpose(0, 1)

        # [*, *, 50]
        gene_gene_edge = self.edgeemb(gene_gene_mat)

        gene_gene_edge_t = gene_gene_edge.transpose(0, 1)

        if self.dru_agg == 'edge':
            gene2gene = (gene_gene_mat == 1).float()
        else:
            gene2gene = gene_gene_mat.float()


        gene_h = torch.zeros_like(gene_emb)
        g_h = gene_emb.new_zeros(self.g_hid)

        for i in range(self.layer):

            if self.dru_agg == 'edge':
                gene_neigh_gene_h = self.sum(gene_h, gene2gene)

            else:
                gene_neigh_gene_h = self.weight(self.sum(gene_h, gene2gene))


            # [811, 881 + 811 + (322 + 50) + (1437 + 50)]
            gene_input = torch.cat((gene_emb, gene_neigh_gene_h), -1)


            gene_h,  g_h = self.slstm(gene_input, gene_h, g_h,gene_gene_mask)

        if self.dp > 0:
            gene_h = F.dropout(gene_h, self.dp, self.training)

        return gene_h, g_h

class BilinearDecoder(nn.Module):
    def __init__(self, input_size):
        super(BilinearDecoder, self).__init__()
        self.weight = nn.Linear((input_size), (input_size), bias=False)
        #self.weight1 = nn.Linear((input_size+128), input_size, bias=False)
    
    def forward(self, zu,zg, zv):
        zu = zu.view(1, -1)
        zv = zv.view(1, -1)
        zg = zg.view(1, -1)

        intermediate_product = self.weight(zu)
        ret = torch.matmul(intermediate_product, zv.reshape(-1, 1))

        return torch.sigmoid(ret)

class FFNDecoder(nn.Module):
    def __init__(self, input_size):
        super(FFNDecoder, self).__init__()
        self.weight = nn.Linear((input_size * 2), 64, bias=False)

        self.weight1 = nn.Linear(64, 1, bias=False)

    
    def forward(self, zu, zv):
        zu = zu.view(1, -1)
        zv = zv.view(1, -1)

        ret = torch.relu(self.weight(torch.cat([zu, zv], -1)))
        ret1=self.weight1(ret)

        return torch.sigmoid(ret1)
    
# Model
class Model(nn.Module):
    def __init__(self,
                 gene_emb,
                 gene_hid,
                 edge_dim,
                 g_hid,
                 decoder='FFN',
                 dp=0.1,
                 layer=2,
                 agg='gate',
                 dru_agg='edge',
                 device='cpu'
                 ):
        super(Model, self).__init__()

        self.dp = dp

        self.gene_linear = nn.Linear(gene_emb, gene_hid)

        self.encoder = GRNGOB(gene_hid, gene_hid, edge_dim, g_hid, dp, layer, agg, dru_agg, device)
        
        if decoder == 'FFN':
            self.decoder = FFNDecoder(gene_hid)
        else:
            self.decoder = BilinearDecoder(gene_hid)
        
#         print(self.decoder)

    def forward(self, gene_emb, gene_gene_mat, mask):
        gene_emb = F.relu(self.gene_linear(gene_emb))

        if self.dp > 0:
            gene_emb = F.dropout(gene_emb, self.dp, self.training)

        gene_h, g_h = self.encoder(gene_emb,gene_gene_mat)

        r = torch.zeros_like(gene_gene_mat).float()

        for i, k in torch.nonzero(mask):
            r[i, k] = self.decoder(gene_h[i], gene_h[k])


        return r

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

        self.loss = nn.BCELoss(reduction='sum')

    def forward(self, ipt, target, mask):
        ipt = ipt.contiguous().view(1, -1)
        target = target.contiguous().view(1, -1)
        mask = mask.contiguous().view(1, -1).float()

        ipt = ipt * mask
        target = target * mask

        return self.loss(ipt, target)

def calculate_auc(ipt, tar, mask):
    a, b = [], []
    for i, k in torch.nonzero(mask):
        a.append(ipt[i, k])
        b.append(tar[i, k])
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    a=a.numpy()
    b=b.numpy()
    return roc_auc_score(b, a)

def calculate_aupr(ipt, tar, mask):
    a, b = [], []
    for i, k in torch.nonzero(mask):
        a.append(ipt[i, k])
        b.append(tar[i, k])
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    a=a.numpy()
    b=b.numpy()
    return average_precision_score(b,a)

def calculate_metr(ipt, tar, mask):
    a, b, c= [], [], [] 
    for i, k in torch.nonzero(mask):
        a.append(ipt[i, k])
        b.append(tar[i, k])
    for j in a:
        if j>0.5:
            j=1
            c.append(j)
        else:
            j=0
            c.append(j)
    c = torch.Tensor(c)
    b = torch.Tensor(b)
    c=c.numpy()
    b=b.numpy()
    return accuracy_score(b, c),f1_score(b, c),precision_score(b, c),recall_score(b, c)
    
def print_total_param(model):
    total_params = 0
    for name, parameters in model.named_parameters():
        params = np.prod(list(parameters.size()))
        total_params += params
    print('total parameters: {:.4f}M'.format(total_params / 1e6))

def train(args,data1,data2,data3,data4,data5,data6,data7,data8,data9, cur_fold=None):


    gene_emb = data1
    gene_gene_A = data2
    train_target=data3
    valid_target = data4
    test_target = data5
    train_mask=data6
    valid_mask =data7
    test_mask =data8
    pred_mask=data9


    device = args.device

    gene_emb = gene_emb.to(device)
    
    gene_gene_A=gene_gene_A.to(device)
    
    train_target = train_target.to(device)
    valid_target = valid_target.to(device)
    test_target = test_target.to(device)

    train_mask = train_mask.to(device)
    valid_mask = valid_mask.to(device)
    test_mask = test_mask.to(device)
    pred_mask=pred_mask.to(device)

    model = Model(
        gene_emb=len(gene_emb[0]),
        gene_hid=args.gene_hid_size,
        edge_dim=args.edge_dim,
        g_hid=args.g_hid,
        layer=args.layer,
        dp=args.dp,
        decoder=args.decoder,
        dru_agg=args.dru_agg,
        device=args.device
    )
    model = model.to(device)
    print_total_param(model)

    loss_fn = MaskLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=100,gamma=0.1)

    best_epoch = 1
    best_valid_loss=loss = np.inf
    best_test_loss= np.inf

    t1 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        pred = model(gene_emb,gene_gene_A, train_mask)
        torch.set_printoptions(threshold=np.inf)

        loss = loss_fn(pred,train_target, train_mask)
        auc = calculate_auc(pred, train_target, train_mask)
        aupr = calculate_aupr(pred, train_target, train_mask)
        acc,f1,pr,recall = calculate_metr(pred, train_target, train_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        if epoch % args.print_epochs == 0:
            t2 = time.time()
            print('====================')
            print(f'train epc:{epoch}    loss:{loss.item():.4f}    auc:{auc:.4f}    aupr:{aupr:.4f}    acc:{acc:.4f}    f1:{f1:.4f}    pr:{pr:.4f}    recall:{recall:.4f}    t:{(t2 - t1) / args.print_epochs:.8f}')
            t1 = time.time()
            # print('the current learning rate is:', optimizer.state_dict()['param_groups'][0]['lr'])

        with torch.no_grad():
            model.eval()

            valid_pred = model(gene_emb,gene_gene_A, valid_mask)

            valid_loss = loss_fn(valid_pred, valid_target, valid_mask)
            valid_auc = calculate_auc(valid_pred, valid_target, valid_mask)
            valid_aupr = calculate_aupr(valid_pred, valid_target, valid_mask)
            valid_acc,valid_f1,valid_pr,valid_recall = calculate_metr(valid_pred, valid_target, valid_mask)
                

            test_pred = model(gene_emb,gene_gene_A, test_mask)

            test_loss = loss_fn(test_pred, test_target, test_mask)
            test_auc = calculate_auc(test_pred, test_target, test_mask)
            test_aupr = calculate_aupr(test_pred, test_target, test_mask)
            test_acc,test_f1,test_pr,test_recall = calculate_metr(test_pred, test_target, test_mask)


            if valid_loss.item() < best_valid_loss:# and test_loss.item() < best_test_loss:
                best_valid_loss = valid_loss.item()
                best_test_loss = test_loss.item()
                best_epoch = epoch
                best_valid_auc = valid_auc
                best_valid_aupr = valid_aupr
                best_valid_acc = valid_acc
                best_valid_f1 = valid_f1
                best_valid_pr = valid_pr
                best_valid_re = valid_recall
                best_test_auc = test_auc
                best_test_aupr = test_aupr
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_test_pr = test_pr
                best_test_re = test_recall
                
                print(f'valid_loss improved at epc:{best_epoch}    best_loss:{valid_loss.item():.4f}    valid_auc:{best_valid_auc:.4f}    valid_aupr:{best_valid_aupr:.4f}    valid_acc:{best_valid_acc:.4f}    valid_f1:{best_valid_f1:.4f}    valid_pr:{best_valid_pr:.4f}    valid_re:{best_valid_re:.4f}')
                print(f'test_loss:{test_loss.item():.4f}    test_auc:{best_test_auc:.4f}    test_aupr:{best_test_aupr:.4f}    test_acc:{best_test_acc:.4f}    test_f1:{best_test_f1:.4f}    test_pr:{best_test_pr:.4f}    test_re:{best_test_re:.4f}')
                ckp_save_dir = './model/'
                save_path = '{}/{}_{}-{}.best.pt'.format(ckp_save_dir, args.model, args.num_folds, cur_fold)
                save_path1 = '{}/{}_{}-{}.bestparam.pt'.format(ckp_save_dir, args.model, args.num_folds, cur_fold)
                torch.save(model, save_path)
                torch.save(model.state_dict(), save_path1)
                prediction='True'
                if prediction=='True' and epoch>=50:
                    model.eval()
                    pre_result=model(gene_emb,gene_gene_A, pred_mask)
                    pre_uses = torch.nonzero(pred_mask)
                    pre_use_list = pre_uses.tolist()
                    pre_used_tup = tuple(([tuple(elemt) for elemt in pre_use_list]))
                    pre_used = set(pre_used_tup)
                    for i, k in torch.nonzero(pred_mask):
                        if ((i >= k) and ((i.item(), k.item()) in pre_used) and ((k.item(), i.item()) in pre_used)):
                            t = torch.add(pre_result[i, k], pre_result[k, i])
                            pre_result[i, k] = t / 2
                            pre_result[k, i] = t / 2
                    result=[]
                    for i,k in torch.nonzero(pred_mask):
                        if(i>k):
                            res=[]
                            res.append(i.item())
                            res.append(k.item())
                            res.append(pre_result[i, k].item())
                            result.append(np.array(res))
                    file_name = 'result{}_{}.csv'.format(cur_fold, epoch)
                    save_path='./result/'+file_name
                    results=np.array(result)
                    resultssave=pd.DataFrame(results)
                    resultssave.to_csv(save_path,index=False,encoding=('utf-8'),sep=',')

            else:
                print(' no improvement since epoch ', best_epoch, '; best_valid_loss:', best_valid_loss,'; best_tset_loss:', best_test_loss)

    print('best_epoch is:', best_epoch)
    print('best_valid_loss is:', best_valid_loss)
    print(f'valid_auc:{best_valid_auc:.4f}    valid_aupr:{best_valid_aupr:.4f}    valid_acc:{best_valid_acc:.4f}    valid_f1:{best_valid_f1:.4f}    valid_pr:{best_valid_pr:.4f}    valid_re:{best_valid_re:.4f}')
    print('best_test_loss is:', best_test_loss)
    print('best_auc is:', best_test_auc)
    print('best_aupr is:', best_test_aupr)
    print('best_acc is:', best_test_acc)
    print('best_f1 is:', best_test_f1)
    print('best_pr is:', best_test_pr)
    print('best_re is:', best_test_re)
    return best_test_loss, best_epoch,best_test_auc,best_test_aupr,best_test_acc,best_test_f1,best_test_pr,best_test_re