# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:18:11 2022

@author: alienware
"""
import pandas as pd
import numpy as np
from data_process import parseA, parseA2,parseFea2,parseFea3, parseDataFea,getdatadict
import torch
import argparse
from random import shuffle
from sklearn import preprocessing
import time
import random

from model import train
import os
import warnings
import gc
warnings.filterwarnings("ignore")
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
# 记录日志
sys.stdout = Logger('log.txt', sys.stdout)

def parse_args():
    parser = argparse.ArgumentParser(description='GRN Model Sim Init')
    parser.add_argument('--seed', type=int, default=[42], help='seed for randomness')
    parser.add_argument('--ckp_save_dir', type=str, default='./result/')
    parser.add_argument('--data_dir', type=str, default='./')
    parser.add_argument('--model', type=str, default='[time]')
    parser.add_argument('--th_rate', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gene_hid_size', type=int, default=256)
    parser.add_argument('--edge_dim', type=int, default=16)
    parser.add_argument('--g_hid', type=int, default=64)
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--dp', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--print_epochs', type=int, default=1)
    parser.add_argument('--valid_epochs', type=int, default=100)
    parser.add_argument('--print_param_sum', type=bool, default=True)
    parser.add_argument('--dru_agg', type=str, default='edge')
    parser.add_argument('--decoder', type=str, default='FFN')
    parser.add_argument('--edge_mask', type=int, nargs='+', default=4)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--cur_fold', type=int, default=0)
    return parser.parse_args()

# 计算 Threshold
def cal_threshold(mat, rate):
    for th in np.arange(0, 1, 0.01):
        cur_mat = mat > th
        if sum(cur_mat.sum(axis=0) > (cur_mat.shape[0] * rate)) == 0:
            return th
# 划分数据集
def k_fold_data(mat_pos,mat_neg, num_folds, cur_fold,gene_dict):
    shuffle(mat_pos)
    shuffle(mat_neg)
    mat_negi=mat_neg[:len(mat_pos)]
    test_pos=mat_pos[:int(len(mat_pos)/5)]
    test_neg=mat_negi[:int(len(mat_pos)/5)]
    train_valid_pos=mat_pos[int(len(mat_pos)/5):]
    train_valid_neg=mat_negi[int(len(mat_pos)/5):]
    shuffle(train_valid_pos)
    shuffle(train_valid_neg)    
    fold_size = len(train_valid_pos) // num_folds
    valid_idx = np.arange(cur_fold * fold_size, (cur_fold + 1) * fold_size)

    train_data = np.zeros([len(gene_dict), len(gene_dict)])
    valid_data = np.zeros([len(gene_dict), len(gene_dict)])
    test_data = np.zeros([len(gene_dict), len(gene_dict)])

    for e, (xi, xk) in enumerate(train_valid_pos):
        if e in valid_idx:
            valid_data[xi, xk] =1
            valid_data[xk, xi] =1
        else:
            train_data[xi, xk] =1
            train_data[xk, xi] =1
    for f, (xj, xl) in enumerate(train_valid_neg):
        if f in valid_idx:
            valid_data[xj, xl] = -1
            valid_data[xl, xj] = -1
        else:
            train_data[xj, xl] = -1
            train_data[xl, xj] = -1
    for t,(x1,y1) in enumerate(test_pos):
        test_data[x1,y1]=1
        test_data[y1,x1]=1
    for q,(x2,y2) in enumerate(test_neg):
        test_data[x2,y2]=-1
        test_data[y2,x2]=-1

    return train_data, valid_data,test_data

# 加载数据
def load_data(data_dir, num_folds, cur_fold,rate):
    # load data
    gene_dict,gene_name,gene_identifier = getdatadict('a549_name.csv', path=data_dir + 'label/')

    fea1_dict, idx_fea1_dict, idx_fea1_fea_dict, idx_fea1_fea_dict1 = parseDataFea('fea1-filt-a549-name.csv',
                                                                                    path=data_dir + 'feature/fea1/')
    fea1, exist1 = parseFea2(fea1_dict, gene_name, idx_fea1_fea_dict, idx_fea1_fea_dict1)
    gene_gene_mat,gene_gene_pred_mat,gene_gene_d = parseA('A549.csv', gene_name, exist1, path=data_dir + 'label/')
    
    fea2_dict, idx_fea2_dict, idx_fea2_fea_dict,idx_fea2_fea_dict1 = parseDataFea('ALL-NAME.csv', path=data_dir + 'feature/1234NEW/')
    fea2, exist2=parseFea2(fea2_dict,gene_name,idx_fea2_fea_dict,idx_fea2_fea_dict1)

    fea3_dict, idx_fea3_dict, idx_fea3_fea_dict, idx_fea3_fea_dict1 = parseDataFea('gene-gene-sim-a549.csv',
                                                                                   path=data_dir + 'feature/fea1/')
    fea3, exist3 = parseFea2(fea3_dict, gene_name, idx_fea3_fea_dict, idx_fea3_fea_dict1)

    location=[]
    location_zero=[]
    x=0
    for i in range(len(exist1)):
        for j in range(len(exist1)):
            if i>j:
                if (gene_gene_mat[i,j] !=0):
                    location.append([i,j])
                    x=x+1
                else:
                    location_zero.append([i,j])
    
    train_gene_gene_mat, valid_gene_gene_mat,test_gene_gene_mat = k_fold_data(location,location_zero, num_folds, cur_fold,exist1)

    gene_gene_sim1 = torch.zeros(len(exist3), len(fea3[0]))
    for i in range(len(exist3)):
        gene_gene_sim1[i] = torch.FloatTensor(fea3[i])
    
    train_gene_gene_mat =torch.LongTensor(train_gene_gene_mat)
    valid_gene_gene_mat=torch.LongTensor(valid_gene_gene_mat)
    test_gene_gene_mat = torch.LongTensor(test_gene_gene_mat)
    gene_gene_pred_mat=torch.LongTensor(gene_gene_pred_mat)
    gene_gene_d_mat = torch.LongTensor(gene_gene_d)

    
    threshold = cal_threshold(gene_gene_sim1, rate)
    print('threshold:','%.5f'% threshold)

    train_mask = train_gene_gene_mat != 0
    valid_mask = valid_gene_gene_mat != 0
    test_mask = test_gene_gene_mat != 0
    pred_mask=gene_gene_pred_mat != 0


    train_gene_gene=torch.zeros_like(train_gene_gene_mat)
    train_gene_gene[train_gene_gene_mat==1]=1
    valid_gene_gene=torch.zeros_like(valid_gene_gene_mat)
    valid_gene_gene[valid_gene_gene_mat==1]=1
    test_gene_gene=torch.zeros_like(test_gene_gene_mat)
    test_gene_gene[test_gene_gene_mat==1]=1

    gene_gene_adj=torch.zeros_like(gene_gene_sim1)
    gene_gene_adj[gene_gene_sim1 > threshold]=1
    gene_gene_adj=gene_gene_adj.long()

    gene_gene_A=torch.add(gene_gene_adj,gene_gene_d_mat)
    gene_emb = torch.zeros(len(exist2), len(fea2[0]))
    for i in range(len(exist2)):
        gene_emb[i] = torch.FloatTensor(fea2[i])

    print('-------------------  data info  -----------------')
    print('gene_size:', len(exist1))
    print('gene_fea1_size:', len(fea1[0]))
    print('gene_fea2_size:', len(fea2[0]))
    print('used_size:', gene_emb.shape)

    return gene_emb,gene_gene_A,train_gene_gene,valid_gene_gene,test_gene_gene,train_mask, valid_mask,test_mask,pred_mask



def curtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = parse_args()
    for key in args.__dict__:
        print(f"{key}: {args.__dict__[key]}")
    for seed_train in args.seed:
        print('cur time: ', curtime())
        print('-------------------  mode: train  -----------------')
        print('cur seed:',seed_train)
        set_seeds(seed_train)
    
        if args.model == '[time]':
            args.model = time.strftime("%m.%d_%H.%M.", time.gmtime())
    
        print('load data from: {} '.format(args.data_dir))
        print(f'train with {args.num_folds} fold')
        auc=[]
        aupr = []
        acc = []
        f1 = []
        pr = []
        re = []
        for cur_fold in range(args.num_folds):
            print(f'Start fold {cur_fold}/{args.num_folds}')
            data1,data2,data3,data4,data5,data6,data7,data8,data9 = load_data(args.data_dir, args.num_folds, cur_fold,args.th_rate)
            best_loss, best_epoch,best_auc,best_aupr,best_acc,best_f1,best_pr,best_re=train(args,data1,data2,data3,data4,data5,data6,data7,data8,data9, cur_fold)
            auc.append(best_auc)
            aupr.append(best_aupr)
            acc.append(best_acc)
            f1.append(best_f1)
            pr.append(best_pr)
            re.append(best_re)
            print(f'Finished fold {cur_fold}/{args.num_folds}')
    
        print('AUC:',auc)
        print('AUPR:',aupr)
        print('ACC:',acc)
        print('F1:',f1)
        print('PR:',pr)
        print('RE:',re)
        auc_mean=np.mean(auc)
        auc_std=np.std(auc)
        aupr_mean=np.mean(aupr)
        aupr_std=np.std(aupr)
        acc_mean=np.mean(acc)
        acc_std=np.std(acc)
        f1_mean=np.mean(f1)
        f1_std=np.std(f1)
        pr_mean=np.mean(pr)
        pr_std=np.std(pr)
        re_mean=np.mean(re)
        re_std=np.std(re)
        print("auc_ave:%.6f, auc_std: %.6f, aupr_ave:%.6f, aupr_std:%.6f,acc_ave:%.6f, acc_std: %.6f,f1_ave:%.6f, f1_std: %.6f,pr_ave:%.6f, pr_std: %.6f,re_ave:%.6f, re_std: %.6f" % (auc_mean, auc_std, aupr_mean, aupr_std, acc_mean, acc_std, f1_mean, f1_std, pr_mean, pr_std, re_mean, re_std))
