# -*- coding: utf-8 -*-

import numpy as np

def parseDataFea(file, path='data/T98G/'):
    data_dict = {}
    idx_data_dict = {}
    idx_data_fea_dict = {}
    idx_data_fea_dict1 = {}
    with open(path + file, 'r') as f:
        for i, line in enumerate(f.readlines()[1:]):
            line = line.strip(' \n ').split(',')
            name = line[0].strip('"')
            fea = [float(i) for i in line[1:]]

            data_dict[name] = i
            idx_data_dict[i] = name
            idx_data_fea_dict[name] = fea
            idx_data_fea_dict1[i] = fea

    return data_dict, idx_data_dict, idx_data_fea_dict, idx_data_fea_dict1


def parseA(file, a_dict, exist, path):
    pos = [0, 1]
    dtype = np.int_
    b_dict = {}
    j = 0
    for i in a_dict:
        if i in exist:
            b_dict[i] = j
            j = j + 1

    mat = np.zeros([len(b_dict), len(b_dict)], dtype=dtype)
    mat_pred=np.ones([len(b_dict), len(b_dict)], dtype=dtype)
    mat_d = np.eye(len(b_dict), dtype=dtype)
    with open(path + file, 'r') as f:
        for line in f.readlines()[1:]:

            line = line.strip(' \n ').split(',')
            a = line[pos[0]].strip('"')
            b = line[pos[1]].strip('"')

            if a not in exist or b not in exist:
                continue
            mat[b_dict[a], b_dict[b]] = 1
            mat[b_dict[b], b_dict[a]] = 1
            mat_pred[b_dict[a], b_dict[b]]=0
            mat_pred[b_dict[b], b_dict[a]]=0
    return mat,mat_pred,mat_d


def parseA2(file, a_dict, exist, path):
    pos = [1, 3]
    dtype = np.int_
    b_dict = {}
    j = 0
    for i in a_dict:
        if i in exist:
            b_dict[i] = j
            j = j + 1

    mat = np.zeros([len(b_dict), len(b_dict)], dtype=dtype)

    with open(path + file, 'r') as f:
        for line in f.readlines()[1:]:

            line = line.strip(' \n ').split(',')
            a = line[pos[0]].strip('"')
            b = line[pos[1]].strip('"')

            if a not in exist or b not in exist:
                continue

            mat[b_dict[a], b_dict[b]] = 1

    return mat


def parseFea2(fea_dict,gene_name,idx_fea_dict,idx_fea_dict1):
    fea_dictionary={}
    exist=[]
    for i in fea_dict:
        if  i in gene_name:
            exist.append(i)
    b_dict={}
    j=0
    for i in gene_name:
        if i in exist:
            b_dict[i]=j
            j=j+1
    for i in gene_name:
        if i in exist:
            fea_dictionary[b_dict[i]]=idx_fea_dict[i]

    return fea_dictionary,exist


def parseFea3(fea_dict, gene_dict, idx_fea_dict, idx_fea_dict1):
    fea_dictionary = {}
    exist = []
    for i in fea_dict:
        if i in gene_dict:
            exist.append(i)
    b_dict={}
    j=0
    for i in gene_dict:
        if i in exist:
            b_dict[i]=j
            j=j+1
    for i in gene_dict:
        if i in exist:
            fea_dictionary[b_dict[i]] = idx_fea_dict[i]
    return fea_dictionary, exist


def getdatadict(file, path='data/T98G/'):
    gene_identifier = {}
    gene_name = {}
    data_dict = {}
    with open(path + file, 'r') as f:
        for i, line in enumerate(f.readlines()[1:]):
            line = line.strip(' \n ').split(',')
            name = line[0].strip('"')
            identifier = line[1].strip('"')
            data_dict[identifier] = i
            gene_name[name] = i
            gene_identifier[i] = identifier
    return data_dict, gene_name, gene_identifier