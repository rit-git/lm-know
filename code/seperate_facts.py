import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import ttest_rel
import torch
import json
import re
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer, AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score
from tqdm import tqdm


def json_process(path):
    f = open(path)
    test_data = json.load(f)
    out = {}
    dictb = {}

    for label in test_data:
        dict_rel = {}
        for samp in test_data[label]:
            rel = samp['rel']
            if rel not in dict_rel:
                dict_rel[rel] = 1
            else:
                dict_rel[rel] += 1

            if len(samp['gen_para'])<20:
                if label not in dictb:
                    dictb[label] = 1
                else:
                    dictb[label] += 1


            # else:
            #     if label == '2':
            #         print(samp['prompt'])
            #         print(samp['gen_para'])
            #         print('--------------')

        out[label] = [len(test_data[label])]#, dict_rel]

    print(dictb)
    print(out)


    return out


def json_process_sub(path):
    f = open(path)
    test_data = json.load(f)
    data = []

    num = 0

    dict_label = {}

    for label in test_data:
        dict_rel = {}
        for samp in test_data[label]:
            num += 1
            temp = {'label':label}
            if (label== '0' and samp['prob']>0.9) or (label== '1' and samp['prob']>0.95) or (label == '2' and samp['prob']>0.98):
                for kk in samp:
                    temp[kk] = samp[kk]
                data += [temp]

                if label not in dict_label:
                    dict_label[label] = 1
                else:
                    dict_label[label] += 1

    print(dict_label)

    return data


if __name__ == '__main__':

    out = json_process_sub('out/pred_flan.json')
    print(len(out))

    with open('out/filtered_pred_flan.jsonl', 'w') as outfile:
        for entry in out:
            json.dump(entry, outfile)
            outfile.write('\n')



