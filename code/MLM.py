import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from tqdm import tqdm
import scipy.stats
import logging
import json
import numpy as np
import random
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)# OPTIONAL
import os
import warnings
warnings.filterwarnings("ignore")

random.seed(1234)

def preprocess(path_data, tokenizer, mode = "train"):
    documents = []
    labels = []
    with open(path_data) as f:
        for line in f:
            dict_line = json.loads(line)
            # if mode == 'train':
            #     doc = dict_line['prompt-mod']
            # else:
            doc = dict_line['prompt']
            label = doc.replace("[Y]", dict_line['obj'])
            mask_str = " ".join(['[MASK]']*len(tokenizer.encode(dict_line['obj'],add_special_tokens=False,return_tensors='pt')[0]))
            doc = doc.replace("[Y]", mask_str)
            documents += [doc]    
            labels += [label]
    return documents, labels


def bert_preprocessing(path_data, tokenizer, mode = 'train'):
    documents, label_list = preprocess(path_data, tokenizer, mode)
    print(documents[:5])
    predict_token_id = tokenizer.convert_tokens_to_ids('[MASK]')

    ind = 0
    data = []
    data_list = []

    for sent in documents:
        data_list += [sent]
        model_inputs = tokenizer(
            sent,
            add_special_tokens=True,
            max_length=50,
            return_tensors='pt',truncation=True, padding='max_length')

        label_tokens = tokenizer.encode(
            label_list[ind],
            add_special_tokens=True,
            max_length=50,
            return_tensors='pt',truncation=True, padding='max_length')

        input_ids = model_inputs['input_ids']
        predict_mask = input_ids.eq(predict_token_id)

        model_inputs['predict_mask'] = predict_mask
        model_inputs["labels"] = label_tokens

        data.append((model_inputs))

        ind += 1

    return data, data_list


def evaluate(model, test_loader, metric=False):
    hit_1 = 0.0
    num = 0
    model.eval()
    ent_all = 0.0
    running_loss = 0.0

    with torch.no_grad():
        iter_ = tqdm(test_loader)
        for i, batch in enumerate(iter_): 
            input_ids = batch['input_ids'].to(device).squeeze()
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)

            loss, preds = outputs.loss, outputs.logits
            running_loss += loss.item()

            if metric:
                for i in range(len(input_ids)):
                    pred = preds[i].unsqueeze(0)[batch['predict_mask'][i]]
                    label = batch['labels'][i][batch['predict_mask'][i]]
                    
                    rank_temp = 0.0
                    ent_temp = 0.0
                    for k in range(len(pred)):
                        val_l = pred[k][label[k]]
                        sort_digit = sorted(pred[k], reverse = True)
                        rank_temp += sort_digit.index(val_l)+1
                        prob = torch.nn.functional.softmax(pred[k], dim=-1).numpy()
                        ent_temp += scipy.stats.entropy(prob) 
                    if rank_temp/len(pred) <= 1:
                        hit_1 += 1

            #     ent_all += ent_temp/len(pred)

                    num += 1
    
    average_test_loss = running_loss/len(test_loader)
    print("test-loss:", average_test_loss)
    if metric:
        print('hits@1:', hit_1/num, hit_1, num)
    # print('entropy:', ent_all/num)
        return hit_1/num
    else:
        return 0


def train(model,
          optimizer,
          train_loader,
          valid_loader,
          num_epochs = 6, 
          best_valid_loss = float("Inf")):

    # training loop


    best_valid = 0#evaluate(model, valid_loader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        iter_ = tqdm(train_loader)
        for i, batch in enumerate(iter_): 
            input_ids = batch['input_ids'].to(device).squeeze()
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)

            loss, preds = outputs.loss, outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()

        average_train_loss = running_loss/len(train_loader)
        # evaluation step
        if epoch %2 == 0:
            model.eval()
            valid_eva = evaluate(model, valid_loader)


        # print progress
        print('Epoch [{}/{}], Train Loss: {:.4f}'
              .format(epoch+1, num_epochs,
                      average_train_loss))
            
        # checkpoint
        # if best_valid < valid_eva:
        #     best_valid = valid_eva
        #     path = 'checkpoint/model.pt'
        #     torch.save(model.state_dict(), path)

    # model.eval()
    # valid_eva = evaluate(model, valid_loader, metric = True)        
    path = 'checkpoint/model.pt'
    torch.save(model.state_dict(), path)
    print('Finished Training!')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

trig_size = 10
batch_size = 10

train_data, train_set = bert_preprocessing('data/train.jsonl', tokenizer)
val_data, valid_set = bert_preprocessing('data/test.jsonl',tokenizer, mode = 'test')

print(len(train_data), len(val_data))

train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

val_sampler = RandomSampler(val_data)
valid_loader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size, shuffle = False)

# model.load_state_dict(torch.load("checkpoint/model.pt"))
# model.eval()
# evaluate(model, valid_loader, metric = True)
# print(p)



model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)#2e-5
train(model=model, optimizer=optimizer, train_loader = train_loader,
      valid_loader = valid_loader)

