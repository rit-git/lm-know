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

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = AutoModel.from_pretrained("roberta-base")#('sentence-transformers/nli-roberta-base-v2')#RobertaModel.from_pretrained("roberta-base")
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output = self.classifier(pooler)
        return output



def evaluate(model, testing_loader):
    model.eval()
    y_pred = []
    y_true = []

    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            y_pred += big_idx.tolist()
            y_true += targets.tolist()
            

    print('accuracy:', accuracy_score(y_true, y_pred))
    # print('F1:', f1_score(y_true, y_pred))
    
     

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")




def roberta_preprocessing(path, tokenizer, max_len):
    data = []
    with open(path) as f:
        for line in f:
            dict_temp = json.loads(line)
            fact = dict_temp['fact']
            prompt = dict_temp['prompt']
            doc = fact + ' </s> ' + prompt
            label = dict_temp['label']
            data.append([doc, label])

    process = []
    for pair in data:
        text = pair[0]

        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        process += [{
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(pair[1], dtype=torch.float)
        }]
    return process

if __name__ == '__main__':

    MAX_LEN = 512
    batch_size = 8
    VALID_BATCH_SIZE = 20
    lr = 1e-05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

    train_data = roberta_preprocessing('data/train.jsonl', tokenizer, MAX_LEN)
    test_data = roberta_preprocessing('data/test.jsonl', tokenizer, MAX_LEN)

    print(len(train_data), len(test_data))


    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }

    training_loader = DataLoader(train_data, **train_params)
    testing_loader = DataLoader(test_data, **test_params)

    model = RobertaClass().to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr)

    EPOCHS = 4
    evaluate(model, testing_loader)
    for epoch in range(EPOCHS):
        print("epoch", epoch)
        train(epoch)
        evaluate(model, testing_loader)
        print("**********************")

    path = 'checkpoint/discriminator.pt'
    torch.save(model.state_dict(), path)
    print('Finished Training!')


