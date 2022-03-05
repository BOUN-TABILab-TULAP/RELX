from flask import Flask, json, g, request, jsonify, json
from random import choice, randint
from transformers import BertModel, get_linear_schedule_with_warmup, AdamW, BertTokenizer, AutoTokenizer, AutoModel
import math
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import random
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from datetime import datetime

modelpath = "relx_finetuned_model.pt"
categories = ['no_relation', 'org:alternate_names(e1,e2)', 'org:alternate_names(e2,e1)', 'org:city_of_headquarters(e1,e2)', 'org:city_of_headquarters(e2,e1)', 'org:country_of_headquarters(e1,e2)', 'org:country_of_headquarters(e2,e1)', 'org:founded(e1,e2)', 'org:founded(e2,e1)', 'org:founded_by(e1,e2)', 'org:founded_by(e2,e1)', 'org:members(e1,e2)', 'org:members(e2,e1)', 'org:stateorprovince_of_headquarters(e1,e2)', 'org:stateorprovince_of_headquarters(e2,e1)', 'org:subsidiaries(e1,e2)', 'org:subsidiaries(e2,e1)', 'org:top_members/employees(e1,e2)',
              'org:top_members/employees(e2,e1)', 'per:alternate_names(e1,e2)', 'per:alternate_names(e2,e1)', 'per:cities_of_residence(e1,e2)', 'per:cities_of_residence(e2,e1)', 'per:countries_of_residence(e1,e2)', 'per:countries_of_residence(e2,e1)', 'per:country_of_birth(e1,e2)', 'per:country_of_birth(e2,e1)', 'per:employee_of(e1,e2)', 'per:employee_of(e2,e1)', 'per:origin(e1,e2)', 'per:origin(e2,e1)', 'per:spouse(e1,e2)', 'per:spouse(e2,e1)', 'per:stateorprovinces_of_residence(e1,e2)', 'per:stateorprovinces_of_residence(e2,e1)', 'per:title(e1,e2)', 'per:title(e2,e1)']

max_seq_length = 256
tokenizer = AutoTokenizer.from_pretrained("akoksal/MTMB")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.is_available())


class Model(nn.Module):
    def __init__(self, is_embedding_layer_free=True, last_free_layer=0, no_classes=37, has_layer_norm=False, has_dropout=True):
        super(Model, self).__init__()
        self.net_bert = AutoModel.from_pretrained("akoksal/MTMB")
        self.has_layer_norm = has_layer_norm
        self.has_dropout = has_dropout
        self.no_classes = no_classes
        unfrozen_layers = ["classifier", "pooler"]
        if is_embedding_layer_free:
            unfrozen_layers.append('embedding')

        last_layer = 12
        hidden_size = 768

        for idx in range(last_free_layer, last_layer):
            unfrozen_layers.append('encoder.layer.'+str(idx))

        for name, param in self.net_bert.named_parameters():
            if not any([layer in name for layer in unfrozen_layers]):
                print("[FROZE]: %s" % name)
                param.requires_grad = False
            else:
                print("[FREE]: %s" % name)
                param.requires_grad = True
        if self.has_layer_norm:
            self.fc1 = nn.LayerNorm(hidden_size)
        if self.has_dropout:
            self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, self.no_classes)

    def forward(self, x, attention):
        x, _ = self.net_bert(x, attention_mask=attention)
        # Getting head
        x = x[:, 0, :]
        if self.has_dropout:
            x = self.dropout(x)
        if self.has_layer_norm:
            x = self.fc1(x)
        x = self.fc2(x)
        return x

    def evaluate(self, X, attention, y, criterion, device, other_class=0, batch_size=32):
        with torch.no_grad():
            outputs = torch.tensor([], device=device)
            for idx in range(math.ceil(len(X)/batch_size)):
                inputs_0 = X[idx *
                             batch_size:min(len(X), (idx+1)*batch_size)].to(device)
                input_attention = attention[idx*batch_size:min(
                    len(attention), (idx+1)*batch_size)].to(device)
                outputs = torch.cat(
                    (outputs, self(inputs_0, input_attention)), 0)
        _, predicted = torch.max(outputs.data, 1)
        total = y.size(0)
        correct = (predicted == y.to(device)).sum().item()
        accuracy = correct/total
        loss = criterion(outputs, y.to(device)).item()
        if self.no_classes == 37 and other_class == 0:
            t = 0
            for i in range(18):
                t += f1_score(y.cpu(), predicted.cpu(),
                              average='micro', labels=[2*i+1, 2*i+2])
            f1 = t/18
        else:
            print(
                f'Evaluation should be added manually for {self.no_classes} classes and other class #{other_class}')
            return 0, 0, 0, np.array(predicted.cpu())
        return accuracy, f1, loss, np.array(predicted.cpu())


model = Model().to(device)
map_location = torch.device('cpu')
x = torch.load(modelpath, map_location=map_location)
model.load_state_dict(x["model_state_dict"])
model.eval()


def to_id(text):
    new_text = []
    for word in text.split():
        if word.startswith('http'):
            continue
        elif word.startswith('www'):
            continue
        elif word.startswith('**********'):
            continue
        elif word.startswith('-------'):
            continue
        new_text.append(word)
    text = ' '.join(new_text)
    if text.index('<e1>') < text.index('<e2>'):
        fc = 'e1'
        sc = 'e2'
        be1 = 1
        le1 = 2
        be2 = 3
        le2 = 4
    else:
        fc = 'e2'
        sc = 'e1'
        be1 = 3
        le1 = 4
        be2 = 1
        le2 = 2
    initial = tokenizer.encode(
        text[:text.index(f'<{fc}>')].strip(), add_special_tokens=False)
    e1 = tokenizer.encode(text[text.index(
        f'<{fc}>')+4:text.index(f'</{fc}>')].strip(), add_special_tokens=False)
    middle = tokenizer.encode(text[text.index(
        f'</{fc}>')+5:text.index(f'<{sc}>')].strip(), add_special_tokens=False)
    e2 = tokenizer.encode(text[text.index(
        f'<{sc}>')+4:text.index(f'</{sc}>')].strip(), add_special_tokens=False)
    final = tokenizer.encode(
        text[text.index(f'</{sc}>')+5:].strip(), add_special_tokens=False)
    return torch.tensor([101]+initial+[be1]+e1+[le1]+middle+[be2]+e2+[le2]+final+[102])


def evaluater(sentences):
    with torch.no_grad():
        input_ids_all = []
        input_attentions = []
        for sentence in sentences.splitlines():
            input_ids_raw = to_id(sentence)[:max_seq_length]
            input_attention = torch.LongTensor(
                [1]*len(input_ids_raw)+[0]*(max_seq_length-len(input_ids_raw)))
            input_attention = input_attention.to(device)
            input_ids = torch.cat((input_ids_raw, torch.tensor(
                [0]*(max_seq_length-len(input_ids_raw)))), 0).to(device)
            input_ids_all.append(input_ids)
            input_attentions.append(input_attention)
        input_attentions = torch.stack(input_attentions)
        input_ids_all = torch.stack(input_ids_all)
        outputs = model(input_ids_all, input_attentions)
        class_ids = np.argmax(outputs.cpu(), axis=1)
        # print(class_ids)
        return [categories[class_id] for class_id in class_ids]


app = Flask(__name__)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    json_data = json.loads(request.data)

    result = {"text": "\n".join(evaluater(json_data['textarea']))}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0')
