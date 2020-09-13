from transformers import BertModel
from transformers import BertTokenizer
import torch
from util.apputil import get_data
import spacy
import json
import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as func

TRAIN_DATA_PATH = '../data-train/training_set.txt'
TEST_BERT_FILE = '../data-test/test_bert.txt'


class IntentClassifier(nn.Module):

    def __init__(self):
        super(IntentClassifier, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-large-uncased')

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768
        self.linear1 = nn.Linear(768, 20)
        # output layer
        self.linear2 = nn.Linear(20, 8)
        self.cls_layer = 9

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        contextualized_reps, _ = self.bert_layer(seq, attention_mask=attn_masks)

        # Obtaining the representation of [CLS] head (the first token)
        cls_rep = contextualized_reps[:, 0]

        # Feeding cls_rep to the classifier layer
        out = self.linear1(cls_rep)
        # Apply an activation function
        out = func.relu(out)
        out = self.linear2(out)
        out = func.softmax(out)

        return out


gpu = 0 #gpu ID
net = IntentClassifier()
net.cuda(gpu)





for t in model.parameters():
    print(t.shape)