import torch
from util.apputil import get_data
import spacy
import json

TRAIN_DATA_PATH = '../data-train/training_set.txt'
TEST_DATA_PATH = '../data-test/testing_set.txt'
TRAIN_BERT_FILE = '../data-train/train_bert.txt'
TEST_BERT_FILE = '../data-test/test_bert.txt'

TRAIN_MLP_FILE = '../data-train/train_mlp.txt'
TEST_MLP_FILE = '../data-test/test_mlp.txt'
WRITE = 'w'

labels = {'GotoLocation': 0, 'PickupObject': 1, 'PutObject': 2,
          'GotoLocation PickupObject': 3,
          'GotoLocation PickupObject GotoLocation': 4,
          'GotoLocation PickupObject GotoLocation PutObject': 5,
          'PickupObject GotoLocation': 6,
          'PickupObject GotoLocation PutObject': 7, 'GotoLocation PutObject': 8,
          'GotoLocation PickupObject PutObject': 9,
          'PickupObject PutObject': 10}


def create_BERT_compliant_dataset(input_file_name, out_file_name):
    out_file = open(out_file_name, WRITE)
    raw_train_data = get_data(input_file_name)
    out_file.write('desc' + '\t' + 'intent' + '\n')
    for datum in raw_train_data:
        json_data = json.loads(datum)
        desc = '[CLS] ' + '[SEP]'.join(json_data['desc']).strip() + '[SEP]'
        intent = '[CLS] ' + '[SEP]'.join(json_data['action_sequence']).strip() + \
                 '[SEP]'

        out_file.write(desc + '\t' + intent + '\n')
    out_file.close()


def create_mlp_compliant_dataset(input_file_name, out_file_name):
    out_file = open(out_file_name, WRITE)
    raw_train_data = get_data(input_file_name)
    out_file.write('desc' + '\t' + 'intent' + '\n')
    for datum in raw_train_data:
        json_data = json.loads(datum)
        desc = ' '.join(json_data['desc']).strip()
        desc = desc.replace('\n', '')
        intent = ' '.join(json_data['action_sequence']).strip()
        intent = labels[intent]

        out_file.write(desc + '\t' + str(intent) + '\n')
    out_file.close()


# create_BERT_compliant_dataset(TRAIN_DATA_PATH, TRAIN_BERT_FILE)
# create_BERT_compliant_dataset(TEST_DATA_PATH, TEST_BERT_FILE)

create_mlp_compliant_dataset(TRAIN_DATA_PATH, TRAIN_MLP_FILE)
create_mlp_compliant_dataset(TEST_DATA_PATH, TEST_MLP_FILE)

