from math import sqrt

from util.apputil import get_data

import json
import random

TRAIN_DATA_PATH = '../data-train/training_set.txt'
DEV_DATA_PATH = '../data-train/dev_set.txt'
TEST_DATA_PATH = '../data-test/testing_set.txt'

TRAIN_BERT_FILE = '../data-train/train_bert.txt'
TEST_BERT_FILE = '../data-test/test_bert.txt'

TRAIN_MLP_FILE = '../data-train/train_mlp.txt'
TEST_MLP_FILE = '../data-test/test_mlp.txt'

TRAIN_MLP_FULL_FILE = '../data-train/train_mlp_full.txt'
DEV_MLP_FULL_FILE = '../data-train/dev_mlp_full.txt'
TEST_MLP_FULL_FILE = '../data-test/test_mlp_full.txt'

WRITE = 'w'

intent_count = {}

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


def create_mlp_full_dataset(input_file_name, out_file_name, sample_rate):
    out_file = open(out_file_name, WRITE)
    raw_train_data = get_data(input_file_name)
    out_file.write(
        'desc' + '\t' + 'agent_pos_x' + '\t' + 'agent_pos_y' + '\t' +
        'agent_pos_z' + '\t' + 'dist_to_obj' + '\t' +
        'dist_to_recep' + '\t' + 'dist_obj_to_recep' +
        '\t' + 'intent' + '\n')

    for datum in raw_train_data:
        json_data = json.loads(datum)

        # sample the data at the provided rate
        number = random.randint(0, 100)
        if number > sample_rate:
            continue

        desc = ' '.join(json_data['desc']).strip()
        # added as there were records which had \n in it
        desc = desc.replace('\n', '')
        visual_data = get_visual_information(json_data['scene_description'])
        intent = ' '.join(json_data['action_sequence']).strip()
        intent = labels[intent]

        if intent in intent_count:
            intent_count[intent] = intent_count[intent] + 1
        else:
            intent_count[intent] = 1

        out_file.write(desc + '\t')

        for v_data in visual_data:
            out_file.write(str(v_data) + '\t')
        out_file.write(str(intent) + '\n')

    out_file.close()


def get_visual_information(scene_desc):
    dist_to_obj = 100
    dist_to_recep = 100
    obj_relevant = 0
    recep_relevant = 0
    current_agent_pos_x = 0
    current_agent_pos_y = 0
    current_agent_pos_z = 0

    for entry in scene_desc:
        if entry['entityName'] == 'agent':
            current_agent_pos_x = entry['position'][0]
            current_agent_pos_y = entry['position'][1]
            current_agent_pos_z = entry['position'][2]

    agent_pos = [current_agent_pos_x, current_agent_pos_y, current_agent_pos_z]

    object_pos = [0, 0, 0]
    recep_pos = [0, 0, 0]

    for entry in scene_desc:
        if not entry['entityName'] == 'agent' and entry['object_type'] == \
                'simple':
            obj_relevant = entry['relevant']
            object_pos = entry['position']
            dist_to_obj = sqrt(pow(current_agent_pos_x - object_pos[0], 2) +
                               pow(current_agent_pos_y - object_pos[1], 2) +
                               pow(current_agent_pos_z - object_pos[2], 2))
        elif not entry['entityName'] == 'agent' and entry['object_type'] == \
                'receptable':
            recep_relevant = entry['relevant']
            recep_pos = entry['position']
            dist_to_recep = sqrt(pow(current_agent_pos_x - recep_pos[0], 2) +
                                 pow(current_agent_pos_y - recep_pos[1], 2) +
                                 pow(current_agent_pos_z - recep_pos[2], 2))

    dist_obj_to_recep = sqrt(pow(recep_pos[0] - object_pos[0], 2) +
                             pow(recep_pos[1] - object_pos[1], 2) +
                             pow(recep_pos[2] - object_pos[2], 2))

    return [current_agent_pos_x, round(current_agent_pos_y, 2), current_agent_pos_z,
            round(dist_to_obj, 2), round(dist_to_recep, 2), round(dist_obj_to_recep, 2)]


# create_BERT_compliant_dataset(TRAIN_DATA_PATH, TRAIN_BERT_FILE)
# create_BERT_compliant_dataset(TEST_DATA_PATH, TEST_BERT_FILE)

# create_mlp_compliant_dataset(TRAIN_DATA_PATH, TRAIN_MLP_FILE)
# create_mlp_compliant_dataset(TEST_DATA_PATH, TEST_MLP_FILE)

intent_count = {}
create_mlp_full_dataset(TRAIN_DATA_PATH, TRAIN_MLP_FULL_FILE, 10)
print(intent_count)

intent_count = {}
create_mlp_full_dataset(TEST_DATA_PATH, TEST_MLP_FULL_FILE, 100)
print(intent_count)

intent_count = {}
create_mlp_full_dataset(DEV_DATA_PATH, DEV_MLP_FULL_FILE, 100)
print(intent_count)
