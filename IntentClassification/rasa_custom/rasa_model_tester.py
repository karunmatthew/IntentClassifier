# This program sends the intents to the rasa server that runs
# by default on the 5005 port. The response will contain the
# probability distribution of all the intents

# The rasa server is started by providing the model path
# rasa run --enable-api -m models/20200722-133414.tar.gz

import requests
import json
from math import sqrt

from sklearn.metrics import precision_recall_fscore_support

RASA_SERVER = 'http://localhost:5005/model/parse'
TEST_FILE_PATH = '/home/karun/PycharmProjects/IntentClassification/data-test/' \
                 'testing_set.txt'
READ = 'r'

headers = {
    'Content-type': 'application/json'
}


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

    return [current_agent_pos_x, current_agent_pos_y, current_agent_pos_z,
            dist_to_obj, dist_to_recep]


def read_test_data(file_path):
    file = open(file_path, READ)
    count = 0.0
    correct = 0.0

    predicted_tags = []
    actual_tags = []

    for line in file:
        count += 1
        print(correct, '   ', count)
        print(line)
        json_object = json.loads(line)
        action_sequence = json_object['action_sequence']
        desc = json_object['desc']
        action_sequence_string = ' '.join(action_sequence)
        desc_string = ' '.join(desc)
        desc_string = desc_string.replace('\"', '')
        # visual_data = get_visual_information(json_object['scene_description'])
        desc_string = desc_string + ' '
        # for visual_info in visual_data:
        #    desc_string = desc_string + '@@@@@@' + str(visual_info)
        # need to remove quotes from data input
        data = '{"text": "' + desc_string + '"}'
        print(data)

        response = requests.post(RASA_SERVER, headers=headers, data=data)
        response_json = json.loads(response.text)

        print(response_json)
        intent = response_json['intent']['name']
        confidence = response_json['intent']['confidence']

        predicted_tags.append(intent.strip())
        actual_tags.append(action_sequence_string.strip())

        if intent.strip() == action_sequence_string.strip():
            print('MATCH')
            print(intent)
            print(action_sequence_string)
            correct += 1
        # elif tag == 'NOT_SUPPORTED':
        #    correct += 1
        else:
            print('NOT MATCH')
            print(intent)
            print(action_sequence_string)
            # print(intent, ":", confidence)
            # print(action_sequence_string, ":")

    print('Accuracy :: ', correct/count)
    print('Correct  :: ', correct)
    print('Total    :: ', count)

    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average='macro'))
    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average=None,
                  labels=['GotoLocation',
                          'PickupObject',
                          'PutObject',
                          'GotoLocation PickupObject',
                          'GotoLocation PickupObject GotoLocation',
                          'GotoLocation PickupObject GotoLocation PutObject',
                          'PickupObject GotoLocation',
                          'PickupObject GotoLocation PutObject',
                          'GotoLocation PutObject',
                          'GotoLocation PickupObject PutObject',
                          'PickupObject PutObject']))


read_test_data(TEST_FILE_PATH)
