# This program sends the intents to the rasa server that runs
# by default on the 5005 port. The response will contain the
# probability distribution of all the intents

# The rasa server is started by providing the model path
# rasa run --enable-api -m models/20200722-133414.tar.gz
import random

import requests
import json
import copy
from math import sqrt
from util.apputil import get_dot_product_score

from sklearn.metrics import precision_recall_fscore_support

RASA_SERVER = 'http://localhost:5005/model/parse'
# TEST_FILE_PATH = '/home/karun/PycharmProjects/IntentClassification/data-train/' \
#                 'dev_set.txt'
TEST_FILE_PATH = '/home/student.unimelb.edu.au/kvarghesemat/PycharmProjects/IntentClassifier/IntentClassification' \
                 '/data-train/dev_set.txt'

READ = 'r'

WITH_VISUAL = True
CONSIDER_ROTATION = True

headers = {
    'Content-type': 'application/json'
}


def get_visual_information(scene_desc):
    dist_to_obj = -1
    dist_to_recep = -1
    dist_obj_to_recep = -1
    obj_relevant = 0
    recep_relevant = 0
    current_agent_pos_x = 0
    current_agent_pos_y = 0
    current_agent_pos_z = 0
    agent_orientation = 0

    for entry in scene_desc:
        if entry['entityName'] == 'agent':
            current_agent_pos_x = round(entry['position'][0], 2)
            current_agent_pos_y = round(entry['position'][1], 2)
            current_agent_pos_z = round(entry['position'][2], 2)

            if len(entry['position']) == 6 and CONSIDER_ROTATION:
                agent_orientation = round(entry['position'][4], 2)

    object_pos = [0, 0, 0]
    recep_pos = [0, 0, 0]

    for entry in scene_desc:
        if not entry['entityName'] == 'agent' and entry['object_type'] == \
                'simple':
            obj_relevant = entry['relevant']
            object_pos = entry['position']
            if obj_relevant == 1:
                dist_to_obj = round(sqrt(pow(current_agent_pos_x - object_pos[0], 2) +
                                         pow(current_agent_pos_y - object_pos[1], 2) +
                                         pow(current_agent_pos_z - object_pos[2], 2)), 2)
        elif not entry['entityName'] == 'agent' and entry['object_type'] == \
                'receptable':
            recep_relevant = entry['relevant']
            recep_pos = entry['position']
            if recep_relevant == 1:
                dist_to_recep = round(sqrt(pow(current_agent_pos_x - recep_pos[0], 2) +
                                           pow(current_agent_pos_y - recep_pos[1], 2) +
                                           pow(current_agent_pos_z - recep_pos[2], 2)), 2)

    if obj_relevant == 1 and recep_relevant == 1:
        dist_obj_to_recep = round(sqrt(pow(recep_pos[0] - object_pos[0], 2) +
                                       pow(recep_pos[1] - object_pos[1], 2) +
                                       pow(recep_pos[2] - object_pos[2], 2)), 2)

    dot_product_score = get_dot_product_score([current_agent_pos_x, current_agent_pos_y, current_agent_pos_z],
                                              object_pos, agent_orientation)
    return [dist_to_obj, dist_to_recep, dist_obj_to_recep, dot_product_score]


def read_test_data(file_path):
    file = open(file_path, READ)
    count = 0.0
    correct = 0.0

    extra_count = 0.0
    extra_correct = 0.0

    predicted_tags = []
    actual_tags = []

    extra_predicted_tags = []
    extra_actual_tags = []

    for line in file:
        count += 1

        if count % 1000 == 0:
            print(correct, '   ', count)

        json_object = json.loads(line)
        action_sequence = json_object['action_sequence']
        desc = json_object['desc']
        desc = [item.strip() for item in desc]
        action_sequence_string = ' '.join(action_sequence)
        desc_string = ' '.join(desc)
        desc_string = desc_string.replace('\"', '')
        desc_string = desc_string.replace('.', '')
        desc_string = desc_string.replace(';', '')
        desc_string = desc_string.replace(',', '')
        desc_string = desc_string.lower()

        # if the record_type is not task_desc add the record without visual information
        if json_object['record_type'] != 'task_desc' and WITH_VISUAL:
            text_string = copy.deepcopy(desc_string)
            extra_correct = post_to_rasa(action_sequence_string, extra_actual_tags, extra_correct, text_string, extra_predicted_tags)
            extra_count += 1

        if WITH_VISUAL:
            visual_data = get_visual_information(json_object['scene_description'])

            # ADD code for Rotation Info
            # if the first action is PickupObject, consider the orientation info as well
            if action_sequence_string.strip().startswith('PickupObject') or \
                    action_sequence_string.strip().startswith('PutObject'):
                if random.random() < 0.5:
                    visual_data[3] = round(random.uniform(-1, 0), 2)
                else:
                    visual_data[3] = round(random.uniform(0, 1), 2)
                if visual_data[3] < 0:
                    action_sequence_string = 'RotateAgent ' + action_sequence_string.strip()
                    action_sequence_string = action_sequence_string.strip()

            desc_string = desc_string + ' @@@@@@'
            for visual_info in visual_data:
                desc_string = desc_string + ' ' + str(visual_info)

        correct = post_to_rasa(action_sequence_string, actual_tags, correct, desc_string, predicted_tags)

    print('Accuracy :: ', correct / count)
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
                                                  'PickupObject PutObject',
                                                  'RotateAgent PickupObject',
                                                  'RotateAgent PutObject',
                                                  'RotateAgent PickupObject PutObject',
                                                  'RotateAgent PickupObject GotoLocation PutObject',
                                                  'RotateAgent PickupObject GotoLocation'
                                                  ]))
    print('Accuracy :: ', (correct + extra_correct) / (extra_count + count))
    print('Correct  :: ', (correct + correct + extra_correct))
    print('Total    :: ', (count + extra_count))

    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(actual_tags, predicted_tags)
    print('Confusion Matrix\n')
    print(confusion)

    print(precision_recall_fscore_support(actual_tags + extra_actual_tags, predicted_tags + extra_predicted_tags,
                                          average='macro'))
    print(precision_recall_fscore_support(actual_tags + extra_actual_tags, predicted_tags + extra_predicted_tags,
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
                                                  'PickupObject PutObject',
                                                  'RotateAgent PickupObject',
                                                  'RotateAgent PutObject',
                                                  'RotateAgent PickupObject PutObject',
                                                  'RotateAgent PickupObject GotoLocation PutObject',
                                                  'RotateAgent PickupObject GotoLocation'
                                                  ]))
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(actual_tags + extra_actual_tags, predicted_tags + extra_predicted_tags)
    print('Confusion Matrix\n')
    print(confusion)


def post_to_rasa(action_sequence_string, actual_tags, correct, desc_string, predicted_tags):

    data = '{"text": "' + desc_string + '"}'
    response = requests.post(RASA_SERVER, headers=headers, data=data)
    response_json = json.loads(response.text)

    intent = response_json['intent']['name']
    confidence = response_json['intent']['confidence']
    predicted_tags.append(intent.strip())
    actual_tags.append(action_sequence_string.strip())
    if intent.strip() == action_sequence_string.strip():
        correct += 1
    # elif tag == 'NOT_SUPPORTED':
    #    correct += 1
    else:
        print('NOT MATCH : ', data)
        print(response_json)
        print('Predicted Intent :', intent, ' Confidence :', confidence)
        print('Actual Intent :', action_sequence_string)
        print('\n')
    return correct


read_test_data(TEST_FILE_PATH)
