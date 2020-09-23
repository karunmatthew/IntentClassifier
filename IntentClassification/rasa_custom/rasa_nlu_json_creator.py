# This program creates the nlu.json file which RASA uses for training
# The file augments the data already present in nlu.md
# The generated nlu.json file must be copied to

import json
from math import sqrt
import copy
import random

TRAINING_INPUT_FILE = '../data-train/training_set.txt'
RASA_OUTFILE = '../data/nlu.json'
READ = 'r'

TRAIN_SAMPLE_RATE = 100


def create_rasa_training_set(file_path, out_file_path):
    training_file = open(file_path, READ)
    intents = []
    count = 0
    for line in training_file:

        number = random.randint(0, 100)

        if number >= TRAIN_SAMPLE_RATE:
            continue

        json_object = json.loads(line)
        action_sequence = json_object['action_sequence']
        desc = json_object['desc']
        action_sequence_string = ' '.join(action_sequence)
        desc_string = ' '.join(desc)
        desc_string = desc_string.replace('\"', '')
        desc_string = desc_string.replace(',', '')

        # if the record_type is not task_desc add the record without visual information
        # as we do not need visual information to map to the intent
        if json_object['record_type'] != 'task_desc':
            text_string = copy.deepcopy(desc_string)
            record_json = {'text': text_string, 'intent': action_sequence_string}
            intents.append(record_json)
            count += 1

        visual_data = get_visual_information(json_object['scene_description'])
        desc_string = desc_string + ' @@@@@@'
        for visual_info in visual_data:
            desc_string = desc_string + ' ' + str(visual_info)
        intent_json = {'text': desc_string, 'intent': action_sequence_string}
        intents.append(intent_json)

        count += 1
        print(count)
        print(intent_json, '\n')

    print(len(intents))
    print(intents[12])
    common_examples_json = {'common_examples': intents}
    json_data = {'rasa_nlu_data': common_examples_json}

    with open(out_file_path, 'w') as outfile:
        json.dump(json_data, outfile)


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
            current_agent_pos_x = round(entry['position'][0], 2)
            current_agent_pos_y = round(entry['position'][1], 2)
            current_agent_pos_z = round(entry['position'][2], 2)

    agent_pos = [current_agent_pos_x, current_agent_pos_y, current_agent_pos_z]

    object_pos = [0, 0, 0]
    recep_pos = [0, 0, 0]

    for entry in scene_desc:
        if not entry['entityName'] == 'agent' and entry['object_type'] == \
                'simple':
            obj_relevant = entry['relevant']
            object_pos = entry['position']
            dist_to_obj = round(sqrt(pow(current_agent_pos_x - object_pos[0], 2) +
                                     pow(current_agent_pos_y - object_pos[1], 2) +
                                     pow(current_agent_pos_z - object_pos[2], 2)), 2)
        elif not entry['entityName'] == 'agent' and entry['object_type'] == \
                'receptable':
            recep_relevant = entry['relevant']
            recep_pos = entry['position']
            dist_to_recep = round(sqrt(pow(current_agent_pos_x - recep_pos[0], 2) +
                                       pow(current_agent_pos_y - recep_pos[1], 2) +
                                       pow(current_agent_pos_z - recep_pos[2], 2)), 2)

    dist_obj_to_recep = round(sqrt(pow(recep_pos[0] - object_pos[0], 2) +
                                   pow(recep_pos[1] - object_pos[1], 2) +
                                   pow(recep_pos[2] - object_pos[2], 2)), 2)

    return [dist_to_obj, dist_to_recep, dist_obj_to_recep]


create_rasa_training_set(TRAINING_INPUT_FILE, RASA_OUTFILE)
