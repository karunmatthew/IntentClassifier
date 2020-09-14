# This program creates the nlu.json file which RASA uses for training
# The file augments the data already present in nlu.md
# The generated nlu.json file must be copied to

import json
from math import sqrt

TRAINING_INPUT_FILE = '../data-train/training_set.txt'
RASA_OUTFILE = '../data/nlu.json'
READ = 'r'

TRAIN_SAMPLE_RATE = 0.1


def create_rasa_training_set(file_path, out_file_path):
    training_file = open(file_path, READ)
    intents = []
    for line in training_file:
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
        intent_json = {'text': desc_string, 'intent': action_sequence_string}
        intents.append(intent_json)
        print(intent_json, '\n')

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

create_rasa_training_set(TRAINING_INPUT_FILE, RASA_OUTFILE)
