# This program creates the nlu.json file which RASA uses for training
# The file augments the data already present in nlu.md
# The generated nlu.json file must be copied to

import json
from math import sqrt
import copy
import random
from util.apputil import get_dot_product_score, get_object_from_sentence

TRAINING_INPUT_FILE = '../data-train/training_set.txt'
RASA_OUTFILE = '../data/nlu.json'
READ = 'r'

TRAIN_SAMPLE_RATE = 20
WITH_VISUAL = True
CONSIDER_ROTATION = True


def not_need_visual(json_obj, action_sequence_string):
    if json_obj['record_type'] != 'task_desc':
        return True
    elif action_sequence_string.strip() == 'PickupObject PutObject':
        return True
    else:
        return False


def create_rasa_training_set(file_path, out_file_path):
    training_file = open(file_path, READ)
    intents = []
    count = 0
    for line in training_file:

        number = random.randint(0, 100)
        if number >= TRAIN_SAMPLE_RATE:
            continue

        json_object = json.loads(line)

        # Get the command sentence
        desc, desc_string = get_command_description_string(json_object)

        # convert the array of intents into a space separated format
        action_sequence = json_object['action_sequence']
        action_sequence_string = ' '.join(action_sequence)

        # if we do not need visual information to map command sentence
        # to the intent, send it without visual info
        if not_need_visual(json_object, action_sequence_string):
            text_string = copy.deepcopy(desc_string)
            record_json = {'text': text_string, 'intent': action_sequence_string}
            intents.append(record_json)
            count += 1

        # for more generalization, combine sentences with conjunctions like 'and'
        # with 50% probability. This set of intents do not need visual info
        if json_object['record_type'] == 'multi_desc' and random.random() < 0.5:
            conjuncted_string = ' and '.join(desc)
            conjuncted_string = conjuncted_string.replace('\"', '')
            conjuncted_string = conjuncted_string.replace('.', '')
            conjuncted_string = conjuncted_string.replace(';', '')
            conjuncted_string = conjuncted_string.replace(',', '')
            conjuncted_string = conjuncted_string.lower()
            record_with_conjunction_json = {'text': conjuncted_string.strip(),
                                            'intent': action_sequence_string}
            intents.append(record_with_conjunction_json)
            count += 1

        # get the visual information
        visual_data = get_visual_information(json_object['scene_description'])

        action_sequence_string, visual_data = add_noise(action_sequence_string, visual_data)

        # if the action is pick up, then add a sample for go and pick up as well
        if action_sequence_string.strip() == 'PickupObject' and WITH_VISUAL:
            # TODO ensure that the orientation of both agent and object is the same

            go_and_pick_intent = {'text': desc_string.strip() + ' @@@@@@ ' + get_pick_up_noise(),
                                  'intent': 'GotoLocation PickupObject'}
            intents.append(go_and_pick_intent)
            count += 1

        # if the action is put, then add a sample for go and put as well
        if action_sequence_string.strip() == 'PutObject' and WITH_VISUAL:
            go_and_put_intent = {'text': desc_string.strip() + ' @@@@@@ ' + get_put_down_noise(),
                                 'intent': 'GotoLocation PutObject'}
            intents.append(go_and_put_intent)
            count += 1

        visual_string = ' @@@@@@ '
        for visual_info in visual_data:
            visual_string = visual_string + str(visual_info) + ' '

        if not WITH_VISUAL:
            visual_string = ''

        desc_string = desc_string + visual_string
        desc_string = desc_string.strip()
        intent_json = {'text': desc_string, 'intent': action_sequence_string}
        intents.append(intent_json)
        count += 1

        # combining multi-desc commands with "AND"
        if json_object['record_type'] == 'multi_desc' and random.random() < 0.5:
            conjuncted_string = copy.deepcopy(' and '.join(desc))
            conjuncted_string = conjuncted_string.replace('\"', '')
            conjuncted_string = conjuncted_string.replace('.', '')
            conjuncted_string = conjuncted_string.replace(';', '')
            conjuncted_string = conjuncted_string.replace(',', '')
            conjuncted_string = conjuncted_string.lower()

            conjuncted_string = conjuncted_string + visual_string
            conjuncted_string = conjuncted_string.strip()
            conj_with_visual = {'text': conjuncted_string, 'intent': action_sequence_string}
            intents.append(conj_with_visual)
            count += 1

        # print(count)
        # print(intent_json, '\n')

    # for i in intents:
    #    print(i)

    # print(len(intents))
    common_examples_json = {'common_examples': intents}
    json_data = {'rasa_nlu_data': common_examples_json}

    with open(out_file_path, 'w') as outfile:
        json.dump(json_data, outfile)


# remove all special characters from the command sentence and join the sentences with space
def get_command_description_string(json_object):
    desc = json_object['desc']
    desc = [item.strip() for item in desc]
    desc_string = ' '.join(desc)
    desc_string = desc_string.replace('\"', '')
    desc_string = desc_string.replace('.', '')
    desc_string = desc_string.replace(';', '')
    desc_string = desc_string.replace(',', '')
    desc_string = desc_string.lower()
    return desc, desc_string


def get_unknown_or_noise():
    if random.uniform(0, 100) > 50:
        return -1
    else:
        # TODO does the lower limit be higher?
        return round(random.uniform(0, 8), 2)


def get_pick_up_noise():
    return str(round(random.uniform(0.6, 8), 2)) + ' ' + str(get_unknown_or_noise()) + ' ' \
           + str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(0, 1), 2))


def get_put_down_noise():
    return str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(0.6, 8), 2)) + ' ' \
           + str(get_unknown_or_noise()) + ' ' + str(round(random.uniform(0, 1), 2))


def add_noise(action_sequence_string, visual_data):
    if action_sequence_string.strip() == 'PickupObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        # noise
        visual_data[1] = get_unknown_or_noise()
        visual_data[2] = get_unknown_or_noise()
        visual_data[3] = round(random.uniform(0, 1), 2)

    if action_sequence_string.strip() == 'PutObject':
        visual_data[1] = round(random.uniform(0, 0.5), 2)
        # noise
        visual_data[0] = get_unknown_or_noise()
        visual_data[2] = get_unknown_or_noise()
        visual_data[3] = round(random.uniform(0, 1), 2)

    if action_sequence_string.strip() == 'GotoLocation PickupObject PutObject':
        visual_data[2] = round(random.uniform(0, 0.5), 2)

    if action_sequence_string.strip() == 'GotoLocation PutObject':
        visual_data[1] = round(visual_data[1] + round(random.uniform(-0.2, 0.2), 2), 2)
        # noise
        visual_data[2] = get_unknown_or_noise()

    if action_sequence_string.strip() == 'PickupObject PutObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[1] = round(random.uniform(0, 0.5), 2)
        visual_data[2] = round(random.uniform(0, 0.5), 2)

    if action_sequence_string.strip() == 'PickupObject GotoLocation PutObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[2] = round(visual_data[2] + round(random.uniform(-0.2, 0.2), 2), 2)

    if action_sequence_string.strip() == 'PickupObject GotoLocation':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[2] = round(visual_data[2] + round(random.uniform(-0.2, 0.2), 2), 2)

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

    return action_sequence_string, visual_data


def add_clean_noise(action_sequence_string, visual_data):
    if action_sequence_string.strip() == 'PickupObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        # noise
        visual_data[1] = round(random.uniform(-1, 8), 2)
        visual_data[2] = round(random.uniform(-1, 8), 2)
    if action_sequence_string.strip() == 'PutObject':
        if random.uniform(0, 100) < 50:
            visual_data[0] = -1
        else:
            visual_data[0] = 0
        visual_data[1] = round(random.uniform(0, 0.5), 2)
        if random.uniform(0, 100) < 80:
            visual_data[2] = -1
        # noise
        visual_data[0] = round(random.uniform(-1, 8), 2)
        visual_data[2] = round(random.uniform(-1, 8), 2)
    if action_sequence_string.strip() == 'GotoLocation PickupObject PutObject':
        visual_data[2] = round(random.uniform(0, 0.5), 2)
    if action_sequence_string.strip() == 'GotoLocation PutObject':
        visual_data[1] = round(visual_data[1] + round(random.uniform(-0.2, 0.2), 2), 2)
        if random.uniform(0, 100) < 50:
            visual_data[0] = -1
        else:
            visual_data[0] = 0
    if action_sequence_string.strip() == 'PickupObject PutObject':
        visual_data[2] = round(random.uniform(0, 0.5), 2)
    if action_sequence_string.strip() == 'PickupObject GotoLocation PutObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[2] = round(visual_data[2] + round(random.uniform(-0.2, 0.2), 2), 2)
    if action_sequence_string.strip() == 'PickupObject GotoLocation':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[2] = round(visual_data[2] + round(random.uniform(-0.2, 0.2), 2), 2)
    if action_sequence_string.strip() == 'PickupObject PutObject':
        visual_data[0] = round(random.uniform(0, 0.5), 2)
        visual_data[1] = round(random.uniform(0, 0.5), 2)

    return visual_data


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


create_rasa_training_set(TRAINING_INPUT_FILE, RASA_OUTFILE)
