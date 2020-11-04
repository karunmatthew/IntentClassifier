# Author     :  Karun Mathew
# Student Id :  1007247
#
# This program creates the nlu.json file which RASA uses for training the model
# The generated nlu.json file must be present in the 'data' folder of the project
# which is the default location that RASA looks for to find the training samples

import json
import copy
import random

from util.noise_generator import add_noise, get_pick_up_noise, get_put_down_noise
from util.alfred_json_parser import get_visual_information
from util.apputil import remove_special_characters, \
    TRAIN_DATA_PATH, RASA_OUTFILE, READ, TRAIN_SAMPLE_RATE, WITH_VISUAL, LANG_VISUAL_DELIMITER


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
    intent_dist = {}
    count = 0
    for line in training_file:

        if random.randint(0, 100) >= TRAIN_SAMPLE_RATE:
            continue

        json_object = json.loads(line)

        # Get the command sentence
        desc, desc_string = get_command_description_string(json_object)

        # convert the array of intents into a space separated format
        action_sequence = json_object['action_sequence']
        action_sequence_string = ' '.join(action_sequence)

        # if we do not need visual information to map command sentence
        # to the intent, send it without the visual information
        if not_need_visual(json_object, action_sequence_string):
            text_string = copy.deepcopy(desc_string)
            record_json = {'text': text_string, 'intent': action_sequence_string}
            intents.append(record_json)
            count += 1

        # for more generalization, combine sentences with conjunctions like 'and'
        # with 50% probability. This set of intents do not need visual info
        if json_object['record_type'] == 'multi_desc' and random.random() < 0.5:
            conjuncted_string = ' and '.join(desc)
            conjuncted_string = remove_special_characters(conjuncted_string)
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

            go_and_pick_intent = {'text': desc_string.strip() + ' ' +
                                          LANG_VISUAL_DELIMITER + ' ' + get_pick_up_noise(),
                                  'intent': 'GotoLocation PickupObject'}
            intents.append(go_and_pick_intent)
            count += 1

        # if the action is put, then add a sample for go and put as well
        if action_sequence_string.strip() == 'PutObject' and WITH_VISUAL:
            go_and_put_intent = {'text': desc_string.strip() + ' ' + LANG_VISUAL_DELIMITER + ' ' + get_put_down_noise(),
                                 'intent': 'GotoLocation PutObject'}
            intents.append(go_and_put_intent)
            count += 1

        visual_string = ' ' + LANG_VISUAL_DELIMITER + ' '
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
            conjuncted_string = remove_special_characters(conjuncted_string)

            conjuncted_string = conjuncted_string + visual_string
            conjuncted_string = conjuncted_string.strip()
            conj_with_visual = {'text': conjuncted_string, 'intent': action_sequence_string}
            intents.append(conj_with_visual)
            count += 1

    for record in intents:
        intent_dist[record['intent']] = intent_dist.get(record['intent'], 0) + 1
    print('intent distribution', intent_dist)

    # convert the data in the form that rasa expects
    common_examples_json = {'common_examples': intents}
    json_data = {'rasa_nlu_data': common_examples_json}

    # create the nlu.json for rasa training
    with open(out_file_path, 'w') as outfile:
        json.dump(json_data, outfile)


# remove all special characters from the command sentence and join the sentences with space
def get_command_description_string(json_object):
    desc = json_object['desc']
    desc = [item.strip() for item in desc]
    desc_string = ' '.join(desc)
    desc_string = remove_special_characters(desc_string)
    return desc, desc_string


create_rasa_training_set(TRAIN_DATA_PATH, RASA_OUTFILE)
