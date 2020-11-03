from util.noise_generator import add_noise
from util.alfred_json_parser import get_visual_information
from util.apputil import get_data, remove_special_characters, \
    TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH, \
    TRAIN_MLP_FULL_FILE, DEV_MLP_FULL_FILE, TEST_MLP_FULL_FILE, WRITE, LABELS_MAP
import json
import random


def create_mlp_specific_dataset(input_file_name, out_file_name, sample_rate):
    global intent_count
    out_file = open(out_file_name, WRITE)
    raw_train_data = get_data(input_file_name)
    # print out the header
    out_file.write(
        'desc' + '\t' + 'dist_to_obj' + '\t' +
        'dist_to_recep' + '\t' + 'dist_obj_to_recep' +
        '\t' + 'agent_facing' + '\t' + 'intent' + '\n')

    for datum in raw_train_data:
        json_data = json.loads(datum)

        # sample the data at the provided rate
        number = random.randint(0, 100)
        if number > sample_rate:
            continue

        desc = ' '.join(json_data['desc']).strip()
        desc = remove_special_characters(desc)

        visual_data = get_visual_information(json_data['scene_description'])
        action_sequence = ' '.join(json_data['action_sequence']).strip()
        action_sequence, visual_data = add_noise(action_sequence, visual_data)

        intent = LABELS_MAP[action_sequence]
        intent_count[intent] = intent_count.get(intent, 0) + 1

        # write directive command and visual data to file
        out_file.write(desc + '\t')
        for v_data in visual_data:
            out_file.write(str(v_data) + '\t')
        out_file.write(str(intent) + '\n')

    out_file.close()


intent_count = {}
create_mlp_specific_dataset(TRAIN_DATA_PATH, TRAIN_MLP_FULL_FILE, 100)
print(intent_count)

intent_count = {}
create_mlp_specific_dataset(TEST_DATA_PATH, TEST_MLP_FULL_FILE, 100)
print(intent_count)

intent_count = {}
create_mlp_specific_dataset(DEV_DATA_PATH, DEV_MLP_FULL_FILE, 100)
print(intent_count)
