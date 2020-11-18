# Author     :  Karun Mathew
# Student Id :  1007247
#
# --------------------------------
# BASELINE TRAINING DATA GENERATOR
# --------------------------------
#
# This python class generates training data in the format that
# is expected by the baseline classifier, a MultiLayer Perceptron
#
# The format is shown below,
# ---------------------------------------------------------------------------------------------------------
# desc	                            dist_to_obj dist_to_recep	dist_obj_to_recep	agent_facing	intent
# ---------------------------------------------------------------------------------------------------------
# put a statue on a coffee table	0.35	    1.84        	2.03	            0.68        	7
# ---------------------------------------------------------------------------------------------------------
# put a statue on a coffee table	4.47	    4.47        	0.14            	-0.4	        9
# ---------------------------------------------------------------------------------------------------------
# walk to the fridge	            -1	        3.2	            -1	                1   	        0
# ---------------------------------------------------------------------------------------------------------
# turn right and go to the coffee
# maker then turn right and face
# the kitchen counter pick up the
# egg from in front of the yellow
# vase turn around and bring the
# egg to the black refrigerator
# open the fridge put the egg on
# the same shelf as the pot and
# then close the fridge	            1.73	    3.52	        3.2             	-0.56	        5
# ---------------------------------------------------------------------------------------------------------

from util.noise_generator import add_noise
from util.alfred_json_parser import get_visual_information
from util.apputil import get_data, remove_special_characters, \
    TRAIN_DATA_PATH, DEV_DATA_PATH, TEST_DATA_PATH, \
    TRAIN_MLP_FULL_FILE, DEV_MLP_FULL_FILE, TEST_MLP_FULL_FILE, WRITE, LABELS_MAP
import json
import random

# Defines the percentage of data that is to be chosen from each of the datasets
TRAIN_SAMPLE_RATE = 100
TEST_SAMPLE_RATE = 100
VALIDATION_SET_SAMPLE_RATE = 100


# this method creates a dataset by reading the ALFRED data file at the specified
# 'input_file_path' and creates a new dataset saved as 'out_file_name'. The sample_rate
# decides the percentage of data of the original data is copied into the output
def create_mlp_specific_dataset(input_file_path, out_file_name, sample_rate):
    global intent_count
    out_file = open(out_file_name, WRITE)
    raw_train_data = get_data(input_file_path)
    # prints out the header
    out_file.write(
        'desc' + '\t' + 'dist_to_obj' + '\t' +
        'dist_to_recep' + '\t' + 'dist_obj_to_recep' +
        '\t' + 'agent_facing' + '\t' + 'intent' + '\n')

    for datum in raw_train_data:
        json_data = json.loads(datum)

        # sample the data at the provided rate
        if random.randint(0, 100) > sample_rate:
            continue

        desc = ' '.join(json_data['desc']).strip()
        desc = remove_special_characters(desc)

        visual_data = get_visual_information(json_data['scene_description'])
        action_sequence = ' '.join(json_data['action_sequence']).strip()
        # add noise to the data records for better generalization
        action_sequence, visual_data = add_noise(action_sequence, visual_data)

        intent = LABELS_MAP[action_sequence]
        intent_count[intent] = intent_count.get(intent, 0) + 1

        # write directive command and visual data to file
        out_file.write(desc + '\t')
        for v_data in visual_data:
            out_file.write(str(v_data) + '\t')
        out_file.write(str(intent) + '\n')

    out_file.close()


# Create train, validation and test data for the baseline model
intent_count = {}
create_mlp_specific_dataset(TRAIN_DATA_PATH, TRAIN_MLP_FULL_FILE, TRAIN_SAMPLE_RATE)
print(intent_count)

intent_count = {}
create_mlp_specific_dataset(TEST_DATA_PATH, TEST_MLP_FULL_FILE, TEST_SAMPLE_RATE)
print(intent_count)

intent_count = {}
create_mlp_specific_dataset(DEV_DATA_PATH, DEV_MLP_FULL_FILE, VALIDATION_SET_SAMPLE_RATE)
print(intent_count)
