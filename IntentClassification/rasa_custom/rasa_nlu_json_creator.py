# This program creates the nlu.json file which RASA uses for training
# The file augments the data already present in nlu.md
# The generated nlu.json file must be copied to

import json

TRAINING_INPUT_FILE = '/home/karun/PycharmProjects/IntentClassification/data' \
                      '-train/outfile1'
RASA_OUTFILE = '/home/karun/PycharmProjects/IntentClassification/data/nlu1.json'
READ = 'r'


def create_rasa_training_set(file_path, out_file_path):
    training_file = open(file_path, READ)
    intents = []
    for line in training_file:
        cols = line.split('\t')
        command = cols[0]
        tag = cols[1].strip()
        intent_json = {'text': command, 'intent': tag,
                       'text_dense_features': tag}
        intents.append(intent_json)

    common_examples_json = {'common_examples': intents}
    json_data = {'rasa_nlu_data': common_examples_json}

    with open(out_file_path, 'w') as outfile:
        json.dump(json_data, outfile)


create_rasa_training_set(TRAINING_INPUT_FILE, RASA_OUTFILE)
