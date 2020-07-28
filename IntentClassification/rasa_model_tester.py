# This program sends the intents to the rasa server that runs
# by default on the 5005 port. The response will contain the
# probability distribution of all the intents

# The rasa server is started by providing the model path
# rasa run --enable-api -m models/20200722-133414.tar.gz

import requests
import json

RASA_SERVER = 'http://localhost:5005/model/parse'
TEST_FILE_PATH = '/home/karun/PycharmProjects/IntentClassification/outfile_pickup_simple_train'
READ = 'r'

headers = {
    'Content-type': 'application/json'
}


def read_test_data(file_path):
    file = open(file_path, READ)
    count = 0.0
    correct = 0.0
    for line in file:
        count += 1
        cols = line.split('\t')
        command = cols[0]
        tag = cols[1].strip()

        # need to remove quotes from data input
        data = '{"text": "' + command + '"}'
        response = requests.post(RASA_SERVER, headers=headers, data=data)
        response_json = json.loads(response.text)
        print(command)
        intent = response_json['intent']['name']
        confidence = response_json['intent']['confidence']
        print(intent, ' ', confidence)
        print(tag, '\n')

        if intent.strip() == tag:
            correct += 1
        #elif tag == 'NOT_SUPPORTED':
        #    correct += 1
        else:
            print('NOT MATCH')
            print(intent, ":")
            print(tag, ":")

    print('Accuracy :: ', correct/count)
    print('Correct  :: ', correct)
    print('Total    :: ', count)


def create_RASA_training_set(file_path, out_file_path):
    training_file = open(file_path, READ)
    intents = []
    for line in training_file:
        cols = line.split('\t')
        command = cols[0]
        tag = cols[1].strip()
        intent_json = {'text': command, 'intent': tag}
        intents.append(intent_json)

    common_examples_json = {'common_examples': intents}
    json_data = {'rasa_nlu_data': common_examples_json}

    with open(out_file_path, 'w') as outfile:
        json.dump(json_data, outfile)


TRAINING_INPUT_FILE = '/home/karun/PycharmProjects/IntentClassification/outfile1'
RASA_OUTFILE = '/home/karun/PycharmProjects/IntentClassification/data/nlu.json'

read_test_data(TEST_FILE_PATH)
# create_RASA_training_set(TRAINING_INPUT_FILE, RASA_OUTFILE)
