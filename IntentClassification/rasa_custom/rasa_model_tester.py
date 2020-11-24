# This program sends the intents to the rasa server that runs
# by default on the 5005 port. The response will contain the
# probability distribution of all the intents

# The rasa server is started by providing the model path
# rasa run --enable-api -m models/20200722-133414.tar.gz

import random
import math
# install tkinter package so as to open the file dialog
import tkinter as tk
from tkinter import filedialog
import platform
import os
import time
import requests
import json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from util.alfred_json_parser import get_visual_information
from util.apputil import READ, RASA_SERVER, DEV_DATA_PATH,\
    MAX_ANGLE, WITH_VISUAL, LABELS, LANG_VISUAL_DELIMITER,\
    RASA_SERVER_STARTUP_TIME


headers = {
    'Content-type': 'application/json'
}

# Kill any process running on the RASA server default port 5005
if platform.system() == 'Linux':
    os.system('fuser -k 5005/tcp')
else:
    os.system('kill pid $(lsof -ti:5005)')

# accepts the test / validation dataset file path
# predicts the intent(s) for each data instance
# prints out the model performance statistics
def test_model(file_path):
    file = open(file_path, READ)

    count = 0.0
    correct = 0.0
    predicted_tags = []
    actual_tags = []

    for line in file:
        count += 1

        json_object = json.loads(line)
        action_sequence = json_object['action_sequence']
        desc = json_object['desc']
        desc = [item.strip() for item in desc]
        action_sequence_string = ' '.join(action_sequence)
        desc_string = ' '.join(desc)
        desc_string = remove_special_characters(desc_string)

        # test with both visual and language data
        if WITH_VISUAL:
            visual_data = get_visual_information(json_object['scene_description'])

            # if the first action is PickupObject, consider the
            # orientation info as well when deciding the intent
            if action_sequence_string.strip().startswith('PickupObject') or \
                    action_sequence_string.strip().startswith('PutObject'):

                if random.random() < 0.5:
                    # Needs to rotate to pick the object
                    visual_data[3] = round(random.uniform(-1, math.cos(math.radians(MAX_ANGLE))), 2)
                    action_sequence_string = 'RotateAgent ' + action_sequence_string.strip()
                else:
                    visual_data[3] = round(random.uniform(math.cos(math.radians(MAX_ANGLE)), 1), 2)

            desc_string = desc_string + ' ' + LANG_VISUAL_DELIMITER
            for visual_info in visual_data:
                desc_string = desc_string + ' ' + str(visual_info)

        correct = post_to_rasa(action_sequence_string, actual_tags, correct, desc_string, predicted_tags)

    # print out the performance statistics of the model
    print_statistics(actual_tags, correct, count, predicted_tags)


def print_statistics(actual_tags, correct, count, predicted_tags):
    print('Accuracy :: ', correct / count)
    print('Correct  :: ', correct)
    print('Total    :: ', count)
    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average='macro'))
    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average='micro'))
    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average=None,
                                          labels=LABELS))
    confusion = confusion_matrix(actual_tags, predicted_tags)
    print('Confusion Matrix\n')
    print(confusion)


# removes special characters in the text data, as we will not encounter
# such data in the real world when we convert voice commands to text
def remove_special_characters(desc_string):
    desc_string = desc_string.replace('\"', '')
    desc_string = desc_string.replace('.', '')
    desc_string = desc_string.replace(';', '')
    desc_string = desc_string.replace(',', '')
    desc_string = desc_string.lower()
    return desc_string


# post a single data instance to the RASA server
# the data includes both language and visual data
# delimited by the LANG_VISUAL_DELIMITER specified in apputil
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
    else:
        print('Intent Prediction Error')
        print('Input            :: ', data)
        print('Output Response  :: ', response_json)
        print('Predicted Intent :: ', intent, ' Confidence :', confidence)
        print('Actual Intent    :: ', action_sequence_string)
        print('\n')
    return correct


# ------------------------------------ START OF MAIN -------------------------------------------#
try:
    print('RASA MODEL TESTER')
    print('PRE-REQUISITES: The rasa model has been trained and the model file has been generated'
          '\n This program starts the server at port 5005 and the test samples are posted to it'
          '\n Please select the model file and the test dataset file using the file dialog')

    root = tk.Tk()
    root.withdraw()
    model_file_path = filedialog.askopenfilename(initialdir="../", title='Select the model tar.gz file')
    root.update()
    print('\nModel file selected at :: ', model_file_path)
    # start the server process
    if len(model_file_path) > 0:
        print('\nStarting the RASA server process.....\n')
        os.system('rasa run --enable-api -m "' + model_file_path + '" &')
        time.sleep(RASA_SERVER_STARTUP_TIME)

        print('Select the test file.')
        test_file_path = filedialog.askopenfilename(initialdir="../", title='Select the test file')
        root.update()
        root.destroy()
        if len(test_file_path) > 0:
            print('Selected test file :: ', test_file_path)
            test_model(test_file_path)

finally:
    # kill the server process running in 5005 port
    print('\nKilling server process')
    if platform.system() == 'Linux':
        os.system('fuser -k 5005/tcp')
    else:
        os.system('kill pid $(lsof -ti:5005)')
# ------------------------------------------------------------------------------------------ #
