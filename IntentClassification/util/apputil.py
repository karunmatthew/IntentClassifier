import os
import spacy
import torch
import numpy as np
import math
from math import sqrt
import random

JSON = '.json'
READ = "r"
WRITE = 'w'

RASA_OUTFILE = '../data/nlu.json'
TRAIN_SAMPLE_RATE = 10
WITH_VISUAL = True
CONSIDER_ROTATION = True
MAX_ANGLE = 60

RASA_SERVER = 'http://localhost:5005/model/parse'


TRAIN_DATA_PATH = '../data-train/training_set.txt'
DEV_DATA_PATH = '../data-train/dev_set.txt'
TEST_DATA_PATH = '../data-test/testing_set.txt'

TRAIN_MLP_FULL_FILE = '../data-train/train_mlp_full.txt'
DEV_MLP_FULL_FILE = '../data-train/dev_mlp_full.txt'
TEST_MLP_FULL_FILE = '../data-test/test_mlp_full.txt'


LANG_VISUAL_DELIMITER = '@@@@@@'
LABELS = ['GotoLocation',
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
          ]

LABELS_MAP = {
          'GotoLocation': 0,
          'PickupObject': 1,
          'PutObject': 2,
          'GotoLocation PickupObject': 3,
          'GotoLocation PickupObject GotoLocation': 4,
          'GotoLocation PickupObject GotoLocation PutObject': 5,
          'PickupObject GotoLocation': 6,
          'PickupObject GotoLocation PutObject': 7,
          'GotoLocation PutObject': 8,
          'GotoLocation PickupObject PutObject': 9,
          'PickupObject PutObject': 10,
          'RotateAgent PickupObject': 11,
          'RotateAgent PutObject': 12,
          'RotateAgent PickupObject PutObject': 13,
          'RotateAgent PickupObject GotoLocation PutObject': 14,
          'RotateAgent PickupObject GotoLocation': 15
          }


def get_json_file_paths(folder_path):
    file_paths = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            extension = os.path.splitext(name)[1]
            if extension == JSON:
                file_path = os.path.join(path, name)
                file_paths.append(file_path)

    return file_paths


# read a file and return a list of lines
def get_data(input_file):
    file = open(input_file, READ)
    data = []
    for line in file:
        data.append(line.strip())
    return data


def get_dot_product(x, y):
    return round(np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))), 2)


def get_agent_facing_direction_vector(agent_face_direction):
    z = round(math.cos(math.radians(agent_face_direction)), 2)
    x = round(math.sin(math.radians(agent_face_direction)), 2)
    return [x, z]


def get_dot_product_score(agent_pos, object_pos, agent_face_direction):
    f = get_agent_facing_direction_vector(agent_face_direction)
    o_a = np.subtract([object_pos[0], object_pos[2]], [agent_pos[0], agent_pos[2]])
    if o_a[0] == 0 and o_a[1] == 0:
        return 1
    else:
        return get_dot_product(f, o_a)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_object_from_sentence(text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        print(chunk.text, ' : ', chunk.root.text, ' : ', chunk.root.dep_, ' : ',
              chunk.root.head.text, ' : ', chunk.root.head.pos_)
    return ''


def remove_special_characters(string_text):
    string_text = string_text.replace('\"', '')
    string_text = string_text.replace('\n', '')
    string_text = string_text.replace('.', '')
    string_text = string_text.replace(',', '')
    string_text = string_text.replace(';', '')
    string_text = string_text.lower()
    return string_text


# returns the distance between two 3D co-ordinates
def get_L2_distance(X1, X2):
    return round(sqrt(pow(X1[0] - X2[0], 2) +
                      pow(X1[1] - X2[1], 2) +
                      pow(X1[2] - X2[2], 2)), 2)


# print(get_dot_product_score([1, 3, -3], [-3, 1, -5], 180))
# print(get_dot_product_score([1, 3, -3], [5, 1, -5], 180))
# print(get_agent_facing_direction_vector(180))