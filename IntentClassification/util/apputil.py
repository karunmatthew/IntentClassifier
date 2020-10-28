import os
import spacy
import torch
import numpy as np
import math

JSON = '.json'
READ = "r"

nlp = spacy.load("en_core_web_lg")


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
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        print(chunk.text, ' : ', chunk.root.text, ' : ', chunk.root.dep_, ' : ',
              chunk.root.head.text, ' : ', chunk.root.head.pos_)
    return ''


print(get_dot_product([6, 8], [3, 4]))
print(get_dot_product_score([1, 3, -3], [-3, 444, -5], 180))
