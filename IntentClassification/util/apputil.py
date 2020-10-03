import os
import spacy
import torch

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

