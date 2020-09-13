import torch
from util.apputil import get_data
import spacy
import json

TRAIN_DATA_PATH = '../data-train/training_set.txt'
N, D_in, H, D_out = 64, 300, 100, 300

# load the entire training data
raw_train_data = get_data(TRAIN_DATA_PATH)
# print(len(train_data))

nlp = spacy.load("en_core_web_lg")
doc = nlp("hello there")
print(len(doc.vector))
train_data = []

# takes in the raw train data from the training set file and
# extracts the desc(x) and action sequence(y)
for datum in raw_train_data:
    json_data = json.loads(datum)
    desc = ' '.join(json_data['desc'])
    intent = ' '.join(json_data['action_sequence'])
    train_data.append((desc.strip(), intent.strip()))


for doc, intent in nlp.pipe(train_data, as_tuples=True):
    print(doc.vector)
