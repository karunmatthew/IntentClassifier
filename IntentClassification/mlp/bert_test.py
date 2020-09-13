import torch
from util.apputil import get_data
import spacy
import json

TRAIN_DATA_PATH = '../data-train/training_set.txt'
TEST_DATA_PATH = '../data-test/testing_set.txt'

N, D_in, H, D_out = 64, 300, 100, 300

# load the entire training data
raw_train_data = get_data(TRAIN_DATA_PATH)

nlp = spacy.load("en_core_web_lg")

train_data = []

bert_model = BertModel.from_pretrained('bert-large-cased')
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

# takes in the raw train data from the training set file and
# extracts the desc(x) and action sequence(y)

SENTENCE_LENGTH = 10

# BERT expects the sentence to start with [CLS] and each sentence to be
# separated by [SEP]
for datum in raw_train_data:
    json_data = json.loads(datum)
    desc_tokens = ['[CLS]'] + tokenizer.tokenize(
        '[SEP]'.join(json_data['desc']).strip()) + ['[SEP]']
    padded_tokens = desc_tokens + \
                    ['[PAD]' for _ in range(SENTENCE_LENGTH - len(desc_tokens))]
    intent_tokens = ['[CLS]'] + \
                    tokenizer.tokenize('[SEP]'.join(json_data[
                                                        'action_sequence']).strip()) + [
                        '[SEP]']
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    seg_ids = [0 for _ in range(len(padded_tokens))]
    token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)


    print(desc_tokens)
    print(intent_tokens)
    print('\n')
    # train_data.append((desc, intent))