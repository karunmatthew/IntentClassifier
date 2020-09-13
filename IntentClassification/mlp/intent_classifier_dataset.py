from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertModel

TRAIN_BERT_FILE = '../data-train/train_bert.txt'
TEST_BERT_FILE = '../data-test/test_bert.txt'


class MMICDataset(Dataset):

    def __init__(self, filename, maxlen):

        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')
        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        sentence = self.df.loc[index, 'desc']
        label = self.df.loc[index, 'intent']

        tokens = self.tokenizer.tokenize(sentence)

        if len(tokens) < self.maxlen:
            # Padding sentences
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(
                tokens))]
        else:
            # Prunning the list to be of specified max length
            tokens = tokens[:self.maxlen-1] + ['[SEP]']
        # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # Converting the list to a pytorch tensor
        tokens_ids_tensor = torch.tensor(tokens_ids)
        # Obtaining the attention mask i.e a tensor containing 1s for no
        # padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


train_set = MMICDataset(filename=TRAIN_BERT_FILE, maxlen=30)
dev_set = MMICDataset(filename=TEST_BERT_FILE, maxlen=30)

# Creating instances of training and development dataloaders
train_loader = DataLoader(train_set, batch_size=64, num_workers=5)
dev_loader = DataLoader(dev_set, batch_size=64, num_workers=5)