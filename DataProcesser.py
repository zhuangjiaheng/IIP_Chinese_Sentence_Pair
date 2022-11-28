import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import random
from utils import load_json_file
from transformers import BertModel, BertTokenizer, BertConfig
from Config import *


class Covid19Dataset(Dataset):
    def __init__(self, query1: tuple, query2: tuple,
                 categories, targets, tokenizer, seq_len):
        super(Covid19Dataset, self).__init__()
        self.texts = query1
        self.text_pairs = query2
        self.categories = categories
        self.targets = targets
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        assert len(self.texts) == len(self.text_pairs) and len(self.texts) == len(self.targets)

    def __getitem__(self, idx):
        category = self.categories[idx]
        text = self.texts[idx]
        text_pair = self.text_pairs[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text, text_pair,
            max_length=self.seq_len,
            truncation=True,
            add_special_tokens=True,      # 添加 '[CLS]' and '[SEP]'
            return_token_type_ids=True,   # 返回
            pad_to_max_length=True,       # 补充到最大长度
            return_attention_mask=True,   # 返回相应的指示序列
            return_tensors="pt",          # 以Pytorch Tensor形式返回
        )

        return {
            'text': text,
            'text_pair': text_pair,
            'category': category,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


train = pd.read_csv("data/datasets/train.csv")
train_query1 = train["query1"].values
train_query2 = train["query2"].values
train_categories = train["category"].values
train_targets = train["label"].values

dev = pd.read_csv("data/datasets/dev.csv")
dev_query1 = dev["query1"].values
dev_query2 = dev["query2"].values
dev_categories = dev["category"].values
dev_targets = dev["label"].values

model_path = './data/pretrained_model/chinese_roberta_wwm_large_ext_pytorch/'
bert_config = BertConfig.from_pretrained(model_path + 'bert_config.json', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt', config=bert_config)

train_td = Covid19Dataset(train_query1, train_query2, train_categories, train_targets, tokenizer, SEQ_LEN)
dev_td = Covid19Dataset(dev_query1, dev_query2, dev_categories, dev_targets, tokenizer, SEQ_LEN)
train_loader = DataLoader(train_td, batch_size=batch_size, num_workers=4)
dev_loader = DataLoader(dev_td, batch_size=batch_size, num_workers=4)


# if __name__ == '__main__':
#     train = pd.read_csv("data/datasets/train.csv")
#     train_query1 = train["query1"].values
#     train_query2 = train["query2"].values
#     train_categories = train["category"].values
#     train_targets = train["label"].values
#
#     dev = pd.read_csv("data/datasets/dev.csv")
#     dev_query1 = train["query1"].values
#     dev_query2 = train["query2"].values
#     dev_categories = train["category"].values
#     dev_targets = train["label"].values
#
#     model_path = './data/pretrained_model/chinese_roberta_wwm_large_ext_pytorch/'
#     bert_config = BertConfig.from_pretrained(model_path + 'bert_config.json', output_hidden_states=True)
#     tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt', config=bert_config)
#
#     train_td = Covid19Dataset(train_query1, train_query2, train_categories, train_targets, tokenizer, SEQ_LEN)
#     dev_td = Covid19Dataset(dev_query1, dev_query2, dev_categories, dev_targets, tokenizer, SEQ_LEN)
#     train_loader = DataLoader(train_td, batch_size=batch_size, num_workers=4)
#     dev_loader = DataLoader(dev_td, batch_size=batch_size, num_workers=4)

