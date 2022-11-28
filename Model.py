import torch
from torch import nn
from transformers import BertModel, BertConfig

model_path = "./data/pretrained_model/chinese_roberta_wwm_large_ext_pytorch/"
bert_config = BertConfig.from_pretrained(model_path + 'bert_config.json', output_hidden_states=True)


class BertClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(BertClassifier, self).__init__()
        self.model_name = 'BertClassifier'
        self.bert_model = BertModel.from_pretrained(model_path, config=bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size * 2, n_classes)

    def forward(self, input_ids, input_masks, segment_ids):
        hidden_output = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks)
        sequence_output, pooler_output, hidden_states = hidden_output[0], hidden_output[1], hidden_output[2]
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        logit = self.classifier(concat_out)
        return logit