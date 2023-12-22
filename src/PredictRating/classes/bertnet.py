from PredictRating.classes.basenet import BaseNet
from PredictRating.constants import *

import torch.nn as nn
from transformers import DistilBertModel

class BertNet(nn.Module):
    def __init__(self):
        super().__init__() # Inherit from BaseNet class
        self.bert_layer = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout_layer = nn.Dropout(DROPOUT_PROB)
        self.linear_layer = nn.Linear(768, 1)

    def forward(self, ids, mask):
        output_bert = self.bert_layer(ids, mask)
        output_dropout = self.dropout_layer(output_bert[0])
        output_linear = self.linear_layer(output_dropout)
        return output_linear