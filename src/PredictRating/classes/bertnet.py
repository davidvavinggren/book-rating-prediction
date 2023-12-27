from PredictRating.classes.basenet import BaseNet
from PredictRating.constants import *

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from transformers import DistilBertModel

class BertNet(BaseNet):
    def __init__(self, model_name: str = 'model'):
        '''
        Initialize with the name of the model.

        The DistilBertModel outputs a tensor of dimension TRAIN_BATCH_SIZE x MAX_LEN x 768;
        for each batch, for each token in the sentence, the model returns a 768 long vector 
        representation of the token.
        '''
        super().__init__(model_name) # Inherit from BaseNet class
        self.bert_layer = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(params = self.parameters(), lr = LEARNING_RATE)

    def forward(self, ids, mask):
        '''
        ids and mask are both tensors of length MAX_LEN
        '''
        logits = self.bert_layer(ids, mask)[0]
        logits = self.classifier(logits[:, 0, :]) # Use [CLS] token
        return logits