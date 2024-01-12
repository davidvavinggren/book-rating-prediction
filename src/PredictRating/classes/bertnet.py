from PredictRating.classes.basenet import BaseNet
from PredictRating.constants import *

import torch.nn as nn
from torch.optim.adam import Adam

class BertNet(BaseNet):
    def __init__(self, bert_model, model_name: str = 'model'):
        '''
        Initialize with the name of the model. Use model_name = bert or distilbert

        The DistilBertModel outputs a tensor of dimension TRAIN_BATCH_SIZE x MAX_LEN x 768;
        for each batch, for each token in the sentence, the model returns a 768 long vector 
        representation of the token.
        '''
        super().__init__(model_name) # Inherit from BaseNet class
        self.bert_layer = bert_model
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.linear_layer = nn.Linear(768, 5)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(params = self.parameters(), lr = LEARNING_RATE)

    def forward(self, ids, mask):
        '''
        ids and mask are both tensors of length MAX_LEN
        '''
        logits = self.bert_layer(ids, mask)[0]
        logits = self.dropout(logits)
        logits = self.linear_layer(logits[:, 0, :]) # Use [CLS] token
        return logits