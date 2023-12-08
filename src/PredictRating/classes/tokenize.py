from torch import tensor, long
from PredictRating.classes.data import Data
from transformers import DistilBertTokenizer

class Tokenizer():
    '''
    Class for tokenizing text; take dataframe as input and tokenize.
    '''
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, index):
        review = self.data.review[index]
        inputs = self.tokenizer.encode_plus(text = review,
                                            add_special_tokens = True,
                                            max_length = self.max_len, 
                                            padding = 'max_length', 
                                            truncation = True)

        return {
            'ids': tensor(inputs['input_ids'], dtype = long),
            'mask': tensor(inputs['attention_mask'], dtype = long),
            'targets': tensor(self.data.rating[index], dtype = long)
        } 
    
    def __len__(self):
        return len(self.data)

d = Data('reviews_light.json')
d.downsample()
t = Tokenizer(d.df, DistilBertTokenizer.from_pretrained('distilbert-base-cased'), 512)
print(t[3])