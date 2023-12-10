from torch import tensor, long
from PredictRating.classes.data import Data
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    '''
    Custom dataset class. Includes tokenization.
    '''
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, idx):
        review = self.df.review[idx]
        inputs = self.tokenizer.encode_plus(text = review,
                                            add_special_tokens = True,
                                            max_length = self.max_len, 
                                            padding = 'max_length', 
                                            truncation = True)

        return {
            'ids': tensor(inputs['input_ids'], dtype = long),
            'mask': tensor(inputs['attention_mask'], dtype = long),
            'targets': tensor(self.df.rating[idx], dtype = long)
        } 
    
    def __len__(self):
        return len(self.df)

d = Data('reviews_light.json')
d.downsample()
t = CustomDataset(d.df, DistilBertTokenizer.from_pretrained('distilbert-base-cased'), 512)
print(t[3])