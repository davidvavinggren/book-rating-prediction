import torch
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
                                            add_special_tokens = True, # Like [CLS] and [SEP]
                                            max_length = self.max_len, 
                                            padding = 'max_length', # Pad to max_length
                                            truncation = True) # Truncate to max_length

        # mask contains the attention mask; indicates what tokens are padding (== 0) and what are not (== 1)
        return {
            'text': review,
            'ids': torch.tensor(inputs['input_ids'], dtype = torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype = torch.long),
            'targets': torch.tensor(self.df.rating[idx], dtype = torch.long)
        } 
    
    def __len__(self):
        return len(self.df)