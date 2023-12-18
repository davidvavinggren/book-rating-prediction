import pandas as pd
import os
import json
import matplotlib.pyplot as plt

class Data:
    ''' 
    Data preprocessing class. Level the class frequencies and remove duplicates. 
    '''
    def __init__(self, file: str):
        data_path = os.path.join(os.getcwd(), 'data', file)
        if not os.path.isfile(data_path):
            raise FileNotFoundError
        
        self.data_path = data_path
        
        # Data is on JSONL format; read line by line and create pandas df
        with open(data_path, 'r') as f:
            lines = [json.loads(line) for line in f]
            df = pd.DataFrame(lines)
        
        df = df[['review_text', 'rating']] # Remove irrelevant comlumns
        df = df.rename(columns = {'review_text': 'review'}) # Rename column review_text to review
        df = df[df.rating != 0] # Remove all reviews with no rating attached to it
        df = df.drop_duplicates()
        self.df = df.sample(frac = 1).reset_index(drop = True) # Shuffle df
        
        self.counts = self.df.rating.value_counts() # Count frequencies

    def plot_bar(self) -> None:
        '''
        Bar plot of data frequencies.
        '''
        plt.bar(self.counts.index, self.counts.values)
        plt.title('Rating frequencies')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.show()

    def downsample(self) -> None:
        '''
        Downsample classes to the minimum frequency.
        '''
        min_freq = self.counts.values[-1]
        samples = []
        for rating in self.counts.index:
            df_with_rating = self.df[self.df.rating == rating] # Find all reviews with rating == rating
            sample = df_with_rating.sample(min_freq, replace = False) # Sample min_freq of them
            samples.append(sample)
        self.df = pd.concat(samples).reset_index(drop = True) # Concat into one df 
        
        self.df = self.df.sample(frac = 1).reset_index(drop = True) # Shuffle df 
        self.counts = self.df.rating.value_counts()

    def subset(self, n: int) -> None:
        '''
        Create subset of downsampled data; n samples in each class.
        '''
        if n > len(self.df):
            raise ValueError(f'Not possible to sample more than {len(self.df)} points!')
        samples = [self.df[self.df.rating == rating].sample(n, replace = False) for rating in range(1, 6)]
        self.df = pd.concat(samples).reset_index(drop = True)
        # Shuffle the df
        self.df = self.df.sample(frac = 1).reset_index(drop = True) 