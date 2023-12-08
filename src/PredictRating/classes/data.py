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
        
        with open(data_path, 'r') as f:
            lines = [json.loads(line) for line in f]
            df = pd.DataFrame(lines)
        
        df = df[['review_text', 'rating']]
        df = df.rename(columns = {'review_text': 'review'})
        df = df[df.rating != 0]
        df = df.drop_duplicates()
        self.df = df.reset_index(drop = True)
        
        self.counts = self.df.rating.value_counts()

    def plot_bar(self) -> None:
        '''
        Bar plot of data frequencies.
        '''
        plt.bar(self.counts.index, self.counts.values)
        plt.title('Rating frequencies')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.show()

    def downsample(self):
        '''
        Downsample classes to the minimum frequency.
        '''
        min_freq = self.counts.values[-1]
        samples = []
        for rating in self.counts.index:
            df_with_rating = self.df[self.df.rating == rating]
            sample = df_with_rating.sample(min_freq, replace = False)
            samples.append(sample)
        self.df = pd.concat(samples).reset_index(drop = True)
        self.counts = self.df.rating.value_counts()
        