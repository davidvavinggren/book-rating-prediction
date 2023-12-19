import pandas as pd
import os
import json
import matplotlib.pyplot as plt

class Data:
    ''' 
    Data preprocessing class. Level the class frequencies and remove duplicates. 
    '''
    def __init__(self, file: str, split: float = 0, p: float = 0):
        '''
        split is the training set percentage, so set split = 0.8 if you want 80/20 train/test split.
        p is the data percentage; p = 0.2 gives you 20 % of the original data set with class distribution preserved.
        '''
        data_path = os.path.join(os.getcwd(), 'data', file)
        if not os.path.isfile(data_path):
            raise FileNotFoundError
        
        self.data_path = data_path
        self.df = self.load_jsonl_data()
        if p:
            self.df = self.subset(self.df, p)
        
        if split:
            self.train, self.test = self.create_train_test(split)

    def load_jsonl_data(self) -> pd.DataFrame:
        '''
        Load data on JSONL format into df. Preprocess data.
        '''
        # Data is on JSONL format; read line by line and create pandas df
        with open(self.data_path, 'r') as f:
            lines = [json.loads(line) for line in f]
            df = pd.DataFrame(lines)
        
        df = df[['review_text', 'rating']] # Remove irrelevant comlumns
        df = df.rename(columns = {'review_text': 'review'}) # Rename column review_text to review
        df = df[df.rating != 0] # Remove all reviews with no rating attached to it
        df = df.drop_duplicates()
        df = df.sample(frac = 1).reset_index(drop = True) # Shuffle df
        return df

    def plot_bar(self, df: pd.DataFrame) -> None:
        '''
        Bar plot of data frequencies.
        '''
        counts = df.rating.value_counts()
        plt.bar(counts.index, counts.values)
        plt.title('Rating frequencies')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.show()

    def undersample(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Undersample df to minimum frequency.
        '''
        counts = df.rating.value_counts()
        min_freq = counts.values[-1]
        samples = []
        for rating in counts.index:
            df_with_rating = df[df.rating == rating] # Find all reviews with rating == rating
            sample = df_with_rating.sample(min_freq, replace = False) # Sample min_freq of them
            samples.append(sample)
        df = pd.concat(samples).reset_index(drop = True) # Concat into one df 
        df = df.sample(frac = 1).reset_index(drop = True) # Shuffle df 
        return df

    def subset(self, df: pd.DataFrame, p: float) -> pd.DataFrame:
        '''
        Create subset of data; (p * 100) percent of each class.
        '''
        samples = []
        for rating in range(1,6):
            rating_set = df[df.rating == rating]
            n = len(rating_set)
            samples.append(rating_set.sample(int(p * n), replace = False)) 
        df = pd.concat(samples).reset_index(drop = True)
        # Shuffle the df
        df = df.sample(frac = 1).reset_index(drop = True)
        return df 
    
    def create_train_test(self, split: float) -> tuple:
        '''
        Create training set and test set. Undersample the training set before returning.
        split is the training set percentage, so set split = 0.8 if you want 80/20 train/test split.
        '''
        train = self.df.sample(frac = split).reset_index(drop = True)
        test = self.df.drop(train.index).reset_index(drop = True)

        train = self.undersample(train)
        return (train, test)