from PredictRating.classes.data import Data
from PredictRating.classes.data import Data
from transformers import DistilBertTokenizer
from PredictRating.classes.customdataset import CustomDataset

def test_downsample():
    data = Data('reviews.json')
    assert len(data.df) == len(data.df.drop_duplicates()), 'There are duplicates in the data'
    data.plot_bar()
    print(data.df.head(50))
    
    min_freq = data.counts.values[-1]
    data.downsample()
    assert len(data.df) == len(data.df.drop_duplicates()), 'There are duplicates in the data'
    assert len(data.df) == min_freq * 5, 'Classes are not balanced'
    data.plot_bar()

def test_subset():
    data = Data('reviews.json')
    data.downsample()
    data.subset(1000)
    assert len(data.df) == 5000

def test_tokenizer():
    max_len = 359
    d = Data('reviews_light.json')
    t = CustomDataset(d.df, DistilBertTokenizer.from_pretrained('distilbert-base-cased'), max_len)
    assert type(t[3]) == dict
    assert len(t[5]['ids']) == max_len