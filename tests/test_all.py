from PredictRating.classes.data import Data
from PredictRating.classes.customdataset import CustomDataset
from PredictRating.classes.basenet import BaseNet
from PredictRating.classes.bertnet import BertNet

from transformers import DistilBertTokenizer

def test_downsample():
    split = 0.8
    data = Data('reviews.json', split = split)
    assert len(data.df) == len(data.df.drop_duplicates()), 'There are duplicates in the data'
    data.plot_bar(data.df)
    data.plot_bar(data.train)
    data.plot_bar(data.test)

def test_subset():
    p = 0.2
    data = Data('reviews.json')
    n = len(data.df)
    data.df = data.subset(data.df, p)
    assert -10 < len(data.df) - n * p < 10

def test_tokenizer():
    max_len = 359
    d = Data('reviews_light.json')
    t = CustomDataset(d.df, DistilBertTokenizer.from_pretrained('distilbert-base-cased'), max_len)
    assert type(t[3]) == dict
    assert len(t[5][0]) == max_len

def test_device():
    base = BaseNet()
    assert base.device == 'mps' # use if on mac
    #assert base.device == 'gpu' # use if on PC

def test_model_load():
    bert = BertNet()
    bert._load_model()
    assert bert._model_loaded is True