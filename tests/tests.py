from PredictRating.classes.data import Data

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

def test_tokenizer():
    data = Data('reviews_light.json').downsample()