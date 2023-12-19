from PredictRating.classes.data import Data
from PredictRating.classes.customdataset import CustomDataset
from PredictRating.constants import *

from transformers import DistilBertTokenizer

from torch.utils.data import DataLoader

print('Loading data...')
data = Data('reviews.json', p = DATA_PERCENTAGE, split = TRAIN_PROP)
train_data = CustomDataset(data.train, DistilBertTokenizer.from_pretrained('distilbert-base-cased'), MAX_LEN)
test_data = CustomDataset(data.test, DistilBertTokenizer.from_pretrained('distilbert-base-cased'), MAX_LEN)
print('Data loaded!')

print(f'Train size = {len(train_data)}')
print(f'Test size = {len(test_data)}')

train_loader = DataLoader(train_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)