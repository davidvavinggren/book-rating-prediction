from PredictRating.classes.data import Data
from PredictRating.classes.customdataset import CustomDataset
from PredictRating.classes.bertnet import BertNet
from PredictRating.constants import *

from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import BertTokenizer
from transformers import BertModel

def load_data(tokenizer, name, p, split):
    print('\nLoading data...')
    data = Data(name, p = p, split = split)
    train_data = CustomDataset(data.train, tokenizer, MAX_LEN)
    test_data = CustomDataset(data.test, tokenizer, MAX_LEN)
    print('Data loaded!')

    print(f'\nTrain size = {len(train_data)}')
    print(f'Test size = {len(test_data)}\n')

    return (DataLoader(train_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True), 
            DataLoader(test_data, batch_size = TEST_BATCH_SIZE, shuffle = True))

def main(model_name, model, tokenizer):
    bert = BertNet(model, model_name = model_name)
    bert.to(bert.device)

    train_data, test_data = load_data(tokenizer = tokenizer, name = 'reviews.json',
                                      p = DATA_PERCENTAGE, split = TRAIN_SPLIT)
    
    print(f'Using {len(train_data)} batches of size {TRAIN_BATCH_SIZE} for training')
    print(f'Using {len(test_data)} batches of size {TEST_BATCH_SIZE} for evaluating\n')

    bert.train_model(train_data = train_data, test_data = test_data, epochs = EPOCHS, display_result = True)

if __name__ == '__main__':
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-cased')
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    bert_model = BertModel.from_pretrained('bert-base-cased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
   
    # Set what model to use during training here; bert or distilbert
    main('distilbert', distilbert_model, distilbert_tokenizer)