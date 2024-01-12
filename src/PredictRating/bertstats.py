from PredictRating.classes.bertnet import BertNet
from PredictRating.constants import *
from PredictRating.train import load_data
from PredictRating.classes.customdataset import CustomDataset

from sklearn.metrics import classification_report as report
from sklearn.metrics import confusion_matrix as conf_mat
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from transformers import DistilBertModel
from transformers import BertTokenizer
from transformers import BertModel

def count_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

def plot_accs(cmat):
    cmat = torch.tensor(cmat)
    colsums = torch.sum(cmat, dim = 1)
    percentages = torch.diag(cmat) / colsums
    percentages[torch.isnan(percentages)] = 0

    colored_heights = colsums * percentages
    x = torch.arange(len(colsums)) + 1
    plt.figure()
    plt.bar(x, colsums, color='grey', alpha=0.5)
    plt.bar(x, colored_heights, color='blue', label='% correct')

    plt.xlabel('Ratings')
    plt.ylabel('Frequency')
    plt.title('Frequency Bars Colored by Percentage')
    plt.legend(loc = 'upper left')
    plt.grid(True, color='k', linestyle='-', linewidth=0.1)

    plt.tight_layout()
    plt.show()
    #plt.savefig('plot1', dpi = 1000)


def plot_devs(correct, preds):
    correct, preds = torch.tensor(correct, dtype = torch.float32), torch.tensor(preds, dtype = torch.float32)
    devs = torch.abs(correct - preds)

    bins = torch.arange(-0.5, 5, 1)
    plt.figure()
    plt.hist(devs, bins = bins, color='grey', alpha=0.5, rwidth = 0.8)
    plt.grid(True, color='k', linestyle='-', linewidth=0.1)

    plt.xlabel('Rating deviation')
    plt.ylabel('Frequency')
    plt.title(f'Rating deviation distribution')
    
    plt.tight_layout()
    plt.show()
    #plt.savefig('plot2', dpi = 1000)
    

def main(model_name, model, tokenizer):
    '''
    Evaluate the BERT classifier. Both on goodreads dataset, handwritten reviews and the Amazon product review dataset.
    '''
    _, test_data = load_data(tokenizer = tokenizer, name = 'reviews.json',
                                      p = DATA_PERCENTAGE, split = TRAIN_SPLIT)
    
    # Load bert model
    bert = BertNet(model, model_name = model_name)
    if not bert._model_exists():
        print('Model does not exist. Please train a model first.')
        return
    bert.load_state_dict(torch.load(bert.model_path))
    print(f'\nTotal number of parameters: {count_params(bert)}')
    bert.to(bert.device)

    # Make predictions on test set
    correct, preds = bert.test_model(test_data, tokenizer)
    cmat = conf_mat(correct, preds)
    acc = cmat.diagonal().sum() / cmat.sum()
    mse = mean_squared_error(correct, preds)
    print('acc:', acc)
    print('mse:', mse)
    plot_accs(cmat)
    plot_devs(correct, preds)
    disp = ConfusionMatrixDisplay(cmat, display_labels = ['1', '2', '3', '4', '5'])
    disp.plot()
    plt.show()
    #plt.savefig('plot3', dpi = 1000)
    print(report(correct, preds))

    # Test a few of my own reviews
    print(bert.eval_review(['this book was really good, enjoyed it a lot', 
                            'i hate this book!!!',
                            'this is the best book i have ever read!!!!, needs to be said; a must read!!',
                            'i hated this book from the beginning. really slow and boring...',
                            'cant believe i read this book!! so mid and bland and boring. clear 3/5!',
                            'I liked this book, but it did not understand the plot? Is it just me?'], tokenizer))

    # Test a few Matilda's reviews
    print(bert.eval_review(['My friend David told me to read this book. I dont know why I am surprised, but this was by far the worst book Ive ever read. He has no taste in books, obviously!!',
                            'Hmm, Im torn about this book... It didnt really catch my attention at first, but it actually grew on me! The character Matilda: comedian! Im not really happy about the ending though.' , 
                             'LET. ME. TELL. YOU! I wasnt convinced at first, but it turned out to be the most heartbreaking yet beautiful book Ive read this year. SO well written! Recommend! Recommend!'], tokenizer))
    
    # Load amazon data set
    test_amzn = pd.read_csv(os.path.join('data', 'reviews_amzn.csv'))
    test_amzn = test_amzn[['Text', 'Score']] # Remove irrelevant comlumns
    test_amzn = test_amzn.rename(columns = {'Text': 'review', 'Score': 'rating'}) # Rename column review_text to review
    test_amzn = test_amzn.drop_duplicates()
    test_amzn = test_amzn.sample(frac = 1, random_state = 100).reset_index(drop = True) # Shuffle df
    test_amzn = CustomDataset(test_amzn, tokenizer, MAX_LEN)
    test_amzn = DataLoader(test_amzn, batch_size = TEST_BATCH_SIZE, shuffle = True)

    # Make predictions on amazon product reviews
    correct, preds = bert.test_model(test_amzn, tokenizer)
    cmat = conf_mat(correct, preds)
    mse = mean_squared_error(correct, preds)
    print('mse:', mse)
    plot_accs(cmat)
    plot_devs(correct, preds)
    disp = ConfusionMatrixDisplay(cmat, display_labels = ['1', '2', '3', '4', '5'])
    disp.plot()
    plt.show()
    #plt.savefig('plot4', dpi = 1000)
    print(report(correct, preds))

if __name__ == '__main__':
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-cased')
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    bert_model = BertModel.from_pretrained('bert-base-cased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
   
    # Set what model to use during training here; bert or distilbert
    main('distilbert', distilbert_model, distilbert_tokenizer)