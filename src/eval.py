from PredictRating.classes.bertnet import BertNet
from PredictRating.constants import *
from PredictRating.train import load_data

from sklearn.metrics import classification_report as report
from sklearn.metrics import confusion_matrix as conf_mat
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
from transformers import DistilBertTokenizer

def main():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    _, test_data = load_data(tokenizer = tokenizer, name = 'reviews.json',
                                      p = DATA_PERCENTAGE, split = TRAIN_SPLIT)
    
    bert = BertNet(model_name = 'model')
    if not bert._model_exists():
        print('Model does not exist. Please train a model first.')
        return
    bert.load_state_dict(torch.load(bert.model_path))
    bert.to(bert.device)

    correct, preds = bert.test_model(test_data)
    c_mat = conf_mat(correct, preds)
    disp = ConfusionMatrixDisplay(c_mat, display_labels = ['1', '2', '3', '4', '5'])
    disp.plot()
    plt.show()

    print(report(correct, preds))

    # Test a few of my own reviews
    print(bert.eval_review(['this book was really good, enjoyed it a lot',
                            'i hate this book!!!',
                            'this is the best book i have ever read!!!!, needs to be said; a must read!!',
                            'i hated this book from the beginning. really slow and boring...',
                            'cant believe i read this book!! so mid and bland and boring. clear 3/5!',
                            'i liked this book, but it is not the best... it starts well with good development of the \
                            characters, but then it just flattens and becomes kind of bland'], tokenizer))

if __name__ == '__main__':
    main()