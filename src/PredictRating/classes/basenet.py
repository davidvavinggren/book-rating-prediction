from PredictRating.constants import *

import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn.functional as F

class BaseNet(torch.nn.Module):
    '''
    Neural net base class.
    Models are saved as '<model_name>_state_dict.pth'.
    '''
    def __init__(self, model_name: str = 'model'):
        super().__init__()
        self._model_loaded = False
        self.model_name = model_name
        self.model_path = os.path.join('models', f'{self.model_name}_state_dict.pth')
        self.device = self._get_device()

    def _get_device(self):
        device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        print(f'Device = {device}')
        return device

    def _load_model(self):
        print(f'Loading {self.model_name}.pth...')
        self.load_state_dict(torch.load(self.model_path))
        self._model_loaded = True
        print('Model loaded!')

    def _save_model(self):
        if os.path.exists(self.model_path):
            confirm =  input('A model with the same name already exists. Are you sure you want to overwrite? [y/n] ')
            while True:
                if confirm == 'y':
                    torch.save(self.state_dict(), self.model_path)
                    print('Model saved!')
                    break
                elif confirm == 'n':
                    print('Model not saved!')
                    break
                else:
                    confirm = input('Invalid entry. Enter [y/n]: ')
        else:
            torch.save(self.state_dict(), self.model_path)
            print('Model saved!')

    def _model_exists(self):
        if os.path.exists(self.model_path):
            return True
        return False
    
    def _confirm_train(self):
        if self._model_exists():
            answ = input(f'There already exists a model with the name "{self.model_name}". Do you want to train and overwrite anyways? [y/n] ')
            while answ != 'y' and answ != 'n':
                answ = input('Invalid entry. Enter [y/n]: ')
            if answ == 'n':
                print('Aborting!')
                exit()
        print()

    def _update_plot(self, ax, train_losses, test_losses, epoch):
        # Update the plot
        ax.clear()
        ax.plot(train_losses, label='Training loss')
        ax.plot(test_losses, label='Validation loss')
        ax.set_title(f"Average Batch Loss, Epoch: {epoch + 1}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.draw()
        plt.pause(0.1)

    def train_model(self, train_data: 'torch.utils.data.DataLoader', test_data: 'torch.utils.data.DataLoader',
                    epochs: int = 3, display_result: bool = False) -> None:
        '''
        Train model. Specify number of epochs and whether or not to plot the result.
        '''
        self._confirm_train()

        if display_result is True:
            # Set up plot; turn on interactive mode
            plt.ion()
            fig, ax = plt.subplots()

        train_losses = []
        test_losses = []
        for epoch in tqdm(range(epochs)):
            print(f'\n\nEpoch {epoch + 1}\n-------------------------------')
            train_losses.append(self._train(train_data))
            test_losses.append(self._eval(test_data))
            if display_result is True:
                self._update_plot(ax, train_losses, test_losses, epoch)
        print('Traning finished!')

        if display_result is True:
            # Keep plot; turn off interactive mode
            plt.ioff()
            plt.show()
        
        self._save_model()

    def _train(self, data):
        num_samples = len(data.dataset)
        num_batches = len(data)
        self.train()
        train_loss, correct = 0, 0
        for batch, (ids, masks, targets) in enumerate(data):
            # Send data to device
            ids, masks, targets = (tensor.to(self.device, dtype = torch.long) for tensor in [ids, masks, targets])

            # Forward pass
            logits = self(ids, masks)
            loss = self.loss_function(logits, targets)
            train_loss += loss.item()
            preds = F.softmax(logits, dim = 1).argmax(1)
            correct += (preds == targets).type(torch.float).sum().item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % int(num_batches / 10) == 0:
                loss, current = loss.item(), (batch + 1) * len(ids)
                print(f'loss: {loss:>7f}    [{current:>5d}/{num_samples:>5d}]')
        
        train_loss /= num_batches
        correct /= num_samples
        print(f"Train Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg batch loss: {train_loss:>8f} \n")

        return train_loss
    
    def _eval(self, data):
        num_samples = len(data.dataset)
        num_batches = len(data)
        self.eval()
        eval_loss, correct = 0, 0
        with torch.no_grad():
            for (ids, masks, targets) in data:
                # Send data to device
                ids, masks, targets = (tensor.to(self.device, dtype = torch.long) for tensor in [ids, masks, targets])

                # Forward pass
                logits = self(ids, masks)
                loss = self.loss_function(logits, targets)
                eval_loss += loss.item()
                preds = F.softmax(logits, dim = 1).argmax(1)
                correct += (preds == targets).type(torch.float).sum().item()
            
        eval_loss /= num_batches
        correct /= num_samples
        print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg batch loss: {eval_loss:>8f} \n")
        
        return eval_loss
            
    def test_model(self, test_data: 'torch.utils.data.DataLoader') -> tuple:
        '''
        Make inference on the test data. Return correct list and prediction list.
        '''
        self.eval()
        correct = []
        preds = []
        with torch.no_grad():
            for (ids, masks, targets) in test_data:
                # Send data to device
                ids, masks, targets = (tensor.to(self.device, dtype = torch.long) for tensor in [ids, masks, targets])
                correct += (targets + 1).tolist() # Add 1 to get ratings instead of classes

                # Forward pass
                logits = self(ids, masks)
                preds += (F.softmax(logits, dim = 1).argmax(1) + 1).tolist() # Add 1 to get ratings instead of classes
        
        return correct, preds


    def eval_review(self, reviews: str, tokenizer) -> int:
        '''
        Evaluate a list of reviews. Return list of predicted ratings.
        '''
        ratings = []
        for review in reviews:
            encoding = tokenizer.encode_plus(text = review,
                                            add_special_tokens = True, # Like [CLS] and [SEP]
                                            max_length = MAX_LEN, 
                                            padding = 'max_length', # Pad to max_length
                                            truncation = True) # Truncate to max_length

            ids = torch.tensor(encoding['input_ids'], dtype = torch.long).to(self.device)
            mask = torch.tensor(encoding['attention_mask'], dtype = torch.long).to(self.device)
            
            logits = self(ids, mask)
            pred = F.softmax(logits, dim = 1).argmax(1).item()
            ratings.append(pred + 1)
        return ratings