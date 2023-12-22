from PredictRating.classes.data import Data
from PredictRating.classes.customdataset import CustomDataset
from PredictRating.classes.bertnet import BertNet
from PredictRating.constants import *

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'\nDevice = {device}')

model = BertNet()
model.to(device)

print('\nLoading data...')
data = Data('reviews.json', p = DATA_PERCENTAGE, split = TRAIN_PROP)
train_data = CustomDataset(data.train, tokenizer, MAX_LEN)
test_data = CustomDataset(data.test, tokenizer, MAX_LEN)
print('Data loaded!')

print(f'\nTrain size = {len(train_data)}')
print(f'Test size = {len(test_data)}\n')

train_loader = DataLoader(train_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_data, batch_size = TEST_BATCH_SIZE, shuffle = True)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr = LEARNING_RATE)

def train(epoch):
    model.train()
    for _, data in enumerate(train_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask).squeeze()

        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        if _ % 5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in tqdm(range(EPOCHS)):
    train(epoch)

torch.save(model, 'model.pth')

#def valid(model, testing_loader):
#    model.eval()
#    n_correct = 0; n_wrong = 0; total = 0
#    with torch.no_grad():
#        for _, data in enumerate(testing_loader, 0):
#            ids = data['ids'].to(device, dtype = torch.long)
#            mask = data['mask'].to(device, dtype = torch.long)
#            targets = data['targets'].to(device, dtype = torch.long)
#            outputs = model(ids, mask).squeeze()
#            big_idx = torch.argmax(outputs.data)
#            total+=targets.size(0)
#            n_correct+=(big_idx==targets).sum().item()
#    return (n_correct*100.0)/total
#
#print('This is the validation section to print the accuracy and see how it performs')
#print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')
#
#acc = valid(model, test_loader)
#print("Accuracy on test data = %0.2f%%" % acc)