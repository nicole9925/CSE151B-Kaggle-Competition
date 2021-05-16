import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path 
import numpy 
import pickle
from glob import glob
import pandas as pd

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler

"""Change to the data folder"""
new_path = "./new_train/new_train"
# number of sequences in each dataset
# train:205942  val:3200 test: 36272 
# sequences sampled at 10HZ rate
if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu" 
    
class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""
    def __init__(self, data_path: str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.pkl_list = glob(os.path.join(self.data_path, '*'))
        self.pkl_list.sort()
        
    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):

        pkl_path = self.pkl_list[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        if self.transform:
            data = self.transform(data)

        return data


# intialize a dataset
train_dataset  = ArgoverseDataset(data_path=new_path)

batch_sz = 32


def my_collate(batch):
    """ collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature] """
    inp = [numpy.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    out = [numpy.dstack([scene['p_out'], scene['v_out']]) for scene in batch]
    out = torch.LongTensor(out)
    inp = torch.tensor(inp, dtype=torch.float)
    out = torch.tensor(out, dtype=torch.float)
    return [inp, out]

# sampler = RandomSampler(train_dataset, num_samples = 1000, replacement=True)
sampler = RandomSampler(train_dataset)

train_loader = DataLoader(train_dataset,batch_size=batch_sz, shuffle = False, collate_fn=my_collate, num_workers=0, sampler = sampler, drop_last=True)



class NN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(NN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)   
        self.fc1 = nn.Linear(hidden_dim, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.gru(x, hidden)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden.to(device)

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch, log_interval=10000):
    model.train()
    iterator = tqdm(train_loader, total=int(len(train_loader)))
    running_loss = 0
    for batch_idx, (data, target) in enumerate(iterator):
        data, target = torch.reshape(data, (batch_sz, 60, -1)).to(device), torch.reshape(target[:,:,:,:2], (batch_sz, 60, -1)).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iterator.set_postfix(loss=running_loss)
    return running_loss
        
def test(model, device, test_loader):
    model.eval()
    loss_ema = -1
    with torch.no_grad():
        for data, target in test_loader:
            data, target = torch.reshape(data, (batch_sz, 60, -1)).to(device), torch.reshape(target[:,:,:,:2], (batch_sz, 60, -1)).to(device)
            output = model(data)
            loss = nn.MSELoss()(output, target)
            if loss_ema < 0:
                loss_ema = loss.item()
            loss_ema = loss_ema * 0.99 + loss.item()*0.01
    print("Test loss: {}".format(loss_ema))
    
input_dim = 76
hidden_dim = 60  
layer_dim = 4  
output_dim = 60   
    
learning_rate = 0.0001
momentum = 0.5

net = NN(input_dim, output_dim, hidden_dim, layer_dim)
model = net.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum, weight_decay=1e-5)

num_epoch = 10
losses = []
for epoch in range(1, num_epoch + 1):
    loss = train(model, device, train_loader, optimizer, epoch)
    losses.append(loss)
    torch.save(model.state_dict(), 'checkpoints/train-epoch-gru-full-{}.pth'.format(epoch + 1)) 
    
with open("losses_test.txt", "w") as output:
    output.write(str(losses))