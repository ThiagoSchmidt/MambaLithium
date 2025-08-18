import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import os
import argparse
import time

start_time = time.time() # Start the timer to measure execution time
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--test', type=str, default='US06',
                    help='Test set')
parser.add_argument('--temp', type=str, default='25',
                    help='Temperature') 
parser.add_argument('--window', type=int, default=30, help='Window size for time sequences')                   

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed,args.cuda)

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        
        # Define layers individually for a more controlled forward pass
        self.in_linear = nn.Linear(in_dim, args.hidden)
        self.mamba = Mamba(self.config)
        self.out_linear = nn.Linear(args.hidden, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x is expected to be 3D: (batch, sequence_length, features)
        
        # 1. Project input features to the model's hidden dimension
        # Input: (batch, seq, features) -> Output: (batch, seq, hidden)
        x = self.in_linear(x)

        # 2. Process the entire sequence through the Mamba model
        # Input: (batch, seq, hidden) -> Output: (batch, seq, hidden)
        x = self.mamba(x)

        # 3. For prediction, we only need the output of the last time step
        # This contains the summarized information of the sequence.
        # Input: (batch, seq, hidden) -> Output: (batch, hidden)
        x = x[:, -1, :] 

        # 4. Pass through the final output layer and activation
        # Input: (batch, hidden) -> Output: (batch, out_dim)
        out = self.out_linear(x)
        out = self.sigmoid(out)

        # Flatten for the loss function (L1 loss expects 1D tensors)
        return out.flatten()

def PredictWithData(trainX, trainy, testX):
    clf = Net(trainX.shape[2], 1)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)
    xt = torch.from_numpy(trainX).float()
    xv = torch.from_numpy(testX).float()
    yt = torch.from_numpy(trainy).float()

    if args.cuda:
        clf = clf.cuda()
        xt, xv, yt = xt.cuda(), xv.cuda(), yt.cuda()

    for e in range(args.epochs):
        clf.train()
        pred = clf(xt)
        loss = F.l1_loss(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 10 == 0:
            print(f"Epoch {e} | Loss: {loss.item():.4f}")

    clf.eval()
    with torch.no_grad():
        preds = clf(xv).cpu().numpy()
    return preds

def ReadData(path, csv):
    f = os.path.join(path, csv)
    data = pd.read_csv(f)
    y = data['SOC'].values
    x = data.drop(['SOC', 'Profile'], axis=1).values
    x = scale(x)

    window = args.window
    xs, ys = [], []

    for i in range(len(x) - window):
        xs.append(x[i:i+window])
        ys.append(y[i+window])  # predict the next SOC

    return np.array(xs), np.array(ys)

path = './data/'+args.temp+'C'
datal = ['DST','FUDS','US06']
datal.remove(args.test)
xt1, yt1 = ReadData(path,datal[0]+'_'+args.temp+'C.csv')
xt2, yt2 = ReadData(path,datal[1]+'_'+args.temp+'C.csv')
trainX = np.concatenate([xt1, xt2], axis=0)
trainy = np.concatenate([yt1, yt2], axis=0)
testX,testy = ReadData(path,args.test+'_'+args.temp+'C.csv')
predictions = PredictWithData(trainX, trainy, testX)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print('MSE RMSE MAE R2')
evaluation_metric(testy, predictions)
plt.figure()
plt.plot(testy, label='True')
plt.plot(predictions, label='Estimation')
plt.title('SOC Estimation')
plt.xlabel('Time(sec)')
plt.ylabel('SOC value')
plt.legend()
plt.show()