import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import os
import argparse
import time

# ---------------------------
# 0. Arguments and setup
# ---------------------------
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False, help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Dimension of representations')
parser.add_argument('--layer', type=int, default=1, help='Num of layers')
parser.add_argument('--window', type=int, default=1, help='Window size for time sequences')
parser.add_argument('--task', type=str, default='SOC', help='Target variable: SOC, SOH or RUL')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

# ---------------------------
# 1. Evaluation Metric
# ---------------------------
def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE, RMSE, MAE, R2))

# ---------------------------
# 2. Reproducibility
# ---------------------------
def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed, args.cuda)

# ---------------------------
# 3. Mamba Neural Network
# ---------------------------
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.in_linear = nn.Linear(in_dim, args.hidden)
        self.mamba = Mamba(self.config)
        self.out_linear = nn.Linear(args.hidden, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.in_linear(x)
        x = self.mamba(x)
        x = x[:, -1, :]  # last time step
        out = self.out_linear(x)
        out = self.sigmoid(out)
        return out.flatten()

# ---------------------------
# 4. Training Function
# ---------------------------
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

# ---------------------------
# 5. Data Reading Function
# ---------------------------
def ReadData(path, csv, target):
    f = os.path.join(path, csv)
    data = pd.read_csv(f)

    # select the target variable
    y = data[target].values
    # remove non-feature columns
    x = data.drop(['SOC', 'SOH', 'RUL'], axis=1).values
    x = scale(x)

    window = args.window
    xs, ys = [], []

    for i in range(len(x) - window):
        xs.append(x[i:i+window])
        ys.append(y[i+window])  # predict the next target value

    return np.array(xs), np.array(ys)

# ---------------------------
# 6. Load NASA/CALCE Battery Repository
# ---------------------------
path = './data'  # adjust to your folder location
xt1, yt1 = ReadData(path, 'battery5_with_SOC_SOH_RUL.csv', args.task)
xt2, yt2 = ReadData(path, 'battery6_with_SOC_SOH_RUL.csv', args.task)
trainX = np.concatenate([xt1, xt2], axis=0)
trainy = np.concatenate([yt1, yt2], axis=0)

testX, testy = ReadData(path, 'battery7_with_SOC_SOH_RUL.csv', args.task)

# ---------------------------
# 7. Run Training & Evaluation
# ---------------------------
predictions = PredictWithData(trainX, trainy, testX)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print('MSE RMSE MAE R2')
evaluation_metric(testy, predictions)

plt.figure()
plt.plot(testy, label='True')
plt.plot(predictions, label='Estimation')
plt.title(f'{args.task} Estimation')
plt.xlabel('Time (sec)')
plt.ylabel(f'{args.task} value')
plt.legend()
plt.show()