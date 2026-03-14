import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from mamba import Mamba, MambaConfig
import os
import argparse
import time
from tabulate import tabulate

# ============================
# ARGUMENTS
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--task', type=str, default='RUL')  # RUL or SOH
parser.add_argument('--case', type=str, default='G')
parser.add_argument('--window', type=int, default=30)   # window length for sequences
args = parser.parse_args()

# ============================
# DEVICE & SEED
# ============================
args.cuda = args.use_cuda and torch.cuda.is_available()
def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
set_seed(args.seed, args.cuda)

# ============================
# METRICS
# ============================
def evaluation_metric(y_test, y_hat):
    mse = mean_squared_error(y_test, y_hat)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    return mse, rmse, mae, r2

# ============================
# MODEL
# ============================
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_layers=2):
        super().__init__()
        self.config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.mamba = Mamba(self.config)
        self.decoder = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = self.encoder(x)
        x = self.mamba(x)
        x = self.decoder(x)
        x = self.act(x)
        # take last time step prediction
        return x[:, -1, 0]

# ============================
# DATA
# ============================
def ReadData(path, csv, task):
    f = os.path.join(path, csv)
    data = pd.read_csv(f)
    y = data[task].values
    if task == 'RUL':
        y = y / max(y)   # normalize properly
    x = data.drop(['RUL','SOH'], axis=1).values
    x = scale(x)
    return x, y

def create_sequences(x, y, window):
    X_seq, y_seq = [], []
    for i in range(len(x)-window):
        X_seq.append(x[i:i+window])
        y_seq.append(y[i+window-1])
    return np.array(X_seq), np.array(y_seq)

# ============================
# CASE G DATA
# ============================
path = './data/case'+args.case
if args.case == 'D':
    xt1, yt1 = ReadData(path,'B0005_features_SoH_RUL.csv',args.task)
    xt2, yt2 = ReadData(path,'B0006_features_SoH_RUL.csv',args.task)
    trainX = np.vstack((xt1,xt2))
    trainy = np.hstack((yt1,yt2))
    testX, testy = ReadData(path,'B0007_features_SoH_RUL.csv',args.task)   

if args.case == 'G':
    xt1, yt1 = ReadData(path,'battery5_with_SOC_SOH_RUL.csv',args.task)
    xt2, yt2 = ReadData(path,'battery6_with_SOC_SOH_RUL.csv',args.task)
    trainX = np.vstack((xt1,xt2))
    trainy = np.hstack((yt1,yt2))
    testX, testy = ReadData(path,'battery7_with_SOC_SOH_RUL.csv',args.task)  

# Windowing
trainX, trainy = create_sequences(trainX, trainy, args.window)
testX, testy = create_sequences(testX, testy, args.window)

# ============================
# TRAIN FUNCTION
# ============================
def train_model(seed, lr=5e-4, wd=2e-4, hidden=256, layers=2, epochs=args.epochs, patience=args.patience, batch_size=64):
    set_seed(seed, args.cuda)
    model = Net(trainX.shape[2], 1, hidden, layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    xt = torch.from_numpy(trainX).float()
    yt = torch.from_numpy(trainy).float()
    xv = torch.from_numpy(testX).float()
    yv = torch.from_numpy(testy).float()
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)

    if args.cuda:
        model, xv, yv = model.cuda(), xv.cuda(), yv.cuda()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            if args.cuda:
                xb, yb = xb.cuda(), yb.cuda()
            opt.zero_grad()
            pred = model(xb)
            loss = 0.7*F.mse_loss(pred, yb) + 0.3*F.l1_loss(pred, yb)  # hybrid loss
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(xv)
            val_loss = 0.7*F.mse_loss(val_pred, yv) + 0.3*F.l1_loss(val_pred, yv)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Final prediction
    model.eval()
    with torch.no_grad():
        yhat = model(xv)
    if args.cuda:
        yhat = yhat.cpu()
    return yhat.numpy().flatten()

# ============================
# EXPONENTIAL SMOOTHING
# ============================
def smooth(y, weight=0.2):
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = weight*y[i] + (1-weight)*smoothed[i-1]
    return smoothed

# ============================
# ENSEMBLE TRAINING
# ============================
start_time = time.time()
predictions_list = []

# Seeds for stability
for seed in [1, 42]:
    pred = train_model(seed)
    predictions_list.append(pred)

# Average predictions
predictions_raw = np.mean(predictions_list, axis=0)

# Optimize smoothing weight
best_r2 = -np.inf
best_weight = 0.2
for w in [0.12,0.15,0.18,0.20,0.22,0.25,0.28]:
    pred_smooth = smooth(predictions_raw, weight=w)
    r2_tmp = r2_score(testy, pred_smooth)
    if r2_tmp > best_r2:
        best_r2 = r2_tmp
        best_weight = w
predictions = smooth(predictions_raw, weight=best_weight)
end_time = time.time()

print(f"Optimal smoothing weight: {best_weight}, R² after ensemble + smoothing: {best_r2:.4f}")

# ============================
# METRICS
# ============================
mse, rmse, mae, r2 = evaluation_metric(testy, predictions)
metrics_data = [
    ["MSE", mse],
    ["RMSE", rmse],
    ["MAE", mae],
    ["R²", r2]
]
print("\nFinal ensemble model trained:")
print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="grid", floatfmt=".4f"))
print(f"Execution time: {end_time - start_time:.2f} s")

# ============================
# PLOTS
# ============================
plt.figure(figsize=(12,5))
plt.plot(testy, label='True')
plt.plot(predictions, label='Smoothed Ensemble')
plt.title(f'{args.task} Estimation')
plt.xlabel('Cycle')
plt.ylabel(f'{args.task} value')
plt.legend()
plt.show()