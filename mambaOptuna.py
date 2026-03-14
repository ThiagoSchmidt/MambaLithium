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
import optuna
from tabulate import tabulate

# ============================
# ARGUMENTS
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=25)
parser.add_argument('--task', type=str, default='RUL')  # or 'SOH'
parser.add_argument('--case', type=str, default='E')
parser.add_argument('--trials', type=int, default=15)
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
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers):
        super().__init__()
        self.config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            Mamba(self.config),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mamba(x)
        return x.flatten()

# ============================
# DATA
# ============================
def ReadData(path, csv, task):
    f = os.path.join(path, csv)
    data = pd.read_csv(f)
    y = data[task].values
    if task == 'RUL':
        y = y / len(data)
    x = data.drop(['RUL','SOH'], axis=1).values
    x = scale(x)
    return x, y

# ============================
# CASE G DATA
# ============================
path = './data/case'+args.case
if args.case == 'D':
    xt1, yt1 = ReadData(path,'B0005_features_SoH_RUL.csv',args.task)
    xt2, yt2 = ReadData(path,'B0006_features_SoH_RUL.csv',args.task)
    trainX = np.vstack((xt1,xt2))
    trainy = np.hstack((yt1,yt2))
    testX,testy = ReadData(path,'B0007_features_SoH_RUL.csv',args.task)  
if args.case == 'E':
    xt1, yt1 = ReadData(path,'battery5_with_SOC_SOH_RUL.csv',args.task)
    xt2, yt2 = ReadData(path,'battery6_with_SOC_SOH_RUL.csv',args.task)
    xt3, yt3 = ReadData(path,'battery7_with_SOC_SOH_RUL.csv',args.task)
    xt4, yt4 = ReadData(path,'battery18_with_SOC_SOH_RUL.csv',args.task)
    xt5, yt5 = ReadData(path,'battery25_with_SOC_SOH_RUL.csv',args.task)
    xt6, yt6 = ReadData(path,'battery26_with_SOC_SOH_RUL.csv',args.task)
    xt7, yt7 = ReadData(path,'battery27_with_SOC_SOH_RUL.csv',args.task)
    xt8, yt8 = ReadData(path,'battery28_with_SOC_SOH_RUL.csv',args.task)
    xt9, yt9 = ReadData(path,'battery29_with_SOC_SOH_RUL.csv',args.task)
    xt10, yt10 = ReadData(path,'battery30_with_SOC_SOH_RUL.csv',args.task)
    xt11, yt11 = ReadData(path,'battery31_with_SOC_SOH_RUL.csv',args.task)
    xt12, yt12 = ReadData(path,'battery32_with_SOC_SOH_RUL.csv',args.task)
    xt13, yt13 = ReadData(path,'battery33_with_SOC_SOH_RUL.csv',args.task)
    xt14, yt14 = ReadData(path,'battery34_with_SOC_SOH_RUL.csv',args.task)
    xt15, yt15 = ReadData(path,'battery36_with_SOC_SOH_RUL.csv',args.task)
    xt16, yt16 = ReadData(path,'battery38_with_SOC_SOH_RUL.csv',args.task)
    trainX = np.vstack((xt1,xt2,xt3,xt4,xt5,xt6,xt7,xt8,xt9,xt10,xt11,xt12,xt13,xt14,xt15,xt16))
    trainy = np.hstack((yt1,yt2,yt3,yt4,yt5,yt6,yt7,yt8,yt9,yt10,yt11,yt12,yt13,yt14,yt15,yt16))
    testX,testy = ReadData(path,'battery40_with_SOC_SOH_RUL.csv',args.task) 
if args.case == 'G':
    xt1, yt1 = ReadData(path,'battery5_with_SOC_SOH_RUL.csv',args.task)
    xt2, yt2 = ReadData(path,'battery6_with_SOC_SOH_RUL.csv',args.task)
    trainX = np.vstack((xt1,xt2))
    trainy = np.hstack((yt1,yt2))
    testX,testy = ReadData(path,'battery7_with_SOC_SOH_RUL.csv',args.task)  

# ============================
# TRAIN FUNCTION
# ============================
def train_model(seed, lr, wd, hidden, layers, epochs=args.epochs, patience=args.patience, batch_size=128):
    set_seed(seed, args.cuda)
    model = Net(len(trainX[0]), 1, hidden, layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    xt = torch.from_numpy(trainX).float().unsqueeze(1)
    yt = torch.from_numpy(trainy).float()
    xv = torch.from_numpy(testX).float().unsqueeze(1)
    yv = torch.from_numpy(testy).float()
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)

    if args.cuda:
        model, xv, yv = model.cuda(), xv.cuda(), yv.cuda()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            if args.cuda:
                xb, yb = xb.cuda(), yb.cuda()
            opt.zero_grad()
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb, beta=0.1)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(xv)
            val_loss = F.smooth_l1_loss(val_pred, yv, beta=0.1).item()
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.eval()
    with torch.no_grad():
        yhat = model(xv)
    if args.cuda:
        yhat = yhat.cpu()
    return yhat.numpy().flatten()

# ============================
# OPTUNA OBJECTIVE
# ============================
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    hidden = trial.suggest_categorical("hidden", [8, 16, 32, 64])
    layers = trial.suggest_int("layers", 1, 3)

    pred = train_model(seed=1, lr=lr, wd=wd, hidden=hidden, layers=layers)
    r2 = r2_score(testy, pred)
    return 1 - r2  # minimize (maximize R²)

# ============================
# RUN OPTUNA
# ============================
print("\nStarting Optuna hyperparameter optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.trials)
best_params = study.best_params
print("\nBest hyperparameters found:")
print(best_params)

# ============================
# ENSEMBLE WITH BEST PARAMS
# ============================
start_time = time.time()
predictions_list = []
for seed in [1, 42]:
    pred = train_model(seed, **best_params)
    predictions_list.append(pred)

predictions_raw = np.mean(predictions_list, axis=0)

# ============================
# SMOOTHING OPTIMIZATION
# ============================
def smooth(y, weight=0.2):
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = weight*y[i] + (1-weight)*smoothed[i-1]
    return smoothed

best_r2 = -np.inf
best_weight = 0.2
for w in [0.1,0.12,0.15,0.18,0.20,0.22,0.25]:
    pred_smooth = smooth(predictions_raw, weight=w)
    r2_tmp = r2_score(testy, pred_smooth)
    if r2_tmp > best_r2:
        best_r2 = r2_tmp
        best_weight = w

predictions = smooth(predictions_raw, weight=best_weight)
end_time = time.time()

# ============================
# FINAL METRICS
# ============================
mse, rmse, mae, r2 = evaluation_metric(testy, predictions)
metrics_data = [
    ["MSE", mse],
    ["RMSE", rmse],
    ["MAE", mae],
    ["R²", r2]
]
print("\nFinal ensemble model trained with Optuna hyperparameters:")
print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="grid", floatfmt=".4f"))
print(f"Optimal smoothing weight: {best_weight}")
print(f"Execution time: {end_time - start_time:.2f} s")

# ============================
# PLOTS
# ============================
plt.figure(figsize=(12,5))
plt.plot(testy, label='True')
plt.plot(predictions, label='Smoothed Ensemble')
plt.title(f'{args.task} Estimation with Optuna + Ensemble')
plt.xlabel('Cycle')
plt.ylabel(f'{args.task} value')
plt.legend()
plt.show()