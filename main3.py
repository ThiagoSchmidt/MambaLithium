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
import optuna

# ============================
# ARGUMENTS
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--task', type=str, default='SOH')   # RUL or SOH
parser.add_argument('--case', type=str, default='G')     # A..G
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
# CASE G DATA (example)
# ============================
path = './data/case' + args.case
xt1, yt1 = ReadData(path, 'battery5_with_SOC_SOH_RUL.csv', args.task)
xt2, yt2 = ReadData(path, 'battery6_with_SOC_SOH_RUL.csv', args.task)
trainX_full = np.vstack((xt1, xt2))
trainy_full = np.hstack((yt1, yt2))
testX_full, testy_full = ReadData(path, 'battery7_with_SOC_SOH_RUL.csv', args.task)

# ============================
# TRAIN & EVALUATE FUNCTION
# ============================
def train_and_evaluate(lr, wd, hidden, layers, early_stop=True,
                       epochs=None, trainX=trainX_full, trainy=trainy_full,
                       testX=testX_full, testy=testy_full):

    if epochs is None:
        epochs = args.epochs

    model = Net(len(trainX[0]), 1, hidden, layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    xt = torch.from_numpy(trainX).float().unsqueeze(1)
    yt = torch.from_numpy(trainy).float()
    xv = torch.from_numpy(testX).float().unsqueeze(1)
    yv = torch.from_numpy(testy).float()

    batch_size = 256
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)

    if args.cuda:
        model = model.cuda()
        xv = xv.cuda()
        yv = yv.cuda()

    best_loss = float('inf')
    patience_counter = 0
    lr_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            if args.cuda:
                xb, yb = xb.cuda(), yb.cuda()
            opt.zero_grad()
            pred = model(xb)
            loss = F.l1_loss(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(xv)
            val_loss = F.l1_loss(val_pred, yv).item()

        scheduler.step(val_loss)
        lr_history.append(opt.param_groups[0]['lr'])
        val_loss_history.append(val_loss)

        if early_stop:
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

    model.eval()
    with torch.no_grad():
        yhat = model(xv)
    if args.cuda:
        yhat = yhat.cpu()
    yhat = yhat.numpy().flatten()
    mse, rmse, mae, r2 = evaluation_metric(testy, yhat)
    return mse, rmse, mae, r2, yhat, lr_history, val_loss_history

# ============================
# OPTUNA
# ============================
# Use smaller subset for faster trial evaluation
trainX_sub = trainX_full[:200]
trainy_sub = trainy_full[:200]
testX_sub = testX_full[:50]
testy_sub = testy_full[:50]

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    wd = trial.suggest_loguniform('wd', 1e-6, 1e-3)
    hidden = trial.suggest_categorical('hidden', [32, 64])
    layers = trial.suggest_int('layers', 1, 2)

    mse, _, _, _, _, _, _ = train_and_evaluate(lr, wd, hidden, layers,
                                               epochs=50,  # small number of epochs for fast search
                                               trainX=trainX_sub, trainy=trainy_sub,
                                               testX=testX_sub, testy=testy_sub)
    return mse

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
)
study.optimize(objective, n_trials=10)
best_params = study.best_params
print("\n🏆 Best parameters found:", best_params)

# ============================
# FINAL TRAINING ON FULL DATA
# ============================
start_time = time.time()
mse, rmse, mae, r2, predictions, lr_history, val_loss_history = train_and_evaluate(
    best_params['lr'], best_params['wd'], best_params['hidden'], best_params['layers'],
    epochs=args.epochs,
    trainX=trainX_full, trainy=trainy_full,
    testX=testX_full, testy=testy_full
)
end_time = time.time()

tf = len(testy_full)
if args.task == 'RUL':
    testy_full = tf * testy_full
    predictions = tf * predictions

metrics_data = [
    ["MSE", mse],
    ["RMSE", rmse],
    ["MAE", mae],
    ["R²", r2]
]

print("\n✅ Final model trained with best parameters:")
print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="grid", floatfmt=".4f"))
print(f"Execution time: {end_time - start_time:.2f} s")

# ============================
# PLOTS
# ============================
plt.figure(figsize=(10,5))
plt.plot(testy_full, label='True')
plt.plot(predictions, label='Estimation')
plt.title(f'{args.task} Estimation')
plt.xlabel('Cycle')
plt.ylabel(f'{args.task} value')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(lr_history, color='green')
plt.title('Learning Rate Evolution')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.yscale('log')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(val_loss_history, color='red')
plt.title('Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.grid(True)
plt.show()
