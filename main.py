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
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--task', type=str, default='SOH',
                    help='RUL or SOH')
parser.add_argument('--case', type=str, default='A',
                    help='A,B or C')                    

args = parser.parse_args() # Parse command-line arguments
# Check if CUDA is available and set the flag accordingly
args.cuda = args.use_cuda and torch.cuda.is_available() # Function to evaluate the model performance using various metrics

def evaluation_metric(y_test,y_hat): 
    MSE = mean_squared_error(y_test, y_hat) # Calculate Mean Squared Error
    RMSE = MSE**0.5 # Calculate Root Mean Squared Error
    MAE = mean_absolute_error(y_test,y_hat) # Calculate Mean Absolute Error
    R2 = r2_score(y_test,y_hat) # Calculate R-squared score
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2)) # Print the evaluation metrics

def set_seed(seed,cuda): # Function to set the random seed for reproducibility
    np.random.seed(seed) # Set the random seed for NumPy
    torch.manual_seed(seed) # Set the random seed for PyTorch
    if cuda: # If CUDA is available, set the random seed for CUDA
        torch.cuda.manual_seed(seed) # Ensure that the results are reproducible by setting the random seed for both NumPy and PyTorch

set_seed(args.seed,args.cuda) #make sure results are reproducible by setting the random seed for both NumPy and PyTorch

class Net(nn.Module): # Define a neural network model using Mamba
    def __init__(self,in_dim,out_dim): # Initialize the model with input and output dimensions
        super().__init__() # Call the parent class constructor
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer) # Create a configuration for Mamba with specified hidden dimension and number of layers
        self.mamba = nn.Sequential( # Define the model architecture using a sequential container
            nn.Linear(in_dim,args.hidden), # Linear layer to transform input dimension to hidden dimension
            Mamba(self.config), # Mamba layer with the specified configuration
            nn.Linear(args.hidden,out_dim), # Linear layer to transform hidden dimension to output dimension
            nn.Sigmoid() # Sigmoid activation function to output values between 0 and 1
        )
    
    def forward(self,x): # Forward pass of the model
        x = self.mamba(x) # Pass the input through the Mamba layers
        return x.flatten() # Flatten the output to a 1D tensor

def PredictWithData(trainX, trainy, testX): # Function to predict the target variable using the trained model
    clf = Net(len(trainX[0]),1) # Create an instance of the Net class with input dimension equal to the number of features in trainX and output dimension 1
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd) # Create an Adam optimizer for the model parameters with specified learning rate and weight decay
    xt = torch.from_numpy(trainX).float().unsqueeze(0) # Convert trainX to a PyTorch tensor and add a batch dimension
    xv = torch.from_numpy(testX).float().unsqueeze(0) # Convert testX to a PyTorch tensor and add a batch dimension
    yt = torch.from_numpy(trainy).float() # Convert trainy to a PyTorch tensor
    if args.cuda: # If CUDA is available, move the tensors to the GPU
        clf = clf.cuda() # Move the model to the GPU
        xt = xt.cuda() # Move the training data to the GPU
        xv = xv.cuda() # Move the validation data to the GPU
        yt = yt.cuda() # Move the target variable to the GPU
    
    for e in range(args.epochs): # Loop through the specified number of epochs for training
        clf.train() # Set the model to training mode
        z = clf(xt) # Forward pass through the model to get predictions
        loss = F.l1_loss(z,yt) # Calculate the L1 loss between predictions and target variable
        opt.zero_grad() # Zero the gradients of the optimizer
        loss.backward() # Backpropagate the loss to compute gradients
        opt.step() # Update the model parameters using the optimizer
        if e%10 == 0 and e!=0: # Print the loss every 10 epochs
            print('Epoch %d | Lossp: %.4f' % (e, loss.item())) # Print the loss value for the current epoch

    clf.eval() # Set the model to evaluation mode
    mat = clf(xv) # Forward pass through the model to get predictions for the validation data
    if args.cuda: mat = mat.cpu() # If CUDA is used, move the predictions back to the CPU
    yhat = mat.detach().numpy().flatten() # Convert the predictions to a NumPy array and flatten it to 1D
    return yhat # Return the predictions as a NumPy array

def ReadData(path,csv,task): # Function to read data from a CSV file and preprocess it
    f = os.path.join(path,csv) # Load the CSV file
    data = pd.read_csv(f) # Read the data into a DataFrame
    tf = len(data) # Total number of cycles in the data
    y = data[task] # Extract the target variable (RUL or SOH)
    y = y.values # Convert the target variable to a NumPy array
    if args.task == 'RUL': y = y/tf # Normalize RUL by total cycles
    x = data.drop(['RUL','SOH'],axis=1).values # Drop the target variables from the features
    x = scale(x) # Scale the features to have zero mean and unit variance
    return x,y # Return the features and target variable

path = './data/Case'+args.case
if args.case == 'A':
    xt1, yt1 = ReadData(path,'91.csv',args.task) # Read data from the first CSV file
    xt2, yt2 = ReadData(path,'100.csv',args.task) # Read data from the second CSV file
    trainX = np.vstack((xt1,xt2)) # Stack the features vertically
    trainy = np.hstack((yt1,yt2)) # Stack the target variables horizontally
    testX,testy = ReadData(path,'124.csv',args.task) # Read data from the test CSV file
if args.case == 'B':
    xt1, yt1 = ReadData(path,'101.csv',args.task)
    xt2, yt2 = ReadData(path,'108.csv',args.task)
    xt3, yt3 = ReadData(path,'120.csv',args.task)
    trainX = np.vstack((xt1,xt2,xt3))
    trainy = np.hstack((yt1,yt2,yt3))
    testX,testy = ReadData(path,'116.csv',args.task)

if args.case == 'C':
    xt1, yt1 = ReadData(path,'15.csv',args.task)
    xt2, yt2 = ReadData(path,'16.csv',args.task)
    trainX = np.vstack((xt1,xt2))
    trainy = np.hstack((yt1,yt2))
    testX,testy = ReadData(path,'17.csv',args.task)

predictions = PredictWithData(trainX, trainy, testX) # Predict the target variable for the test data
tf = len(testy) # Total number of cycles in the test data
if args.task == 'RUL': # Normalize the predictions if the task is RUL
    testy = tf*testy 
    predictions = tf*predictions 

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.4f} seconds")

print('MSE RMSE MAE R2') # Print the evaluation metrics
evaluation_metric(testy, predictions) # Evaluate the model performance using various metrics
plt.figure()
plt.plot(testy, label='True') # Plot the true values
plt.plot(predictions, label='Estimation') # Plot the predicted values
plt.title(args.task+' Estimation') # Set the title of the plot
plt.xlabel('Cycle') # Set the x-axis label
plt.ylabel(args.task+' value') # Set the y-axis label
plt.legend() # Add a legend to the plot
plt.show() # Show the plot with true and predicted values



# The code reads battery data, trains a neural network model using Mamba, and evaluates its performance on RUL or SOH prediction tasks. It uses PyTorch for model training and evaluation, and Matplotlib for visualization of results.
# The model is trained using Adam optimizer and L1 loss function, and the results are visualized with plots showing true vs predicted values. The code is modular, allowing for easy adjustments of parameters like learning rate, weight decay, and number of epochs through command-line arguments.
# The evaluation metrics include MSE, RMSE, MAE, and R2 score, providing a comprehensive assessment of the model's performance. The code is designed to be flexible, allowing for different battery datasets and configurations based on user input.