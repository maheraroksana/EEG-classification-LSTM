#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:12:04 2024

@author: mahera
"""

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from lstm_v3 import *


def load_dataset():
    file = './archive/EEG_data.csv'
    data = pd.read_csv(file)

    #drop cols
    col = ['SubjectID', 'VideoID', 'user-definedlabeln']
    features = data.drop(col, axis=1)

    #grp by sub and vid -> dataframe
    grouped = data.groupby(['SubjectID', 'VideoID'])[features.columns[0]].count()
    df = pd.DataFrame(columns=data.columns)

    #append all rows (112) for each combo of sub and vid
    dfs = []
    for i in range(10):
        for j in range(10):
            grouped = data.loc[(data['SubjectID'] == i) & (data['VideoID'] == j)]
            dfs.append(grouped.iloc[:112])
            
    df = pd.concat(dfs, ignore_index=True)

    col = ['SubjectID', 'VideoID', 'predefinedlabel']
    features = df.drop(col, axis=1)
    selected = np.array(features)
    X_input = selected[:, 0:11]  # 11 cols -> features
    Y_input = selected[:, 11]    # 12th cols -> label
    
    return X_input, Y_input

def train(args):
    
    #load dataset
    X_input, Y_input = load_dataset()


    Y_input = np.where(Y_input > 0.5, 1, 0) 

    #reshape
    X_input = np.reshape(X_input, (-1, 112, 11))
    Y_input = np.reshape(Y_input, (-1, 112, 1))

    #CV
    kfold = KFold(n_splits=args.n_splits, shuffle=False)
    cross_val_results = []

    #kfold
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_input, Y_input)):
        print(f"Training fold {fold+1}/{args.n_splits}")
        
        # Convert to torch tensors
        X_train = torch.tensor(X_input[train_idx], dtype=torch.float32)
        Y_train = torch.tensor(Y_input[train_idx], dtype=torch.float32)
        X_test = torch.tensor(X_input[test_idx], dtype=torch.float32)
        Y_test = torch.tensor(Y_input[test_idx], dtype=torch.float32)
        
        #initialize model, loss function, and optimizer
        input_size = 11
        hidden_size = args.hidden_size
        output_size = 1  
        
        model = LSTMModel(input_size, hidden_size, output_size)
        criterion = nn.BCELoss()  
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        train_losses = []
        validation_losses = []

        #Main trianing loop
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train)
            loss = criterion(outputs[:, -1, :].squeeze(), Y_train[:, -1, :].squeeze())  
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            #Val loss
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs[:, -1, :].squeeze(), Y_test[:, -1, :].squeeze())
                validation_losses.append(val_loss.item())

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve for Fold {fold+1}')
        plt.legend()
        plt.savefig(f'loss_curve_fold_{fold+1}.png')
        plt.close()  
        
        # Final preds
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            predictions = predictions.round()  
            correct = (predictions == Y_test).sum().item()
            accuracy = correct / np.prod(Y_test.shape) * 100
            print(f'Accuracy: {accuracy:.2f}%')
            cross_val_results.append(accuracy)
            validation_losses.append(val_loss.item())

    # Print CV results
    print(f'Mean Accuracy: {np.mean(cross_val_results):.2f}% (+/- {np.std(cross_val_results):.2f}%)')
    torch.save(model.state_dict(), 'model_weights.pth')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--n_splits', type=int, default=10)

    args = parser.parse_args()

    # Train 
    train(args)
