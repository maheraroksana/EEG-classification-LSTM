#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:15:17 2024

@author: mahera
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size 
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)        
        self.batch_norm = nn.BatchNorm1d(2 * hidden_size) 
        
        self.fc = nn.Linear(2 * hidden_size, output_size)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        batch_size, seq_len, _ = lstm_out.size()
        
        lstm_out = lstm_out.contiguous().view(-1, 2 * self.hidden_size)
        norm_out = self.batch_norm(lstm_out)
        norm_out = norm_out.view(batch_size, seq_len, 2 * self.hidden_size)

        out = self.fc(norm_out)
        out = self.sigmoid(out)     
        return out
