#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 03:22:34 2024

@author: mahera
"""

import torch
import torch.nn as nn

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.fc(out)  
        return out
