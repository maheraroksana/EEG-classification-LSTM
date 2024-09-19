Dataset: [Confused student EEG brainwave data](https://www.kaggle.com/datasets/wanghaohan/confused-eeg)
| File(s)    | LSTM + details    | Mean Accuracy    |
|-------------|-------------|-------------|
| `lstm_v3.py`, `train.py` | Simple LSTM with 50 units | 50.77% (+/- 4.15%) |
| `lstm_v4.py`, `train2.py` | Stacked bidirectional LSTM with dropout + learning rate scheduler| Mean Accuracy: 54.69% (+/- 8.90%)|
