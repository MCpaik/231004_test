import torch
import torch.nn as nn

class TextLSTM(nn.Module):
  def __init__(self, n_class, n_hidden, dtype):
    super(TextLSTM, self).__init__()

    self.n_class = n_class
    self.n_hidden = n_hidden
    self.dtype = dtype
    self.lstm = nn.LSTM(dropout=0.3, input_size=self.n_class, hidden_size=self.n_hidden)
    self.W = nn.Parameter(torch.randn([self.n_hidden, self.n_class]).type(self.dtype))
    self.b = nn.Parameter(torch.randn([self.n_class]).type(self.dtype))
    self.Softmax = nn.Softmax(dim=1)

  def forward(self, hidden_and_cell, X):
    X = X.transpose(0, 1)
    outputs, hidden = self.lstm(X, hidden_and_cell)
    outputs = outputs[-1]  # 최종 예측 Hidden Layer
    model = torch.mm(outputs, self.W) + self.b  # 최종 예측 최종 출력 층
    return model