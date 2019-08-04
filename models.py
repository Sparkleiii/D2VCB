import torch.nn as nn
import torch


# Neural Net definition


class BCL_Network(nn.Module):
    def __init__(self):
        super(BCL_Network, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=100,
                      out_channels=64,
                      kernel_size=16,
                      stride=1,
                      padding=8),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=64,
                      out_channels=32,
                      kernel_size=16,
                      stride=1,
                      padding=8),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=32,
                      out_channels=16,
                      kernel_size=16,
                      stride=1,
                      padding=8),
            nn.ReLU(True),
            nn.MaxPool1d(2),
        )

        self.BiLSTM = nn.Sequential(
            nn.LSTM(input_size=25,
                    hidden_size=32,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    bias=True)
        )

        self.Prediction = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # print(input.shape)
        cnn_output = self.cnn(input)
        bilstm_out, _ = self.BiLSTM(cnn_output)
        # print(bilstm_out.shape)
        bilstm_out = bilstm_out[:, -1, :]
        result = self.Prediction(bilstm_out)
        return result
