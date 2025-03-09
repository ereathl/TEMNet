import torch
import torch.nn as nn

from .Residuleblk import resnet_block


class MyNet_Sequence(nn.Module):
    def __init__(self, seq_lstm_num_layer, lstm_hidder_size, seq_lstm_drop, dropout):
        super(MyNet_Sequence, self).__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=lstm_hidder_size, num_layers=seq_lstm_num_layer, bidirectional=True, batch_first=True, dropout=seq_lstm_drop)

        self.res1 = nn.Sequential(*resnet_block(lstm_hidder_size * 2, lstm_hidder_size * 2, 2, first_block=True))
        self.res2 = nn.Sequential(*resnet_block(lstm_hidder_size * 2, lstm_hidder_size * 4, 2))
        self.convolution_seq_1 = nn.Sequential(
            nn.BatchNorm2d(num_features=lstm_hidder_size * 4),
            nn.ELU(),
            nn.Conv2d(in_channels=lstm_hidder_size * 4, out_channels=lstm_hidder_size * 4, kernel_size=(1, 3), stride=(1, 1))
        )
        self.drop = nn.Dropout(dropout)
        self.convolution_seq_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=lstm_hidder_size * 4),
            nn.ELU(),
            nn.Conv2d(in_channels=lstm_hidder_size * 4, out_channels=lstm_hidder_size * 4, kernel_size=(1, 3), stride=(1, 1))
        )
        self.max_pooling_seq1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.res1, self.res2, self.convolution_seq_1, self.convolution_seq_2]:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)

    def execute(self, seq):
        seq = seq.float()

        seq, _ = self.lstm(seq.permute(0, 2, 1))
        seq = seq.permute(0, 2, 1)
        seq = seq.unsqueeze(2)
        seq = self.res1(seq)
        seq = self.res2(seq)
        seq = self.convolution_seq_1(seq)
        seq = self.drop(seq)
        seq = self.convolution_seq_2(seq)
        seq = self.max_pooling_seq1(seq)
        return seq

    def forward(self, seq):
        return self.execute(seq)

if __name__ == '__main__':
    net = MyNet_Sequence()
    x = torch.randn((64, 12, 99))
    net(x)