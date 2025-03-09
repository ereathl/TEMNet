import torch
import torch.nn as nn

from .EMA import EMA
from .MyNet_sequence import MyNet_Sequence
from .MyNet_shape import MyNet_Shape


class MyNet(nn.Module):
    def __init__(self, seq_lstm_num_layer = 2, lstm_hidder_size = 64, seq_lstm_drop = 0.2, dropout = 0.4, ta_out_channels = 8, shape_lstm_num_layer = 16, shape_lstm_drop = 0.2, factor = 4):
        super(MyNet, self).__init__()
        self.sequence = MyNet_Sequence(seq_lstm_num_layer = seq_lstm_num_layer, lstm_hidder_size = lstm_hidder_size, seq_lstm_drop = seq_lstm_drop, dropout = dropout)
        self.shape = MyNet_Shape(ta_out_channels = ta_out_channels, conv_shape_1_out_channel=lstm_hidder_size*4, shape_lstm_num_layer=shape_lstm_num_layer, shape_lstm_drop=shape_lstm_drop, conv_shape_2_out_channel=lstm_hidder_size*4)
        self.ema = EMA(channels = lstm_hidder_size*4, factor=lstm_hidder_size * factor)
        self.convolution_1 = nn.Sequential(
            nn.Conv2d(in_channels=lstm_hidder_size * 4, out_channels=lstm_hidder_size * 8, kernel_size=(1, 5),
                      stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(num_features=lstm_hidder_size * 8)
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1))
        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=lstm_hidder_size*8, out_features=lstm_hidder_size*3),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(lstm_hidder_size*3),
            nn.Dropout(0.2),
            nn.Linear(in_features=lstm_hidder_size*3, out_features=1),
            nn.Sigmoid()
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.ema, self.convolution_1, self.output]:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)


    def execute(self, seq, shape):
        seq = self.sequence(seq)
        shape = self.shape(shape)
        x = torch.concat([seq, shape], dim=3)
        x = self.ema(x)
        x = self.convolution_1(x)
        x = self.max_pooling_1(x)
        x = self.output(x)
        return x


    def forward(self, seq, shape):
        return self.execute(seq, shape)


if __name__ == '__main__':
    net = MyNet()
    sequence = torch.randn((64, 12, 99))
    shape = torch.randn((64, 5, 101))
    net(sequence, shape)