import torch
import torch.nn as nn

from .CTA import TripletAttention_MobileNetV2


class MyNet_Shape(nn.Module):
    def __init__(self, ta_out_channels, conv_shape_1_out_channel, shape_lstm_num_layer, shape_lstm_drop, conv_shape_2_out_channel):
        super(MyNet_Shape, self).__init__()
        self.conv_to_ta = nn.Conv2d(in_channels=1, out_channels=ta_out_channels, kernel_size=(1, 1))
        self.ta = TripletAttention_MobileNetV2(input_c=ta_out_channels, output_c=conv_shape_1_out_channel)
        self.convolution_shape_1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_shape_1_out_channel, out_channels=conv_shape_1_out_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ELU(),
            nn.BatchNorm2d(num_features=conv_shape_1_out_channel)
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1))
        self.lstm = nn.LSTM(5, 5, shape_lstm_num_layer, bidirectional=True, batch_first=True, dropout=shape_lstm_drop)  # Bi-LSTM
        self.convolution_shape_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=conv_shape_1_out_channel),
            nn.ELU(),
            nn.Conv2d(in_channels=conv_shape_1_out_channel, out_channels=conv_shape_2_out_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
        )
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.conv_to_ta, self.convolution_shape_1, self.convolution_shape_2]:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)

    def execute(self, shape):
        shape = self.conv_to_ta(shape)
        shape = self.ta(shape)
        shape = self.convolution_shape_1(shape)
        shape = self.max_pooling_1(shape)
        shape = shape.squeeze(2)
        shape, _ = self.lstm(shape)
        shape = shape.unsqueeze(2)
        shape = self.convolution_shape_2(shape)
        shape = self.max_pooling_2(shape)
        return shape




    def forward(self, shape):
        shape = shape.float()
        shape = z_score_normalize(shape)
        shape = shape.unsqueeze(1)
        return self.execute(shape)

def z_score_normalize(data):
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    return (data - mean) / (std + 1e-8)  # 加一个小值以避免除以零



if __name__ == '__main__':
    net = MyNet_Shape()
    x = torch.randn((64, 5, 101))
    net(x)