import torch.nn as nn
from easydict import EasyDict
from torch.nn import Dropout, ModuleList


class DecoderBaseModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderBaseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.config = EasyDict(
            {
            "input_size": input_size,  # input_size
            "hidden_size": hidden_size,
        })


class DecoderLSTM(DecoderBaseModel):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(DecoderLSTM, self).__init__(input_size, hidden_size)
        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if bidirectional:
            self.config["hidden_size"] *= 2
        self.config["num_layers"] = num_layers

    def forward(self, x):
        output, _ = self.model(x)

        return output


class Block(nn.Module):
    """
    TODO
    """

    def __init__(self, baseNN, dropout):
        super(Block, self).__init__()
        self.network = baseNN
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.network(x)
        x = self.dropout(x)

        return x
