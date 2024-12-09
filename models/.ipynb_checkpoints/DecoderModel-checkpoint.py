import torch.nn as nn
from easydict import EasyDict
from torch.nn import Dropout, ModuleList


class DecoderBaseModel(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(DecoderBaseModel, self).__init__()

        # useless
        # self.input_size = input_size
        # self.hidden_size = hidden_size

        self.config = EasyDict(
            {
                "input_size": input_size,  # input_size
                "hidden_size": hidden_size,
                "is_transformer": False,
            }
        )


class DecoderBaseRNN(DecoderBaseModel):
    def __init__(self, rnn, input_size, hidden_size, num_layers, bidirectional=False):
        super(DecoderBaseRNN, self).__init__(input_size, hidden_size)
        self.model = rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if bidirectional:
            self.config["hidden_size"] *= 2
        self.config["num_layers"] = num_layers
        self.config["bidirectional"] = bidirectional

    def forward(self, x):
        output, _ = self.model(x)

        return output


class DecoderTransformer(DecoderBaseModel):
    def __init__(self, input_size, nhead, num_encoder_layers, num_decoder_layers):
        super(DecoderTransformer, self).__init__(input_size)
        self.model = nn.Transformer(
            d_model=input_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )

        self.config["num_encoder_layers"] = num_encoder_layers
        self.config["num_decoder_layers"] = num_decoder_layers
        self.config["is_transformer"] = True

    def forward(self, src):
        output = self.model(src, src)

        return output


class Block(nn.Module):
    """
    TODO, maybe
    """

    def __init__(self, baseNN, dropout):
        super(Block, self).__init__()
        self.network = baseNN
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.network(x)
        x = self.dropout(x)

        return x
