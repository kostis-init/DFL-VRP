from torch import nn


class BaseCostModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class CostPredictorLinear(BaseCostModel):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = x.view(-1)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.activation(out)
        return out


class EncoderDecoder(BaseCostModel):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1)
        x = self.encoder(x)
        x = self.relu(x)
        x = self.decoder(x)
        x = self.activation(x)
        return x
