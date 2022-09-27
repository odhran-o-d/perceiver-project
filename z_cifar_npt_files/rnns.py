
import torch
from torch import nn, einsum

class LSTM(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_channels, latent_dim,
                 depth, ff_dropout, num_classes,
                 **kwargs):
        super().__init__()
        self.rnn = nn.LSTM(
            batch_first=True,
            input_size=input_channels,
            num_layers=depth,
            hidden_size=latent_dim,
            dropout=ff_dropout
            )
        self.to_logits = nn.Linear(
            latent_dim, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        final_step = out[:, -1, :]
        return self.to_logits(final_step)


class GRU(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_channels, latent_dim,
                 depth, ff_dropout, num_classes,
                 **kwargs):
        super().__init__()
        self.rnn = nn.GRU(
            batch_first=True,
            input_size=input_channels,
            num_layers=depth,
            hidden_size=latent_dim,
            dropout=ff_dropout
            )
        self.to_logits = nn.Linear(
            latent_dim, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        final_step = out[:, -1, :]
        return self.to_logits(final_step)


class RNN(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_channels, latent_dim,
                 depth, ff_dropout, num_classes,
                 **kwargs):
        super().__init__()
        self.rnn = nn.RNN(
            batch_first=True,
            input_size=input_channels,
            num_layers=depth,
            hidden_size=latent_dim,
            dropout=ff_dropout
            )
        self.to_logits = nn.Linear(
            latent_dim, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        final_step = out[:, -1, :]
        return self.to_logits(final_step)