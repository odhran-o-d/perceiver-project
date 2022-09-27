"""Learn uncertainties per feature."""
import torch
import torch.nn as nn
import numpy as np


class MultitaskUncertainty(nn.Module):
    """Learn uncertainties per feature.
    See
        > Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
        > Geometry and Semantics, Kendall et al., 2018
        > https://arxiv.org/pdf/1705.07115.pdf.

    For each feature column, we learn a parameter which corresponds to its
    uncertainty.
    For continuous features, this can be interpreted as the standard deviation
    of a Gaussian likelihood.
    For categorical features, this is the softmax temperature.

    We log-parametrise the values to ease learning and
    assure non-negativity.

    MultitaskUncertainty is instantiated as a member of
    `ttnn.models.ttnn.TTNNModel` and is used for loss computation in
    `ttnn.loss.Loss`.

    Note: We no longer use the approximation Eq. 10 of the paper and instead
    directly use softmax temperatures for discrete variables.
    """

    def __init__(self, num_features):
        """
        Args:
            num_features (int):
                Number of features for which to learn uncertainties.
        """
        super().__init__()

        # Init log-variance as 0 --> variance = exp(log_var) = 1
        self.log_variances = nn.Parameter(
            torch.zeros(num_features), requires_grad=True)

    @property
    def variances(self):
        return torch.exp(self.log_variances)

    @property
    def log_stds_with_penalty(self):
        # log(sigma) + ...
        return 0.5 * self.log_variances + 0.5 * np.log(2 * np.pi)

    @property
    def stds(self):
        return torch.exp(0.5 * self.log_variances)

    def factor(self, col):
        """Return 1 / (2 * sigma ** 2) for continuous."""
        return (
            1 / (2 * self.variances[col]))

    def temp(self, col):
        """Return exp(log temperature)."""
        return self.variances[col]

    def log(self):
        return self.variances.detach().cpu().numpy()
