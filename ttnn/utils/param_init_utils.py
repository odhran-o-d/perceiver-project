import math

import torch
import torch.nn as nn


def xavier_normal_identity_(tensor, gain=1.):
    # type: (Tensor, float, float) -> Tensor
    r"""Initialization as described in
    `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(loc, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    This variant uses Glorot initialization centered at the identity.
    In other words, we perform Glorot with location 1 for the main diagonal
    of the tensor (

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> xavier_normal_loc_(w, 1)
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    with torch.no_grad():
        return torch.add(
            input=nn.init.eye_(torch.empty(*tensor.shape)),
            other=torch.empty(*tensor.shape).normal_(0, std),
            out=tensor)

def xavier_normal_loc_(tensor, loc, gain=1.):
    # type: (Tensor, float, float) -> Tensor
    r"""Initialization as described in
    `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(loc, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        loc: the location of the normal
            (e.g. may prefer 1 for a multiplicative weight)
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> xavier_normal_loc_(w, 1)
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return nn.init._no_grad_normal_(tensor, loc, std)


def xavier_uniform_loc_(tensor, loc, gain=1.):
    # type: (Tensor, float, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(loc - a, loc + a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> xavier_uniform_loc_(w, loc=1, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return nn.init._no_grad_uniform_(tensor, loc - a, loc + a)
