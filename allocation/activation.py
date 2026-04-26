from torch import nn

_activations = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
}

def build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in _activations:
        supported = ", ".join(sorted(_activations))
        raise ValueError(f"Unsupported activation: {name}. Supported: {supported}")
    return _activations[key]()