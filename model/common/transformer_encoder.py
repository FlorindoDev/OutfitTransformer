"""Factory for task-specific Transformer encoders."""

from torch import nn

from .config import TransformerConfig


def build_transformer_encoder(config: TransformerConfig) -> nn.TransformerEncoder:
    """Build one Transformer encoder from validated task configuration."""
    config.validate()
    layer = nn.TransformerEncoderLayer(
        d_model=config.model_dim,
        nhead=config.attention_heads,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation=_activation(config.activation),
        batch_first=True,
        norm_first=config.norm_first,
    )
    return nn.TransformerEncoder(
        encoder_layer=layer,
        num_layers=config.layers,
        norm=nn.LayerNorm(
            config.model_dim,
            eps=config.layer_norm_epsilon,
        ),
        enable_nested_tensor=False,
    )


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "mish": nn.Mish,
    }
    try:
        return activations[name]()
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error
