from dataclasses import dataclass


@dataclass(frozen=True)
class OutfitEncoderConfig:
    image_embedding_dim: int = 64
    text_embedding_dim: int = 64
    transformer_layers: int = 6
    attention_heads: int = 16
    feedforward_dim: int = 512
    dropout: float = 0.1
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    pretrained_image_encoder: bool = True

    @property
    def item_embedding_dim(self) -> int:
        return self.image_embedding_dim + self.text_embedding_dim

    def validate(self) -> None:
        if self.image_embedding_dim <= 0 or self.text_embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        if self.transformer_layers <= 0 or self.attention_heads <= 0:
            raise ValueError("transformer layers and attention heads must be positive")
        if self.item_embedding_dim % self.attention_heads != 0:
            raise ValueError("item embedding dimension must be divisible by attention heads")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward dimension must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
