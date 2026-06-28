from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ImageEncoder(nn.Module):
    """ResNet-18 producing one visual embedding per clothing item."""

    def __init__(self, embedding_dim: int = 64, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(input_features, embedding_dim)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [items, channels, height, width]")
        return self.backbone(images)
