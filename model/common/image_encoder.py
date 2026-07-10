from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from .config import IMAGE_FINE_TUNE_MODES, ImageFineTuneMode


class ImageEncoder(nn.Module):
    """ResNet-18 producing one visual embedding per clothing item."""

    def __init__(
        self,
        embedding_dim: int = 64,
        pretrained: bool = True,
        fine_tune_mode: ImageFineTuneMode = "fc_only",
    ) -> None:
        super().__init__()
        if fine_tune_mode not in IMAGE_FINE_TUNE_MODES:
            raise ValueError(
                f"fine_tune_mode must be one of {IMAGE_FINE_TUNE_MODES}"
            )

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(input_features, embedding_dim)
        self.fine_tune_mode = fine_tune_mode
        self._frozen_feature_modules = self._configure_fine_tuning()
        self.train(self.training)

    def train(self, mode: bool = True) -> "ImageEncoder":
        super().train(mode)
        if mode:
            for module in self._frozen_feature_modules:
                module.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [items, channels, height, width]")
        return self.backbone(images)

    def _configure_fine_tuning(self) -> tuple[nn.Module, ...]:
        if self.fine_tune_mode == "full":
            self.backbone.requires_grad_(True)
            return ()

        self.backbone.requires_grad_(False)
        self.backbone.fc.requires_grad_(True)

        frozen_modules: tuple[nn.Module, ...] = (
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
        )
        if self.fine_tune_mode == "fc_and_layer4":
            self.backbone.layer4.requires_grad_(True)
            return frozen_modules
        return (*frozen_modules, self.backbone.layer4)
