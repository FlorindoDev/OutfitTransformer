from collections.abc import Callable

from PIL import Image
from torch import Tensor
from torchvision import transforms

ImageTransform = Callable[[Image.Image], Tensor]


def build_image_transform(image_size: int = 224) -> ImageTransform:
    pipeline = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    def transform(image: Image.Image) -> Tensor:
        transformed = pipeline(image)
        if not isinstance(transformed, Tensor):
            raise TypeError("image transform must return a Tensor")
        return transformed

    return transform
