from torch import Tensor, nn


class TaskMLP(nn.Module):
    """Small task-specific projection head used after a transformer token."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input and output dimensions must be positive")
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("hidden dimension must be positive")

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)
