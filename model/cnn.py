"""PyTorch baseline 1D CNN for accelerometer event classification."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI import guard
    raise ModuleNotFoundError(
        "The baseline CNN requires PyTorch. Install project dependencies with "
        "`uv sync` or install `torch` in the active environment."
    ) from exc


class SmallAccelCNN(nn.Module):
    """Small Conv1D baseline from docs/pipeline.md.

    Input shape is ``[batch, channels, samples]``. For the E0 baseline that is
    ``[B, 3, 200]`` for ax/ay/az at 100 Hz over a 2-second window.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(64, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())
