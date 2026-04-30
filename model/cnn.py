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


class MediumAccelCNN(nn.Module):
    """Higher-capacity Conv1D benchmark model with dropout regularization."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualTemporalBlock(nn.Module):
    """Residual dilated temporal block for HAR sequence modeling."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TemporalAttentionPool(nn.Module):
    """Lightweight attention pooling over time."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class MultiScaleHARNet(nn.Module):
    """Paper-informed HAR model with multi-scale temporal features.

    The architecture follows common HAR strategies from the papers in
    ``papers/``: parallel temporal kernels for multi-scale events, residual
    dilated temporal blocks for sequence context, and attention pooling so short
    events can influence the final class decision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        branch_channels = 24
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        32, branch_channels, kernel_size=kernel, padding=kernel // 2
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(),
                )
                for kernel in (3, 5, 9)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * 3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(64, dilation=1, dropout=dropout),
            ResidualTemporalBlock(64, dilation=2, dropout=dropout),
        )
        self.pool = TemporalAttentionPool(64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)
        self.eventness = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fuse(x)
        x = self.temporal(x)
        x = self.pool(x)
        x = self.dropout(x)
        return self.classifier(x), self.eventness(x)


class FeatureFusionHARNet(nn.Module):
    """CNN fused with differentiable HAR statistical features.

    The real-time HAR paper emphasizes explicit feature extraction in addition
    to neural inference. This model keeps a temporal CNN branch, but also feeds
    per-channel summary features into the classifier: mean, standard deviation,
    min, max, range, RMS, absolute mean, and first-to-last delta.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=5, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        stat_features = in_channels * 8
        self.stats = nn.Sequential(
            nn.Linear(stat_features, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 2 + 32, 80),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(80, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(x)
        pooled = torch.cat(
            [
                torch.mean(temporal, dim=-1),
                torch.amax(temporal, dim=-1),
            ],
            dim=1,
        )
        return self.classifier(torch.cat([pooled, self.stats(window_stats(x))], dim=1))


class ConvGRUHARNet(nn.Module):
    """Compact CNN-GRU HAR model for temporal event structure."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=48,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(96, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x).transpose(1, 2)
        sequence, _ = self.gru(x)
        weights = torch.softmax(self.attention(sequence), dim=1)
        pooled = torch.sum(sequence * weights, dim=1)
        return self.classifier(pooled)


class StatsMLPHARNet(nn.Module):
    """Explicit HAR feature model using window and derivative statistics."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 12, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(extended_window_stats(x))


def window_stats(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=-1)
    std = torch.std(x, dim=-1, unbiased=False)
    minimum = torch.amin(x, dim=-1)
    maximum = torch.amax(x, dim=-1)
    value_range = maximum - minimum
    rms = torch.sqrt(torch.mean(x * x, dim=-1) + 1e-6)
    abs_mean = torch.mean(torch.abs(x), dim=-1)
    delta = x[:, :, -1] - x[:, :, 0]
    return torch.cat(
        [mean, std, minimum, maximum, value_range, rms, abs_mean, delta],
        dim=1,
    )


def extended_window_stats(x: torch.Tensor) -> torch.Tensor:
    diff = x[:, :, 1:] - x[:, :, :-1]
    diff_mean = torch.mean(diff, dim=-1)
    diff_std = torch.std(diff, dim=-1, unbiased=False)
    diff_rms = torch.sqrt(torch.mean(diff * diff, dim=-1) + 1e-6)
    diff_abs_max = torch.amax(torch.abs(diff), dim=-1)
    return torch.cat(
        [window_stats(x), diff_mean, diff_std, diff_rms, diff_abs_max],
        dim=1,
    )


def build_model(
    name: str,
    in_channels: int,
    num_classes: int,
    dropout: float = 0.2,
) -> nn.Module:
    if name == "small":
        return SmallAccelCNN(
            in_channels=in_channels, num_classes=num_classes, dropout=0.0
        )
    if name == "medium":
        return MediumAccelCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    if name == "multiscale":
        return MultiScaleHARNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    if name == "featurefusion":
        return FeatureFusionHARNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    if name == "convgru":
        return ConvGRUHARNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    if name == "statsmlp":
        return StatsMLPHARNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())
