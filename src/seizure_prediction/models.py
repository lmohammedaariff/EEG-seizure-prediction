from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.autograd import Function

from .config import (
    MODEL_VARIANTS,
    MULTIHORIZON_ABLATION_VARIANTS,
    MULTIHORIZON_BASELINE_VARIANTS,
)


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(value: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(value, strength)


def _conv_block(
    in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
        nn.MaxPool1d(2),
    )


class MultiDomainEEGNet(nn.Module):
    """Temporal, PSD, and STFT branches with gated fusion and a domain head."""

    def __init__(
        self,
        n_channels: int = 18,
        n_subjects: int = 0,
        sampling_rate: int = 128,
        variant: str = "subject_invariant",
        embedding_size: int = 64,
        n_horizons: int = 0,
    ) -> None:
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model variant: {variant}")
        if variant in {"cnn", "lstm", "cnn_lstm"}:
            raise ValueError("Use BaselineEEGNet for CNN/LSTM baseline variants.")
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.variant = variant
        self.use_spectral = variant in {
            "temporal_spectral",
            "multi_domain",
            "subject_invariant",
            "multi_horizon_temporal_spectral",
            "multi_horizon_multi_domain",
            "multi_horizon",
        }
        self.use_time_frequency = variant in {
            "multi_domain",
            "subject_invariant",
            "multi_horizon_multi_domain",
            "multi_horizon",
        }
        self.use_subject_head = variant in {"subject_invariant", "multi_horizon"} and n_subjects > 1
        self.use_horizon_heads = (
            variant in MULTIHORIZON_ABLATION_VARIANTS and n_horizons > 0
        )

        self.temporal = nn.Sequential(
            _conv_block(n_channels, 32, 7),
            _conv_block(32, 64, 5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_size),
            nn.GELU(),
        )
        self.spectral = nn.Sequential(
            nn.Linear(n_channels * 5, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_size),
            nn.GELU(),
        )
        self.tf_conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embedding_size),
            nn.GELU(),
        )
        domains = 1 + int(self.use_spectral) + int(self.use_time_frequency)
        self.attention = nn.Linear(embedding_size, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Dropout(0.25),
            nn.Linear(embedding_size, 3),
        )
        self.prediction_classifier = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Dropout(0.25),
            nn.Linear(embedding_size, 2),
        )
        self.horizon_classifiers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(embedding_size),
                nn.Dropout(0.25),
                nn.Linear(embedding_size, 2),
            )
            for _ in range(n_horizons)
        ) if self.use_horizon_heads else nn.ModuleList()
        self.subject_classifier = (
            nn.Sequential(
                nn.Linear(embedding_size, 64),
                nn.GELU(),
                nn.Linear(64, n_subjects),
            )
            if self.use_subject_head
            else None
        )
        self.n_domains = domains

    def _spectral_features(self, signals: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(signals, dim=-1)
        power = spectrum.abs().square() / signals.shape[-1]
        frequencies = torch.fft.rfftfreq(
            signals.shape[-1],
            d=1.0 / self.sampling_rate,
            device=signals.device,
        )
        edges = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
        bands = []
        for low, high in edges:
            selected = (frequencies >= low) & (frequencies < min(high, self.sampling_rate / 2))
            if not torch.any(selected):
                bands.append(torch.zeros_like(power[..., 0]))
            else:
                bands.append(power[..., selected].mean(dim=-1))
        features = torch.stack(bands, dim=-1)
        return torch.log1p(features).flatten(start_dim=1)

    def _time_frequency_features(self, signals: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = signals.shape
        n_fft = min(64, samples)
        if n_fft < 16:
            raise ValueError("EEG windows need at least 16 samples for the STFT branch.")
        flattened = signals.reshape(batch * channels, samples)
        stft = torch.stft(
            flattened,
            n_fft=n_fft,
            hop_length=max(1, n_fft // 4),
            win_length=n_fft,
            window=torch.hann_window(n_fft, device=signals.device),
            return_complex=True,
            center=True,
        )
        magnitude = torch.log1p(stft.abs()).reshape(batch, channels, stft.shape[-2], stft.shape[-1])
        return self.tf_conv(magnitude)

    def forward(
        self, signals: torch.Tensor, reversal_strength: float = 0.0
    ) -> dict[str, torch.Tensor | None]:
        if signals.ndim != 3 or signals.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected EEG tensor [batch, {self.n_channels}, samples], "
                f"got {tuple(signals.shape)}."
            )
        embeddings = [self.temporal(signals)]
        names = ["temporal"]
        if self.use_spectral:
            embeddings.append(self.spectral(self._spectral_features(signals)))
            names.append("spectral")
        if self.use_time_frequency:
            embeddings.append(self._time_frequency_features(signals))
            names.append("time_frequency")

        stacked = torch.stack(embeddings, dim=1)
        weights = torch.softmax(self.attention(stacked).squeeze(-1), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        subject_logits = None
        if self.subject_classifier is not None:
            subject_logits = self.subject_classifier(
                gradient_reverse(fused, reversal_strength)
            )
        horizon_logits = (
            torch.stack([head(fused) for head in self.horizon_classifiers], dim=1)
            if self.horizon_classifiers
            else None
        )
        return {
            "state_logits": self.classifier(fused),
            "prediction_logits": self.prediction_classifier(fused),
            "horizon_logits": horizon_logits,
            "subject_logits": subject_logits,
            "attention": weights,
            "domain_names": names,
            "features": fused,
        }


class BaselineEEGNet(nn.Module):
    """Conventional CNN, LSTM, or CNN-LSTM state-classification baselines."""

    def __init__(self, n_channels: int, variant: str, n_horizons: int = 0) -> None:
        super().__init__()
        if variant not in {"cnn", "lstm", "cnn_lstm"}:
            raise ValueError(f"Not a baseline variant: {variant}")
        self.variant = variant
        if variant == "cnn":
            self.encoder = nn.Sequential(
                _conv_block(n_channels, 32, 7),
                _conv_block(32, 64, 5),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.output_size = 64
        elif variant == "lstm":
            self.encoder = nn.LSTM(
                input_size=n_channels,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.output_size = 128
        else:
            self.convolution = nn.Sequential(
                _conv_block(n_channels, 32, 7),
                _conv_block(32, 64, 5),
            )
            self.encoder = nn.LSTM(
                input_size=64,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.output_size = 128
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.output_size),
            nn.Dropout(0.25),
            nn.Linear(self.output_size, 3),
        )
        self.prediction_classifier = nn.Sequential(
            nn.LayerNorm(self.output_size),
            nn.Dropout(0.25),
            nn.Linear(self.output_size, 2),
        )
        self.horizon_classifiers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.output_size),
                nn.Dropout(0.25),
                nn.Linear(self.output_size, 2),
            )
            for _ in range(n_horizons)
        )

    def forward(
        self, signals: torch.Tensor, reversal_strength: float = 0.0
    ) -> dict[str, torch.Tensor | None]:
        if signals.ndim != 3:
            raise ValueError("Expected EEG tensor [batch, channels, samples].")
        if self.variant == "cnn":
            features = self.encoder(signals)
        elif self.variant == "lstm":
            sequence = signals.transpose(1, 2)[:, ::4, :]
            _, (hidden, _) = self.encoder(sequence)
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            sequence = self.convolution(signals).transpose(1, 2)
            _, (hidden, _) = self.encoder(sequence)
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        horizon_logits = (
            torch.stack(
                [head(features) for head in self.horizon_classifiers], dim=1
            )
            if self.horizon_classifiers
            else None
        )
        return {
            "state_logits": self.classifier(features),
            "prediction_logits": self.prediction_classifier(features),
            "horizon_logits": horizon_logits,
            "subject_logits": None,
            "attention": None,
            "domain_names": [],
            "features": features,
        }


def build_model(
    n_channels: int,
    n_subjects: int,
    sampling_rate: int,
    variant: str,
    horizons_minutes: tuple[int, ...] = (),
) -> nn.Module:
    if variant in MULTIHORIZON_BASELINE_VARIANTS:
        baseline_variant = variant.removeprefix("multi_horizon_")
        return BaselineEEGNet(
            n_channels=n_channels,
            variant=baseline_variant,
            n_horizons=len(horizons_minutes),
        )
    if variant in MULTIHORIZON_ABLATION_VARIANTS:
        return MultiDomainEEGNet(
            n_channels=n_channels,
            n_subjects=n_subjects,
            sampling_rate=sampling_rate,
            variant=variant,
            n_horizons=len(horizons_minutes),
        )
    if variant in {"cnn", "lstm", "cnn_lstm"}:
        return BaselineEEGNet(n_channels, variant)
    return MultiDomainEEGNet(
        n_channels=n_channels,
        n_subjects=n_subjects,
        sampling_rate=sampling_rate,
        variant=variant,
    )
