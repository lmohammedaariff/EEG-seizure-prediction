from __future__ import annotations
import torch
from torch import nn
from torch.autograd import Function


class _Reverse(Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def grad_reverse(x, strength):
    return _Reverse.apply(x, strength)


class TemporalBranch(nn.Module):
    def __init__(self, channels, width, architecture="cnn"):
        super().__init__()
        self.architecture = architecture
        self.cnn = nn.Sequential(nn.Conv1d(channels, width, 7, padding=3), nn.BatchNorm1d(width), nn.GELU(),
                                 nn.MaxPool1d(2), nn.Conv1d(width, width * 2, 5, padding=2), nn.GELU())
        self.lstm = nn.LSTM(width * 2 if architecture == "cnn_lstm" else channels, width,
                            batch_first=True, bidirectional=True)
        self.project = nn.Linear(width * 2, width * 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        if self.architecture in ("lstm", "cnn_lstm"):
            seq = self.cnn(x).transpose(1, 2) if self.architecture == "cnn_lstm" else x.transpose(1, 2)
            z, _ = self.lstm(seq)
            return self.project(z.mean(dim=1))
        return self.project(self.pool(self.cnn(x)).squeeze(-1))


class SpectralBranch(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(channels * 5, width * 2), nn.GELU(),
                                 nn.Dropout(.1), nn.Linear(width * 2, width * 2), nn.GELU())
    def forward(self, x): return self.net(x)


class TimeFrequencyBranch(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.GELU(),
                                 nn.MaxPool2d(2), nn.Conv2d(width, width * 2, 3, padding=1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, width * 2))
    def forward(self, x): return self.net(x)


class MultiDomainNet(nn.Module):
    """Three-state head plus primary preictal-vs-non-preictal head."""
    def __init__(self, channels=18, width=32, dropout=.25, n_subjects=1, fusion="attention",
                 domain_adversarial=True, active_domains=None, temporal_architecture="cnn"):
        super().__init__()
        dim = width * 2
        self.temporal = TemporalBranch(channels, width, temporal_architecture)
        self.spectral = SpectralBranch(channels, width)
        self.time_frequency = TimeFrequencyBranch(channels, width)
        self.fusion = fusion
        self.active_domains = active_domains or ("temporal", "spectral", "time_frequency")
        self.attention = nn.Sequential(nn.Linear(dim, width), nn.Tanh(), nn.Linear(width, 1))
        self.combine = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.state_head = nn.Linear(dim, 3)
        self.predict_head = nn.Linear(dim, 1)
        self.subject_head = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, max(1, n_subjects)))
        self.domain_adversarial = domain_adversarial

    def forward(self, temporal, spectral, time_frequency, grl_strength=0.0):
        encoders = {"temporal": (self.temporal, temporal), "spectral": (self.spectral, spectral),
                    "time_frequency": (self.time_frequency, time_frequency)}
        encoded = {name: encoders[name][0](encoders[name][1]) for name in self.active_domains}
        branches = torch.stack([encoded[name] for name in self.active_domains], dim=1)
        if self.fusion == "attention":
            weights = torch.softmax(self.attention(branches), dim=1)
            fused = (branches * weights).sum(dim=1)
        else:
            weights = torch.full((branches.size(0), len(self.active_domains), 1),
                                 1 / len(self.active_domains), device=branches.device)
            fused = branches.mean(dim=1)
        embedding = self.combine(fused)
        subject_logits = self.subject_head(grad_reverse(embedding, grl_strength)) if self.domain_adversarial else None
        return {"state_logits": self.state_head(embedding), "prediction_logits": self.predict_head(embedding).squeeze(-1),
                "subject_logits": subject_logits, "attention": weights.squeeze(-1), "embedding": embedding}
