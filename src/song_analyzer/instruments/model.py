from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FamilyClassifier(nn.Module):
    """Lightweight CNN on log-mel for NSynth-style family classification."""

    def __init__(self, num_classes: int, n_mels: int = 64, channels: tuple[int, ...] = (32, 64, 128, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(c4)
        self.pool = nn.MaxPool2d(2)
        # Adaptive pool so variable time frames work
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(c4 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, time_frames)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        x = x.flatten(1)
        return self.fc(x)
