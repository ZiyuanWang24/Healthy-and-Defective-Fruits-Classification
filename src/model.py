import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Encoder(nn.Module):
    """
    Wraps torchvision MobileNetV2 as a feature extractor:
    input:  [B, 3, H, W]
    output: [B, 1280]
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)

        self.features = base.features                # convolutional backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))     # global average pooling
        self.out_dim = 1280                          # MobileNetV2 final feature dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)             # [B, 1280, h, w]
        x = self.pool(x)                 # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)          # [B, 1280]
        return x


class FusionBlock(nn.Module):
    """
    Fuse two feature vectors f1 and f2 (both [B, D]).
    Supported:
      - "concat": [f1; f2] -> [B, 2D]
      - "sum":    f1 + f2  -> [B, D]
      - "gated":  learn gate g in (0,1): g*f1 + (1-g)*f2 -> [B, D]
    """
    def __init__(self, dim: int, mode: str = "concat"):
        super().__init__()
        assert mode in {"concat", "sum", "gated"}
        self.mode = mode
        self.out_dim = 2 * dim if mode == "concat" else dim

        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.Sigmoid()
            )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return torch.cat([f1, f2], dim=1)
        if self.mode == "sum":
            return f1 + f2
        if self.mode == "gated":
            g = self.gate(torch.cat([f1, f2], dim=1))   # [B, D] in (0,1)
            return g * f1 + (1.0 - g) * f2
        
        return None


class MultiInputMobileNetV2(nn.Module):
    """
    Two-branch MobileNetV2 -> Fusion -> MLP classifier

    forward(rgb, sil) returns logits: [B, num_classes]
    """
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        fusion: str = "gated",          # "concat" | "sum" | "gated"
        mlp_hidden: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.rgb_encoder = MobileNetV2Encoder(pretrained=pretrained)
        self.sil_encoder = MobileNetV2Encoder(pretrained=pretrained)

        feat_dim = self.rgb_encoder.out_dim
        self.fusion = FusionBlock(dim=feat_dim, mode=fusion)

        fused_dim = self.fusion.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, rgb: torch.Tensor, sil: torch.Tensor) -> torch.Tensor:
        f1 = self.rgb_encoder(rgb)   # [B, 1280]
        f2 = self.sil_encoder(sil)   # [B, 1280]
        fused = self.fusion(f1, f2)  # [B, 2560] if concat else [B, 1280]
        logits = self.mlp(fused)     # [B, num_classes]
        return logits

    @torch.no_grad()
    def extract_features(self, rgb: torch.Tensor, sil: torch.Tensor) -> torch.Tensor:
        """Optional: get fused feature vector before classifier."""
        self.eval()
        f1 = self.rgb_encoder(rgb)
        f2 = self.sil_encoder(sil)
        return self.fusion(f1, f2)
