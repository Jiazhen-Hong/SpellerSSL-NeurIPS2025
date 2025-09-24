import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1D(nn.Module):
    """
    1D-UNet for EEG reconstruction, supporting encoder feature extraction for downstream tasks.
    repr_mode:
      - "bottleneck": GAP(bottleneck) -> [B, 1024]
      - "multiscale": GAP(x1p,x2p,x3p,x4p) concat -> [B, 64+128+256+512=960]
      - "tokens":     bottleneck -> [B, 1024, T/16]
    """
    def __init__(self, in_channels, out_channels, repr_mode: str = "bottleneck"):
        super().__init__()
        assert repr_mode in ("bottleneck", "multiscale", "tokens")
        self.repr_mode = repr_mode

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    @torch.no_grad()
    def _gap(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,T] -> [B,C]
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)

    def forward(self, x, return_repr: bool = False):
        # Encoder
        x1 = self.encoder1(x)      # [B, 64, T]
        x1p = self.pool1(x1)       # [B, 64, T/2]

        x2 = self.encoder2(x1p)    # [B, 128, T/2]
        x2p = self.pool2(x2)       # [B, 128, T/4]

        x3 = self.encoder3(x2p)    # [B, 256, T/4]
        x3p = self.pool3(x3)       # [B, 256, T/8]

        x4 = self.encoder4(x3p)    # [B, 512, T/8]
        x4p = self.pool4(x4)       # [B, 512, T/16]

        # Bottleneck (pre-upsample)
        x5_preup = self.bottleneck(x4p)  # [B, 1024, T/16]
        repr_vec = None
        if return_repr:
            if self.repr_mode == "bottleneck":
                repr_vec = self._gap(x5_preup)                    # [B, 1024]
            elif self.repr_mode == "multiscale":
                r1 = self._gap(x1p)                               # 64
                r2 = self._gap(x2p)                               # 128
                r3 = self._gap(x3p)                               # 256
                r4 = self._gap(x4p)                               # 512
                repr_vec = torch.cat([r1, r2, r3, r4], dim=1)     # [B, 960]
            elif self.repr_mode == "tokens":
                repr_vec = x5_preup                                # [B, 1024, T/16]

        # Decoder
        x5 = self.upconv4(x5_preup)
        x5 = torch.cat((x4, x5), dim=1)
        x5 = self.decoder4(x5)

        x6 = self.upconv3(x5)
        x6 = torch.cat((x3, x6), dim=1)
        x6 = self.decoder3(x6)

        x7 = self.upconv2(x6)
        x7 = torch.cat((x2, x7), dim=1)
        x7 = self.decoder2(x7)

        x8 = self.upconv1(x7)
        x8 = torch.cat((x1, x8), dim=1)
        x8 = self.decoder1(x8)

        out = self.final_conv(x8)

        if return_repr:
            return out, repr_vec
        return out