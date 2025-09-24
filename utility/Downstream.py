# utility_fine_tuning.py
import os, math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models import UNet1D


# ---------------------------
# UNet1D + classifier head
# ---------------------------
class UNet1DClassifier(nn.Module):
    def __init__(
        self,
        base_unet,
        num_classes: int = 2,
        dropout: float = 0.0,
        tap: str = "bottleneck",   # 'enc1'|'enc2'|'enc3'|'enc4'|'bottleneck'|'tokens'
        head_type: str = "tconv",  # 'tconv'|'attnpool'|'mhsa'
        proj_dim: int = 256,
        n_heads: int = 2,
        attn_drop: float = 0.0,
        ffn_mult: int = 1,
        pos_max_len: int = 256,    # rough max, ~ T/16
    ):
        super().__init__()
        assert tap in {"enc1","enc2","enc3","enc4","bottleneck","tokens"}
        assert head_type in {"tconv","attnpool","mhsa"}
        self.base = base_unet
        self.tap = tap
        self.head_type = head_type

        in_ch_map = {
            "enc1": 64, "enc2": 128, "enc3": 256, "enc4": 512,
            "bottleneck": 1024, "tokens": 1024
        }
        self.in_feat_channels = in_ch_map[tap]

        if head_type == "tconv":
            self.head = nn.Sequential(
                nn.Conv1d(self.in_feat_channels, proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.GELU(),
                nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=1, groups=proj_dim),  # depthwise
                nn.GELU(),
                nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=2, dilation=2, groups=proj_dim),
                nn.GELU(),
                nn.Conv1d(proj_dim, proj_dim, kernel_size=1, bias=False),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(proj_dim, num_classes)
            )

        elif head_type == "attnpool":
            self.proj = nn.Sequential(
                nn.Conv1d(self.in_feat_channels, proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.GELU(),
            )
            self.attn = nn.Conv1d(proj_dim, 1, kernel_size=1, bias=True)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(proj_dim, num_classes)
            )

        else:  # mhsa
            self.proj = nn.Conv1d(self.in_feat_channels, proj_dim, kernel_size=1, bias=False)
            self.temporal_dw = nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=1, groups=proj_dim, bias=False)
            self.pos_emb = nn.Parameter(torch.zeros(1, pos_max_len, proj_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.ln1 = nn.LayerNorm(proj_dim)
            self.mhsa = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=n_heads,
                                              dropout=attn_drop, batch_first=True)
            self.ln2 = nn.LayerNorm(proj_dim)
            self.ffn = nn.Sequential(
                nn.Linear(proj_dim, ffn_mult * proj_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(ffn_mult * proj_dim, proj_dim),
            )
            self.head_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.classifier = nn.Linear(proj_dim, num_classes)

    def _tap_feature(self, x: torch.Tensor) -> torch.Tensor:
        # encoder forward only
        x1 = self.base.encoder1(x)      # [B,64,T]
        x1p = self.base.pool1(x1)       # [B,64,T/2]
        x2 = self.base.encoder2(x1p)    # [B,128,T/2]
        x2p = self.base.pool2(x2)       # [B,128,T/4]
        x3 = self.base.encoder3(x2p)    # [B,256,T/4]
        x3p = self.base.pool3(x3)       # [B,256,T/8]
        x4 = self.base.encoder4(x3p)    # [B,512,T/8]
        x4p = self.base.pool4(x4)       # [B,512,T/16]

        if self.tap == "enc1": return x1p
        if self.tap == "enc2": return x2p
        if self.tap == "enc3": return x3p
        if self.tap == "enc4": return x4

        x5 = self.base.bottleneck(x4p)  # [B,1024,T/16]
        return x5

    @torch.no_grad()
    def _gap(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._tap_feature(x)

        if self.head_type == "tconv":
            z = self.head(feat)
            return self.classifier(z)

        elif self.head_type == "attnpool":
            h = self.proj(feat)  # [B,D,L]
            L = h.size(-1)
            if L == 1:
                z = h.squeeze(-1)
            else:
                att = torch.softmax(self.attn(h), dim=-1)  # [B,1,L]
                z = torch.sum(h * att, dim=-1)             # [B,D]
            return self.classifier(z)

        else:  # mhsa
            h = self.proj(feat)            # [B,D,L]
            h = self.temporal_dw(h)
            B, D, L = h.shape
            if L == 1:
                z = self.head_dropout(h.squeeze(-1))
                return self.classifier(z)

            pe = self.pos_emb
            if L > pe.size(1):
                pe = F.interpolate(pe.transpose(1,2), size=L, mode='linear', align_corners=False).transpose(1,2)
            tokens = h.transpose(1, 2) + pe[:, :L, :]   # [B,L,D]
            cls = self.cls_token.expand(B, -1, -1)      # [B,1,D]
            tokens = torch.cat([cls, tokens], dim=1)    # [B,L+1,D]

            x0 = tokens
            x1 = self.ln1(x0)
            attn_out, _ = self.mhsa(x1, x1, x1, need_weights=False)
            x1 = x0 + attn_out
            x2 = self.ln2(x1)
            x2 = x1 + self.ffn(x2)
            cls_out = self.head_dropout(x2[:, 0])
            return self.classifier(cls_out)


# Lightning wrapper
class UNet1DClsLit(LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        onecycle_max_lr: float = 5e-3,
        onecycle_epochs: int = 10,
        onecycle_steps_per_epoch: int = 100,
        dropout: float = 0.0,
        freeze_encoder: bool = False,
        class_weight: Tuple[float, float] = (1.0, 5.0),   # assume class 1 is rare
    ):
        super().__init__()
        self.save_hyperparameters()

        base = UNet1D(in_channels, in_channels)
        self.model = UNet1DClassifier(base, num_classes=num_classes, dropout=dropout)

        w = torch.tensor(class_weight, dtype=torch.float32)
        self.register_buffer("ce_weight", w, persistent=False)
        self.criterion = nn.CrossEntropyLoss(weight=self.ce_weight)

        if freeze_encoder:
            for p in self.model.base.encoder1.parameters(): p.requires_grad_(False)
            for p in self.model.base.encoder2.parameters(): p.requires_grad_(False)
            for p in self.model.base.encoder3.parameters(): p.requires_grad_(False)
            for p in self.model.base.encoder4.parameters(): p.requires_grad_(False)
            for p in self.model.base.bottleneck.parameters(): p.requires_grad_(False)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.long())
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True,             on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.onecycle_max_lr,
            epochs=self.hparams.onecycle_epochs,
            steps_per_epoch=self.hparams.onecycle_steps_per_epoch,
            pct_start=0.1, anneal_strategy="cos",
            div_factor=10.0, final_div_factor=100.0,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


def _unwrap_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model.model."):
            nk = k[len("model.model."):]
        elif k.startswith("model."):
            nk = k[len("model."):]
        else:
            nk = k
        new_sd[nk] = v
    return new_sd

def _filter_for_unet_encoder(sd: Dict[str, torch.Tensor], base_module: nn.Module) -> Dict[str, torch.Tensor]:
    target_keys = set(base_module.state_dict().keys())
    out = {}
    hit = 0
    for k, v in sd.items():
        if k in target_keys:
            out[k] = v
            hit += 1
    print(f"[Load SSL] matched={hit}, skipped={len(sd)-hit}")
    return out


# main fine-tune 
def finetune_unet1d_cls(
    *,
    ckpt_path: Optional[str] = None,          # None -> train from scratch
    train_loader: DataLoader,                 # train (val merged ok)
    val_loader: Optional[DataLoader] = None,  # not used here
    test_loader: DataLoader,
    result_dir: str,
    epochs: int = 10,
    seed: int = 42,
    log_to_tensorboard: bool = True,
    lr: float = 1e-3,
    max_lr: float = 5e-3,
    weight_decay: float = 1e-2,
    dropout: float = 0.0,
    freeze_encoder: bool = False,
    class_weight: Tuple[float,float] = (1.0, 5.0),
    tap: str = "bottleneck",                  # kept for API compat
    head_type: str = "tconv",
    proj_dim: int = 256,
) -> Tuple[LightningModule, List[Dict[str, float]]]:
    os.makedirs(result_dir, exist_ok=True)
    seed_everything(seed)

    xb, yb = next(iter(train_loader))
    assert xb.ndim == 3 and yb.ndim == 1, "need (x:[B,C,T], y:[B])"
    B, C, T = xb.shape
    assert T % 16 == 0, f"T must be divisible by 16, got {T}"
    steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / max(1, train_loader.batch_size)))

    lit = UNet1DClsLit(
        in_channels=C,
        num_classes=2,
        lr=lr,
        weight_decay=weight_decay,
        onecycle_max_lr=max_lr,
        onecycle_epochs=epochs,
        onecycle_steps_per_epoch=steps_per_epoch,
        dropout=dropout,
        freeze_encoder=freeze_encoder if (ckpt_path and os.path.isfile(ckpt_path)) else False,
        class_weight=class_weight
    )

    load_ok = False
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            raw = torch.load(ckpt_path, map_location="cpu")
            flat = _unwrap_state_dict(raw)
            enc_only = _filter_for_unet_encoder(flat, lit.model.base)
            missing, unexpected = lit.model.base.load_state_dict(enc_only, strict=False)
            print(f"[Load SSL] done. missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:    print(f"  missing(sample): {missing[:6]}")
            if unexpected: print(f"  unexpected(sample): {unexpected[:6]}")
            load_ok = True
        except Exception as e:
            print(f"[Load SSL] failed: {type(e).__name__}: {e}")
    else:
        print("[Load SSL] skip: train from scratch.")

    if freeze_encoder and not load_ok:
        print("[Warn] freeze_encoder=True but no pretrained loaded, unfreezing.")

    tb_logger = TensorBoardLogger(save_dir=result_dir, name="tb") if log_to_tensorboard else None
    ckpt_dir = os.path.join(result_dir, "models_cls_ft")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_best = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:03d}-best-train_loss={train_loss:.4f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_last=False,
        auto_insert_metric_name=False,
    )
    ckpt_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:03d}-last",
        save_top_k=0,
        save_last=True,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        default_root_dir=result_dir,
        logger=tb_logger,
        callbacks=[ckpt_best, ckpt_last],
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(lit, train_dataloaders=train_loader)

    best_path = ckpt_best.best_model_path
    if best_path and os.path.isfile(best_path):
        print(f"[Select ckpt] best-by-train_loss: {best_path}")
        lit_best = UNet1DClsLit.load_from_checkpoint(best_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lit_best = lit_best.to(device)
        lit = lit_best
    else:
        last_path = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.isfile(last_path):
            print(f"[Select ckpt] fallback last: {last_path}")
            lit_last = UNet1DClsLit.load_from_checkpoint(last_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lit_last = lit_last.to(device)
            lit = lit_last
        else:
            print("[Select ckpt] best and last missing, keep in-memory.")

    test_metrics = trainer.test(lit, dataloaders=test_loader)
    return lit, test_metrics