import os
import math
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# project modules
from models import UNet1D
from utility.setup_logging import setup_logging
from utility.loss_mask import MSEWithSpectralLoss, L1WithSpectralLoss, random_time_mask


class ArrayReconDataset(Dataset):
    def __init__(self, X_np: np.ndarray):
        assert X_np.ndim == 3, f"[N,C,T] expected, got {X_np.shape}"
        self.X = torch.from_numpy(X_np.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [C, T]
        return x, x

def _resample_to_len(X_np: np.ndarray, new_len: int) -> np.ndarray:
    x = torch.from_numpy(X_np.astype(np.float32))  # [N,C,T]
    x_rs = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return x_rs.cpu().numpy()

def _compute_target_len(T_old: int, fs_old: int, fs_new: int) -> int:
    return int(round(T_old * fs_new / fs_old))

def _crop_to_duration(
    X_np: np.ndarray,
    fs: int,
    duration_ms: int,
    align: str = "start",
    multiple_of: int = 16,
) -> np.ndarray:
    N, C, T = X_np.shape
    need = int(round(fs * duration_ms / 1000.0))
    if need <= T:
        if align == "start":
            X_np = X_np[:, :, :need]
        elif align == "center":
            s = (T - need) // 2
            X_np = X_np[:, :, s : s + need]
        elif align == "end":
            X_np = X_np[:, :, T - need :]
        else:
            raise ValueError("align must be start|center|end")
    else:
        pad = need - T
        X_np = np.pad(X_np, ((0, 0), (0, 0), (0, pad)), mode="constant")

    rem = X_np.shape[-1] % multiple_of
    if rem != 0:
        X_np = np.pad(
            X_np, ((0, 0), (0, 0), (0, multiple_of - rem)), mode="constant"
        )
    return X_np

def preprocess_concat_all(
    X_train, X_val, X_test, orig_fs, target_fs, dur_ms, align, multiple_of
):
    def _maybe_resample(X):
        if orig_fs == target_fs:
            return X
        new_len = _compute_target_len(X.shape[-1], orig_fs, target_fs)
        return _resample_to_len(X, new_len)

    X_train = _maybe_resample(X_train)
    X_val = _maybe_resample(X_val)
    X_test = _maybe_resample(X_test)

    X_train = _crop_to_duration(X_train, target_fs, dur_ms, align=align, multiple_of=multiple_of)
    X_val   = _crop_to_duration(X_val,   target_fs, dur_ms, align=align, multiple_of=multiple_of)
    X_test  = _crop_to_duration(X_test,  target_fs, dur_ms, align=align, multiple_of=multiple_of)

    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    return X_all


class ReconLitModule(LightningModule):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        loss_name: str = "L1WithSpectralLoss",
        lr: float = 5e-5,
        weight_decay: float = 1e-2,
        onecycle_max_lr: float = 5e-4,
        onecycle_epochs: int = 200,
        onecycle_steps_per_epoch: int = 100,
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "UNet1D":
            self.model = UNet1D(in_channels, in_channels)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        if loss_name == "L1Loss":
            self.criterion = nn.L1Loss()
        elif loss_name == "L1WithSpectralLoss":
            self.criterion = L1WithSpectralLoss()
        elif loss_name == "MSEWithSpectralLoss":
            self.criterion = MSEWithSpectralLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad():
            y = x.clone()
        x_masked = random_time_mask(x, mask_ratio=self.hparams.mask_ratio, mode="zero")
        out = self(x_masked)
        loss = self.criterion(out, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.onecycle_max_lr,
            epochs=self.hparams.onecycle_epochs,
            steps_per_epoch=self.hparams.onecycle_steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=2.0,
            final_div_factor=100.0,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# Load arrays (expects X_train/X_val/X_test)
def load_arrays(data_file: str):
    if data_file.endswith(".npz"):
        pack = np.load(data_file, allow_pickle=True)
        keys = pack.files
        need = ["X_train", "X_val", "X_test"]
        if not all(k in keys for k in need):
            raise ValueError(f"{data_file} missing keys {need}")
        X_train, X_val, X_test = pack["X_train"], pack["X_val"], pack["X_test"]
    elif data_file.endswith(".npy"):
        obj = np.load(data_file, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
            obj = obj.item()
        if isinstance(obj, dict):
            need = ["X_train", "X_val", "X_test"]
            if not all(k in obj for k in need):
                raise ValueError(f"{data_file} dict missing keys {need}")
            X_train, X_val, X_test = obj["X_train"], obj["X_val"], obj["X_test"]
        else:
            raise ValueError("NPY must store a dict-like object with X_train/X_val/X_test")
    else:
        raise ValueError("data_file must be .npz or .npy")
    return X_train, X_val, X_test


# CLI
def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_file", type=str, required=True,
                   help="NPZ/NPY with keys X_train/X_val/X_test")
    p.add_argument("--orig_fs", type=int, default=240, help="original sampling rate")
    p.add_argument("--target_fs", type=int, default=240, help="resample target sampling rate")
    p.add_argument("--dur_ms", type=int, default=667, help="crop length (ms)")
    p.add_argument("--align", type=str, default="start", choices=["start", "center", "end"])
    p.add_argument("--multiple_of", type=int, default=16, help="pad time length to multiple")
    # training
    p.add_argument("--model_name", type=str, default="UNet1D",
                   choices=["EEGM2","EEGM2_S1","EEGM2_S3","EEGM2_S4","EEGM2_S5","UNet","UNet1D"])
    p.add_argument("--loss", type=str, default="L1WithSpectralLoss",
                   choices=["L1Loss", "L1WithSpectralLoss", "MSEWithSpectralLoss"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--onecycle_max_lr", type=float, default=5e-4)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    # device & output
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out_dir", type=str, default="./SSL_results")
    return p.parse_args()


# Main
if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed, workers=True)

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(
        args.out_dir,
        f'{datetime.now().strftime("%Y%m%d")}/{args.model_name}-{datetime.now().strftime("%H%M%S")}'
    )
    os.makedirs(run_dir, exist_ok=True)
    logger, _ = setup_logging(run_dir)

    # load data
    X_train, X_val, X_test = load_arrays(args.data_file)

    # preprocess and concat into one SSL set
    X_all = preprocess_concat_all(
        X_train, X_val, X_test,
        orig_fs=args.orig_fs, target_fs=args.target_fs,
        dur_ms=args.dur_ms, align=args.align, multiple_of=args.multiple_of
    )
    logger.info("=" * 78)
    logger.info(f"ALL-IN-ONE training set: {X_all.shape} (SSL, no labels)")
    logger.info("-" * 78)

    # dataloader
    train_ds = ArrayReconDataset(X_all)
    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent
    )

    # shape probe
    xb, _ = next(iter(train_loader))
    B, C, T = xb.shape
    logger.info(f"Example batch: {xb.shape} -> channels={C}, T={T}")

    # lightning module
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    lit = ReconLitModule(
        model_name=args.model_name,
        in_channels=C,
        loss_name=args.loss,
        lr=args.lr,
        weight_decay=args.weight_decay,
        onecycle_max_lr=args.onecycle_max_lr,
        onecycle_epochs=args.epochs,
        onecycle_steps_per_epoch=steps_per_epoch,
        mask_ratio=args.mask_ratio,
    )

    # callbacks / logger
    tb_logger = TensorBoardLogger(save_dir=run_dir, name="tb")
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "models"),
        filename="{epoch:03d}-train_loss={train_loss:.6f}",
        save_top_k=5,
        monitor="train_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )

    accelerator = "cpu" if args.cpu else ("gpu" if torch.cuda.is_available() else "cpu")
    devices = 1

    trainer = Trainer(
        default_root_dir=run_dir,
        logger=tb_logger,
        callbacks=[ckpt_cb],
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # train only
    trainer.fit(lit, train_dataloaders=train_loader)

    logger.info(f"Done. Checkpoints saved to: {os.path.join(run_dir, 'models')}")

