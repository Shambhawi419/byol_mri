"""
pretrain_byol.py
----------------
BYOL (Bootstrap Your Own Latent) pretraining for MRI reconstruction.

Replaces the SimCLR-based pretraining in CL-MRI with BYOL which:
    - Requires NO negative pairs
    - Works better with small datasets
    - Uses EMA (Exponential Moving Average) target network
    - Asymmetric predictor prevents representational collapse

Usage:
    python pretrain_byol.py \
        --dset fastmriknee \
        --seq_types CORPDFS_FBK \
        --dp 0 \
        --bs 2 \
        --ne 100 \
        --num_workers 4

Reference:
    Grill et al., "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning", NeurIPS 2020.
    https://arxiv.org/abs/2006.07733
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import os

from utils.data import Data
import sys
sys.path.insert(0, ".")
from utils.transform import Transform_CLR
from utils.manager import set_seed, set_cuda, fetch_paths, set_logger, set_device

from models.varnet import VarNet, VarNetBYOL
from losses.byolloss import BYOLLoss


def update_target_network(online_net: nn.Module,
                          target_net: nn.Module,
                          tau: float = 0.996) -> None:
    """
    Exponential Moving Average (EMA) update of target network.

    target_params = tau * target_params + (1 - tau) * online_params

    The target network is NEVER updated via backpropagation —
    only through this EMA update after every training step.
    tau=0.996 means target moves slowly, providing stable targets.

    Args:
        online_net: the online network being trained
        target_net: the target network updated via EMA
        tau:        EMA decay rate (higher = slower target update)
    """
    for online_params, target_params in zip(
        online_net.parameters(), target_net.parameters()
    ):
        target_params.data = (
            tau * target_params.data +
            (1.0 - tau) * online_params.data
        )


def forward_pass_byol(positive_samples, online_net, target_net,
                      loss_fn, args):
    """
    BYOL forward pass.

    Two views of the same MRI (different undersampling masks) are
    passed through online and target networks symmetrically:

        view1 → online  (encoder + projector + predictor) → pred_1
        view2 → target  (encoder + projector only)        → proj_2
        view2 → online  (encoder + projector + predictor) → pred_2
        view1 → target  (encoder + projector only)        → proj_1

        loss = 0.5 * [MSE(pred_1, proj_2) + MSE(pred_2, proj_1)]

    Symmetric loss improves stability and representation quality.

    Args:
        positive_samples: batch containing two views of same MRI
        online_net:       online VarNetBYOL (with predictor)
        target_net:       target VarNetBYOL (no predictor, EMA updated)
        loss_fn:          BYOLLoss instance
        args:             parsed arguments
    Returns:
        scalar loss
    """
    batch_view1, batch_view2 = positive_samples[0], positive_samples[1]

    # view 1
    kspace_und1 = batch_view1.kspace_und.to(args.dv)
    mask1 = batch_view1.mask.to(args.dv)
    num_low_freqs1 = batch_view1.num_low_freqs.to(args.dv)

    # view 2
    kspace_und2 = batch_view2.kspace_und.to(args.dv)
    mask2 = batch_view2.mask.to(args.dv)
    num_low_freqs2 = batch_view2.num_low_freqs.to(args.dv)

    # online processes view 1 → prediction
    online_pred_1 = online_net(masked_kspace=kspace_und1,
                               mask=mask1,
                               num_low_frequencies=num_low_freqs1)

    # target processes view 2 → projection (no grad)
    with torch.no_grad():
        target_proj_2 = target_net(masked_kspace=kspace_und2,
                                   mask=mask2,
                                   num_low_frequencies=num_low_freqs2)

    # online processes view 2 → prediction
    online_pred_2 = online_net(masked_kspace=kspace_und2,
                               mask=mask2,
                               num_low_frequencies=num_low_freqs2)

    # target processes view 1 → projection (no grad)
    with torch.no_grad():
        target_proj_1 = target_net(masked_kspace=kspace_und1,
                                   mask=mask1,
                                   num_low_frequencies=num_low_freqs1)

    # symmetric loss
    loss = (loss_fn(online_pred_1, target_proj_2.detach()) +
            loss_fn(online_pred_2, target_proj_1.detach())) / 2.0

    return loss


def train_():
    parser = argparse.ArgumentParser(
        description="BYOL pretraining for MRI reconstruction"
    )

    # DATA ARGS
    parser.add_argument("--trainacc", type=str, default="2,4,6,8",
                        help="Acceleration factors for training k-space undersampling")
    parser.add_argument("--valacc", type=str, default="2,4,6,8",
                        help="Acceleration factors for validation k-space undersampling")
    parser.add_argument("--tnv", type=int, default=0,
                        help="Number of volumes for training (0=full dataset)")
    parser.add_argument("--vnv", type=int, default=0,
                        help="Number of volumes for validation (0=full dataset)")
    parser.add_argument("--viznv", type=int, default=0,
                        help="Number of slices to visualize")
    parser.add_argument("--mtype", type=str, default="random",
                        choices=("random", "equispaced"),
                        help="Type of k-space undersampling mask")
    parser.add_argument("--dset", type=str, default="fastmriknee",
                        choices=("fastmriknee", "fastmribrain"),
                        help="Dataset to use")
    parser.add_argument("--seq_types", type=str, default="CORPDFS_FBK",
                        help="Comma-separated sequence types to use")

    # TRAINING ARGS
    parser.add_argument("--bs", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--ne", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--dv", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu/mps)")
    parser.add_argument("--dp", type=str, default=None,
                        help="Data parallel GPU ids (e.g. '0' or '0,1')")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")

    # BYOL SPECIFIC ARGS
    parser.add_argument("--tau", type=float, default=0.996,
                        help="EMA decay rate for target network update")
    parser.add_argument("--proj_dim", type=int, default=256,
                        help="Projection/prediction MLP output dimension")
    parser.add_argument("--hidden_dim", type=int, default=4096,
                        help="Projection/prediction MLP hidden dimension")

    # EXPERIMENT ARGS
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--pf", type=int, default=10,
                        help="Plotting/saving frequency (epochs)")

    # MODEL ARGS — full paper settings by default
    parser.add_argument("--num_cascades", type=int, default=12,
                        help="Number of VarNet unrolled iterations")
    parser.add_argument("--pools", type=int, default=4,
                        help="Number of U-Net pooling layers")
    parser.add_argument("--chans", type=int, default=18,
                        help="Number of top-level U-Net channels")
    parser.add_argument("--sens_pools", type=int, default=4,
                        help="Number of sensitivity U-Net pooling layers")
    parser.add_argument("--sens_chans", type=int, default=8,
                        help="Number of sensitivity U-Net channels")

    # PARSE
    args = parser.parse_args()
    args.seq_types = args.seq_types.split(',')
    args.trainacc = [int(a) for a in args.trainacc.split(',')]
    args.valacc = [int(a) for a in args.valacc.split(',')]

    # RESUME FROM CHECKPOINT
    ckpt = torch.load(Path(args.ckpt), map_location='cpu') \
        if args.ckpt else None
    args.ne = args.ne - ckpt['epoch'] if args.ckpt else args.ne

    # SETUP
    set_seed()
    set_cuda()
    data_path, exp_path = fetch_paths(args.dset)

    # LOGGING
    logger = set_logger(exp_path)
    for entry in vars(args):
        logger.info(f'{entry}: {vars(args)[entry]}')
    logger.info(f'data_path = {str(data_path)}')
    logger.info(f'experiment_path = {str(exp_path)}')

    # BUILD ONLINE NETWORK (encoder + projector + predictor)
    online_backbone = VarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
    )
    online_net = VarNetBYOL(
        varnet=online_backbone,
        use_predictor=True,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
    )

    # BUILD TARGET NETWORK (encoder + projector only, no predictor)
    target_backbone = VarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
    )
    target_net = VarNetBYOL(
        varnet=target_backbone,
        use_predictor=False,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
    )

    # initialize target with same weights as online
    target_net.load_state_dict(online_net.state_dict(), strict=False)

    # target NEVER trains via backprop
    for param in target_net.parameters():
        param.requires_grad = False

    # LOAD CHECKPOINT IF RESUMING
    best_val_loss = float('inf')
    epoch_count = 0
    best_online_state_dict = None

    if ckpt:
        online_net.load_state_dict(ckpt['online_state_dict'])
        target_net.load_state_dict(ckpt['target_state_dict'])
        best_val_loss = ckpt['best_val_loss']
        epoch_count = ckpt['epoch']
        best_online_state_dict = ckpt['best_online_state_dict']

    logger.info(
        f'No. of trainable parameters (online): '
        f'{sum(p.numel() for p in online_net.parameters() if p.requires_grad)}'
    )

    # SET DEVICE
    online_net, args, device_ids = set_device(online_net, args)
    target_net = target_net.to(args.dv)

    logger.info(f'num GPUs available: {torch.cuda.device_count()}')
    logger.info(f'num GPUs using: {len(device_ids)}')
    if torch.cuda.device_count() > 0:
        logger.info(f'GPU model: {torch.cuda.get_device_name(args.dv)}')

    # LOAD DATA
    train_transform = Transform_CLR(
        train=True,
        mask_type=args.mtype,
        accelerations=args.trainacc
    )
    train_dataset = Data(
        root=data_path, train=True,
        seq_types=args.seq_types,
        transform=train_transform,
        nv=args.tnv
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )
    logger.info(
        f'Training set: No. of volumes: {train_dataset.num_volumes} '
        f'| No. of slices: {len(train_dataset)}'
    )
    logger.info(f'{train_dataset.data_per_seq[:-1]}')

    val_transform = Transform_CLR(
        train=False,
        mask_type=args.mtype,
        accelerations=args.valacc
    )
    val_dataset = Data(
        root=data_path, train=False,
        seq_types=args.seq_types,
        transform=val_transform,
        nv=args.vnv
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    logger.info(
        f'Validation set: No. of volumes: {val_dataset.num_volumes} '
        f'| No. of slices: {len(val_dataset)}'
    )
    logger.info(f'{val_dataset.data_per_seq[:-1]}')

    # LOSS AND OPTIMIZER
    loss_fn = BYOLLoss()
    logger.info(f'Optimizer: RMSprop | LR: {args.lr} | Tau: {args.tau}')
    # only optimize online network — target is updated via EMA
    optimizer = torch.optim.RMSprop(
        params=online_net.parameters(),
        lr=args.lr
    )

    summary = OrderedDict({
        "epoch_no": [],
        "train_loss": [],
        "val_loss": [],
    })

    # TRAINING LOOP
    for _ in range(args.ne):
        epoch_count += 1

        # ── TRAIN ──────────────────────────────────────────────────
        online_net.train()
        target_net.eval()
        train_loss = 0
        batch_count = 0

        with tqdm(train_loader, unit="batch") as train_epoch:
            for b in train_epoch:
                train_epoch.set_description(
                    f"Epoch {epoch_count} [Training]"
                )

                optimizer.zero_grad()
                loss = forward_pass_byol(
                    b, online_net, target_net, loss_fn, args
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                optimizer.step()

                # EMA update of target network after every batch
                update_target_network(online_net, target_net, tau=args.tau)

                train_epoch.set_postfix(train_loss=loss.detach().item())
                train_loss += (
                    loss.detach().to('cpu') * b[0].kspace_und.shape[0]
                )
                batch_count += b[0].kspace_und.shape[0]

        epoch_train_loss = train_loss / batch_count

        # ── VALIDATE ───────────────────────────────────────────────
        online_net.eval()
        val_loss = 0
        batch_count = 0

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as val_epoch:
                for b in val_epoch:
                    val_epoch.set_description(
                        f"Epoch {epoch_count} [Validation]"
                    )
                    loss = forward_pass_byol(
                        b, online_net, target_net, loss_fn, args
                    )
                    val_epoch.set_postfix(val_loss=loss.detach().item())
                    val_loss += (
                        loss.detach().to('cpu') * b[0].kspace_und.shape[0]
                    )
                    batch_count += b[0].kspace_und.shape[0]

        epoch_val_loss = val_loss / batch_count

        # ── SAVE ───────────────────────────────────────────────────
        summary["epoch_no"].append(epoch_count)
        summary["train_loss"].append(float(epoch_train_loss))
        summary["val_loss"].append(float(epoch_val_loss))
        pd.DataFrame.from_dict(summary, orient='columns').to_csv(
            Path(os.path.join(
                f'{exp_path}',
                f'{exp_path.name}_summary.csv'
            )), index=False
        )

        online_state_dict = (
            online_net.module.state_dict()
            if hasattr(online_net, 'module')
            else online_net.state_dict()
        )
        target_state_dict = (
            target_net.module.state_dict()
            if hasattr(target_net, 'module')
            else target_net.state_dict()
        )

        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            best_online_state_dict = online_state_dict

        # save checkpoint with keys compatible with train_unet_with_clmri.py
        # strip 'encoder.' prefix from online state dict for downstream use
        encoder_state_dict = {
            k[len('encoder.'):]: v
            for k, v in online_state_dict.items()
            if k.startswith('encoder.')
        }
        best_encoder_state_dict = {
            k[len('encoder.'):]: v
            for k, v in best_online_state_dict.items()
            if k.startswith('encoder.')
        } if best_online_state_dict else encoder_state_dict

        torch.save({
            'epoch': epoch_count,
            # BYOL specific keys
            'online_state_dict': online_state_dict,
            'target_state_dict': target_state_dict,
            'best_online_state_dict': best_online_state_dict,
            # compatible keys for train_unet_with_clmri.py
            'last_model_state_dict': encoder_state_dict,
            'best_model_state_dict': best_encoder_state_dict,
            'last_optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, os.path.join(exp_path, f'{exp_path.name}_model.pth'))


if __name__ == '__main__':
    train_()
    print('Done!')
