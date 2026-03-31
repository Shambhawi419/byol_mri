"""
prepare_data.py
---------------
Converts raw fastMRI .h5 files into .pt format required by the dataloader
and builds the library.pt index file.

Usage:
    python scripts/prepare_data.py --data_dir /path/to/h5/files \
                                   --train_split 0.8 \
                                   --seq_types CORPDFS_FBK CORPD_FBK

The script will:
    1. Split files into train/val
    2. Convert each .h5 file to .pt format
    3. Build library.pt index
"""

import h5py
import torch
import numpy as np
import os
import argparse
from pathlib import Path


def convert_h5_to_pt(h5_path, output_path):
    """Convert a single .h5 file to .pt format."""
    with h5py.File(h5_path, 'r') as hf:
        kspace = hf['kspace'][:]
        attrs = dict(hf.attrs)
        seq_type = attrs.get('acquisition', 'CORPDFS_FBK')
        max_val = float(attrs.get('max', 1.0))

    # convert complex to real/imag split → (slices, coils, h, w, 2)
    kspace_real = np.stack([kspace.real, kspace.imag], axis=-1)
    kspace_tensor = torch.tensor(kspace_real, dtype=torch.float32)

    torch.save({
        'kspace': kspace_tensor,
        'sequence': seq_type,
        'max_val': max_val
    }, output_path)

    return seq_type, kspace_tensor.shape[0]


def prepare_data(data_dir, train_ratio=0.8, seq_types=None):
    """
    Main preparation function.

    Args:
        data_dir: directory containing .h5 files
        train_ratio: fraction of files to use for training
        seq_types: list of sequence types to include
    """
    data_dir = Path(data_dir)

    # find all h5 files
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    print(f"Found {len(h5_files)} .h5 files")

    # split into train/val
    n_train = int(len(h5_files) * train_ratio)
    train_files = h5_files[:n_train]
    val_files = h5_files[n_train:]
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    # create output directories
    train_dir = data_dir / 'multicoil_train'
    val_dir = data_dir / 'multicoil_val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # build library
    library = {"train": {}, "val": {}}

    for split, files in [("train", train_files), ("val", val_files)]:
        out_dir = train_dir if split == "train" else val_dir

        for h5_file in files:
            h5_path = data_dir / h5_file
            fname = h5_file.replace('.h5', '')
            pt_path = out_dir / f'{fname}.pt'

            print(f"  [{split}] Processing {h5_file}...")
            seq_type, num_slices = convert_h5_to_pt(h5_path, pt_path)

            # filter by seq_types if specified
            if seq_types and seq_type not in seq_types:
                print(f"    Skipping — seq_type {seq_type} not in {seq_types}")
                continue

            slices = [(fname, i) for i in range(num_slices)]

            if seq_type not in library[split]:
                library[split][seq_type] = []
            library[split][seq_type].append(slices)

            print(f"    ✓ seq_type={seq_type}, slices={num_slices}")

    # save library
    lib_path = data_dir / 'library.pt'
    torch.save(library, lib_path)
    print(f"\n✓ library.pt saved!")
    for split in ['train', 'val']:
        for seq, vols in library[split].items():
            total_slices = sum(len(v) for v in vols)
            print(f"  {split}/{seq}: {len(vols)} volumes | {total_slices} slices")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .h5 files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of files for training (default: 0.8)')
    parser.add_argument('--seq_types', type=str, nargs='+',
                        default=['CORPDFS_FBK', 'CORPD_FBK'],
                        help='Sequence types to include')
    args = parser.parse_args()

    prepare_data(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        seq_types=args.seq_types
    )