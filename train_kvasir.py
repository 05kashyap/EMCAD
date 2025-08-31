import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataloader import PolypDataset
from utils.utils import DiceLoss  # utils.utils.DiceLoss
from lib.networks import EMCADNet  # lib.networks.EMCADNet

def build_loaders(data_root, train_list, val_list, img_size, batch_size, workers):
    img_dir = os.path.join(data_root, "Kvasir-SEG", "Kvasir-SEG", "images")
    msk_dir = os.path.join(data_root, "Kvasir-SEG", "Kvasir-SEG", "masks")

    train_ds = PolypDataset(image_root=img_dir, gt_root=msk_dir, trainsize=img_size, augmentations='True',  file_list=train_list)
    val_ds   = PolypDataset(image_root=img_dir, gt_root=msk_dir, trainsize=img_size, augmentations='False', file_list=val_list)

    if len(train_ds) == 0:
        raise ValueError(f'Empty train dataset. Check paths:\n  images={img_dir}\n  masks={msk_dir}\n  list={train_list}')
    if len(val_ds) == 0:
        raise ValueError(f'Empty val dataset. Check paths:\n  images={img_dir}\n  masks={msk_dir}\n  list={val_list}')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,         shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def validate(model, loader, device):
    model.eval()
    dice_loss_fn = DiceLoss(2)
    ce_loss_fn = nn.CrossEntropyLoss()
    tot_dice, tot_loss, n = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, gts in loader:
            imgs = imgs.to(device)
            gtl = (gts > 0.5).long().squeeze(1).to(device)

            out = model(imgs)
            logits = out[-1] if isinstance(out, (list, tuple)) else out

            loss = 0.3 * ce_loss_fn(logits, gtl) + 0.7 * dice_loss_fn(logits, gtl, softmax=True)
            tot_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            inter = ((pred == 1) & (gtl == 1)).sum().float()
            denom = (pred == 1).sum().float() + (gtl == 1).sum().float() + 1e-5
            dice1 = (2.0 * inter / denom).item()
            tot_dice += dice1
            n += 1
    model.train()
    return (tot_loss / max(n, 1)), (tot_dice / max(n, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./data/kvasir')
    ap.add_argument('--train_list', type=str, default='./data/kvasir/train.txt')
    ap.add_argument('--val_list', type=str,   default='./data/kvasir/val.txt')
    ap.add_argument('--img_size', type=int,   default=224)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int,     default=80)
    ap.add_argument('--lr', type=float,       default=1e-5)
    ap.add_argument('--encoder', type=str,    default='pvt_v2_b2')
    ap.add_argument('--pretrained_dir', type=str, default='./pretrained_pth/pvt/')
    ap.add_argument('--no_pretrain', action='store_true', default=False)
    ap.add_argument('--workers', type=int,    default=4)
    ap.add_argument('--out', type=str,        default='./model_pth/Kvasir')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EMCADNet(
        num_classes=2, kernel_sizes=[1, 3, 5], expansion_factor=2,
        dw_parallel=True, add=True, lgag_ks=3, activation='relu6',
        encoder=args.encoder, pretrain=not args.no_pretrain, pretrained_dir=args.pretrained_dir
    ).to(device)

    train_loader, val_loader = build_loaders(args.data_root, args.train_list, args.val_list, args.img_size, args.batch_size, args.workers)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dice = 0.0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', ncols=80)
        for imgs, gts in pbar:
            imgs = imgs.to(device)
            gtl = (gts > 0.5).long().squeeze(1).to(device)

            out = model(imgs)
            logits = out[-1] if isinstance(out, (list, tuple)) else out

            loss = 0.3 * ce_loss(logits, gtl) + 0.7 * dice_loss(logits, gtl, softmax=True)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=f'{loss.item():.4f}')

        val_loss, val_dice = validate(model, val_loader, device)
        print(f'Val loss {val_loss:.4f} | Val Dice (fg) {val_dice:.4f}')

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.out, 'best_kvasir.pth'))
        torch.save(model.state_dict(), os.path.join(args.out, 'last_kvasir.pth'))

if __name__ == '__main__':
    main()