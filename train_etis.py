import os
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import numpy as np

from lib.networks import EMCADNet  # lib.networks.EMCADNet

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


class ETISDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        msk_dir: str,
        names: List[str],
        img_size: int = 224,
        augment: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.names = names
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize

        self.to_tensor = transforms.ToTensor()
        self.normalize_tf = transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.names)

    def _load_pair(self, name: str):
        img_p = os.path.join(self.img_dir, name)
        msk_p = os.path.join(self.msk_dir, name)
        img = Image.open(img_p).convert('RGB')
        msk = Image.open(msk_p).convert('L')
        return img, msk

    def _random_augment(self, img: Image.Image, msk: Image.Image):
        if random.random() < 0.5:
            img = TF.hflip(img); msk = TF.hflip(msk)
        if random.random() < 0.5:
            img = TF.vflip(img); msk = TF.vflip(msk)
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])  # 90, 180, 270
            img = img.rotate(90 * k, resample=Image.BILINEAR, expand=False)
            msk = msk.rotate(90 * k, resample=Image.NEAREST, expand=False)
        if random.random() < 0.8:
            b = 0.2; c = 0.2; s = 0.2; h = 0.02
            img = TF.adjust_brightness(img, 1.0 + random.uniform(-b, b))
            img = TF.adjust_contrast(img, 1.0 + random.uniform(-c, c))
            img = TF.adjust_saturation(img, 1.0 + random.uniform(-s, s))
            img = TF.adjust_hue(img, random.uniform(-h, h))
        return img, msk

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img, msk = self._load_pair(name)

        if self.augment:
            img, msk = self._random_augment(img, msk)

        img = TF.resize(img, [self.img_size, self.img_size], interpolation=Image.BILINEAR)
        msk = TF.resize(msk, [self.img_size, self.img_size], interpolation=Image.NEAREST)

        img = self.to_tensor(img)
        if self.normalize:
            img = self.normalize_tf(img)

        msk_np = np.array(msk, dtype=np.uint8)
        msk = torch.from_numpy((msk_np > 0).astype(np.float32)).unsqueeze(0)  # [1,H,W]
        return img, msk


def discover_pairs(img_dir: str, msk_dir: str) -> List[str]:
    imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    masks_set = set(os.listdir(msk_dir))
    names = [f for f in imgs if f in masks_set]
    names.sort()
    if not names:
        raise RuntimeError(f'No matching image/mask names found.\n  images: {img_dir}\n  masks: {msk_dir}')
    return names


def split_names(names: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    names = names.copy()
    rnd.shuffle(names)
    n_val = max(1, int(round(len(names) * val_ratio)))
    val = names[:n_val]
    train = names[n_val:]
    return train, val


class DiceLossMC(nn.Module):
    def __init__(self, num_classes: int = 2, smooth: float = 1e-5, exclude_bg: bool = False):
        super().__init__()
        self.C = num_classes
        self.smooth = smooth
        self.exclude_bg = exclude_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor, softmax: bool = True) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1) if softmax else logits
        onehot = F.one_hot(target, num_classes=self.C).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        den = probs.sum(dims) + onehot.sum(dims)
        dice_c = (2 * inter + self.smooth) / (den + self.smooth)
        dice = dice_c[1:].mean() if (self.exclude_bg and self.C > 1) else dice_c.mean()
        return 1.0 - dice


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss()
    dice_loss_fn = DiceLossMC(2, exclude_bg=False)
    tot_dice_fg, tot_loss, n = 0.0, 0.0, 0
    for imgs, gts in loader:
        imgs = imgs.to(device)
        gtl = (gts > 0.5).long().squeeze(1).to(device)

        out = model(imgs)
        logits = out[-1] if isinstance(out, (list, tuple)) else out

        loss = 0.3 * ce_loss_fn(logits, gtl) + 0.7 * dice_loss_fn(logits, gtl, softmax=True)
        tot_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        inter = ((pred == 1) & (gtl == 1)).sum().float()
        denom = (pred == 1).sum().float() + (gtl == 1).sum().float() + 1e-5
        dice1 = (2.0 * inter / denom).item()
        tot_dice_fg += dice1
        n += 1
    model.train()
    return (tot_loss / max(n, 1)), (tot_dice_fg / max(n, 1))


def build_loaders(args):
    img_dir = os.path.join(args.data_root, "images")
    msk_dir = os.path.join(args.data_root, "masks")

    if args.train_list and os.path.isfile(args.train_list) and args.val_list and os.path.isfile(args.val_list):
        with open(args.train_list, 'r') as f:
            train_names = [l.strip() for l in f if l.strip()]
        with open(args.val_list, 'r') as f:
            val_names = [l.strip() for l in f if l.strip()]
    else:
        names = discover_pairs(img_dir, msk_dir)
        train_names, val_names = split_names(names, args.val_ratio, args.seed)

    train_ds = ETISDataset(img_dir, msk_dir, train_names, img_size=args.img_size, augment=True, normalize=True)
    val_ds   = ETISDataset(img_dir, msk_dir, val_names,   img_size=args.img_size, augment=False, normalize=True)

    if len(train_ds) == 0:
        raise ValueError(f'Empty train dataset. images={img_dir} masks={msk_dir}')
    if len(val_ds) == 0:
        raise ValueError(f'Empty val dataset. images={img_dir} masks={msk_dir}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./data/etis', help='folder with images/ and masks/')
    ap.add_argument('--train_list', type=str, default='', help='optional file with image filenames (one per line)')
    ap.add_argument('--val_list', type=str,   default='', help='optional file with image filenames (one per line)')
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--img_size', type=int,   default=224)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int,     default=80)
    ap.add_argument('--lr', type=float,       default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)

    ap.add_argument('--encoder', type=str,    default='pvt_v2_b2')
    ap.add_argument('--pretrained_dir', type=str, default='./pretrained_pth/pvt/')
    ap.add_argument('--no_pretrain', action='store_true', default=False)

    ap.add_argument('--workers', type=int,    default=4)
    ap.add_argument('--out', type=str,        default='./model_pth/etis')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    model = EMCADNet(
        num_classes=2, kernel_sizes=[1, 3, 5], expansion_factor=2,
        dw_parallel=True, add=True, lgag_ks=3, activation='relu6',
        encoder=args.encoder, pretrain=not args.no_pretrain, pretrained_dir=args.pretrained_dir
    ).to(device)

    train_loader, val_loader = build_loaders(args)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLossMC(2, exclude_bg=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_dice = 0.0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', ncols=100)
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
            torch.save(model.state_dict(), os.path.join(args.out, 'best_etis.pth'))
        torch.save(model.state_dict(), os.path.join(args.out, 'last_etis.pth'))

    print(f'Training complete. Best Dice (fg): {best_dice:.4f}')


if __name__ == '__main__':
    main()