#!/usr/bin/env python3
# pix2pix_gan.py
# Minimal, production-ready Pix2Pix (U-Net + PatchGAN) with BCE-with-logits + L1.
# Supports "aligned" datasets (single image split left|right) or "folders" datasets (root/A, root/B).

import os
import argparse
from glob import glob
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

# -------------------
# 1) Dataset
# -------------------

class PairedAlignedDataset(Dataset):
    """
    Each file is a single image with input|target concatenated horizontally.
    We split it into left (A=input) and right (B=target).
    """
    def __init__(self, root, image_size=256, in_channels=3, out_channels=3):
        self.files = sorted(glob(os.path.join(root, "*")))
        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")
        # We normalize to [-1,1] to match Tanh output from G
        self.transform = T.Compose([
            T.Resize((image_size, image_size * 2), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            # split later; normalize after split (channel-wise)
        ])
        self.norm_in = T.Normalize([0.5]*in_channels, [0.5]*in_channels)
        self.norm_out = T.Normalize([0.5]*out_channels, [0.5]*out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")  # handle channels with transforms later
        ten = self.transform(img)  # [3, H, 2W] after resize
        C, H, W2 = ten.shape
        W = W2 // 2
        A = ten[:, :, :W]   # left
        B = ten[:, :, W:]   # right

        # If user wants in/out channels !=3, handle conversion simply
        if self.in_channels == 1:
            A = A.mean(dim=0, keepdim=True)
        if self.out_channels == 1:
            B = B.mean(dim=0, keepdim=True)

        A = self.norm_in(A)
        B = self.norm_out(B)
        return A, B, os.path.basename(path)


class PairedFoldersDataset(Dataset):
    """
    root/folderA/*.png  (inputs) and root/folderB/*.png (targets) with matching filenames.
    """
    def __init__(self, root, folderA="A", folderB="B", image_size=256, in_channels=3, out_channels=3):
        self.dirA = os.path.join(root, folderA)
        self.dirB = os.path.join(root, folderB)
        self.filesA = sorted(glob(os.path.join(self.dirA, "*")))
        if not self.filesA:
            raise FileNotFoundError(f"No images in {self.dirA}")
        self.filesB = {os.path.basename(p): p for p in glob(os.path.join(self.dirB, "*"))}
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.tA = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        self.tB = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        self.norm_in = T.Normalize([0.5]*in_channels, [0.5]*in_channels)
        self.norm_out = T.Normalize([0.5]*out_channels, [0.5]*out_channels)

    def __len__(self):
        return len(self.filesA)

    def __getitem__(self, idx):
        pathA = self.filesA[idx]
        name = os.path.basename(pathA)
        pathB = self.filesB.get(name)
        if pathB is None:
            raise FileNotFoundError(f"Matching file for {name} not found in {self.dirB}")

        imgA = Image.open(pathA).convert("RGB")
        imgB = Image.open(pathB).convert("RGB")
        A = self.tA(imgA)
        B = self.tB(imgB)

        if self.in_channels == 1:
            A = A.mean(dim=0, keepdim=True)
        if self.out_channels == 1:
            B = B.mean(dim=0, keepdim=True)

        A = self.norm_in(A)
        B = self.norm_out(B)
        return A, B, name


# -------------------
# 2) Models (U-Net G, PatchGAN D)
# -------------------

def conv_block(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_c, out_c, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    """
    Standard pix2pix U-Net (8 downs, 8 ups). Works for 256x256.
    """
    def __init__(self, in_channels=3, out_channels=3, base_dim=64):
        super().__init__()
        # Encoder
        self.d1 = nn.Sequential(nn.Conv2d(in_channels, base_dim, 4, 2, 1), nn.LeakyReLU(0.2, True))  # no norm first
        self.d2 = conv_block(base_dim, base_dim*2)
        self.d3 = conv_block(base_dim*2, base_dim*4)
        self.d4 = conv_block(base_dim*4, base_dim*8)
        self.d5 = conv_block(base_dim*8, base_dim*8)
        self.d6 = conv_block(base_dim*8, base_dim*8)
        self.d7 = conv_block(base_dim*8, base_dim*8)
        self.d8 = nn.Sequential(nn.Conv2d(base_dim*8, base_dim*8, 4, 2, 1), nn.ReLU(True))  # bottleneck (no norm)

        # Decoder (with skip connections)
        self.u1 = deconv_block(base_dim*8, base_dim*8, dropout=True)
        self.u2 = deconv_block(base_dim*16, base_dim*8, dropout=True)
        self.u3 = deconv_block(base_dim*16, base_dim*8, dropout=True)
        self.u4 = deconv_block(base_dim*16, base_dim*8)
        self.u5 = deconv_block(base_dim*16, base_dim*4)
        self.u6 = deconv_block(base_dim*8, base_dim*2)
        self.u7 = deconv_block(base_dim*4, base_dim)
        self.u8 = nn.Sequential(
            nn.ConvTranspose2d(base_dim*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)  # 128
        d2 = self.d2(d1) # 64
        d3 = self.d3(d2) # 32
        d4 = self.d4(d3) # 16
        d5 = self.d5(d4) # 8
        d6 = self.d6(d5) # 4
        d7 = self.d7(d6) # 2
        d8 = self.d8(d7) # 1

        u1 = self.u1(d8)              # 2
        u2 = self.u2(torch.cat([u1, d7], dim=1)) # 4
        u3 = self.u3(torch.cat([u2, d6], dim=1)) # 8
        u4 = self.u4(torch.cat([u3, d5], dim=1)) # 16
        u5 = self.u5(torch.cat([u4, d4], dim=1)) # 32
        u6 = self.u6(torch.cat([u5, d3], dim=1)) # 64
        u7 = self.u7(torch.cat([u6, d2], dim=1)) # 128
        u8 = self.u8(torch.cat([u7, d1], dim=1)) # 256
        return u8


class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN: maps (in+out) image to a grid of logits (no sigmoid).
    """
    def __init__(self, in_channels=3, out_channels=3, base_dim=64):
        super().__init__()
        C = in_channels + out_channels
        self.model = nn.Sequential(
            # no norm on first layer
            nn.Conv2d(C, base_dim, 4, 2, 1), nn.LeakyReLU(0.2, True),          # 128
            nn.Conv2d(base_dim, base_dim*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_dim*2), nn.LeakyReLU(0.2, True),            # 64
            nn.Conv2d(base_dim*2, base_dim*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_dim*4), nn.LeakyReLU(0.2, True),            # 32
            nn.Conv2d(base_dim*4, base_dim*8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(base_dim*8), nn.LeakyReLU(0.2, True),            # 31
            nn.Conv2d(base_dim*8, 1, 4, 1, 1)                                   # ~30x30 logits map
        )

    def forward(self, x, y):
        # Concatenate condition and output along channels
        inp = torch.cat([x, y], dim=1)
        return self.model(inp)  # logits map (no sigmoid)


# -------------------
# 3) Training utils
# -------------------

def save_visuals(A, B, B_hat, out_dir, tag, max_vis=8):
    os.makedirs(out_dir, exist_ok=True)
    # denorm from [-1,1] to [0,1]
    def denorm(t):
        return (t.clamp(-1,1) + 1) * 0.5
    # Stack triplets: [A | B_hat | B]
    vis = []
    n = min(max_vis, A.size(0))
    for i in range(n):
        row = torch.cat([denorm(A[i]), denorm(B_hat[i]), denorm(B[i])], dim=2)
        vis.append(row)
    grid = torch.cat(vis, dim=1).unsqueeze(0)  # single big image
    vutils.save_image(grid, os.path.join(out_dir, f"{tag}.png"))


# -------------------
# 4) Main training
# -------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root.")
    parser.add_argument("--mode", type=str, default="aligned", choices=["aligned", "folders"])
    parser.add_argument("--folderA", type=str, default="A")
    parser.add_argument("--folderB", type=str, default="B")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset / Loader
    if args.mode == "aligned":
        dataset = PairedAlignedDataset(args.data_root, args.image_size, args.in_channels, args.out_channels)
    else:
        dataset = PairedFoldersDataset(args.data_root, args.folderA, args.folderB, args.image_size,
                                       args.in_channels, args.out_channels)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Models
    G = UNetGenerator(args.in_channels, args.out_channels).to(device)
    D = PatchDiscriminator(args.in_channels, args.out_channels).to(device)

    # Losses
    adv_criterion = nn.BCEWithLogitsLoss()  # patch-wise BCE on logits
    l1_criterion  = nn.L1Loss()             # pixel-wise L1

    # Optims
    optim_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optim_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # dirs
    os.makedirs("samples_pix2pix", exist_ok=True)
    os.makedirs("checkpoints_pix2pix", exist_ok=True)

    for epoch in range(args.epochs):
        G.train(); D.train()
        for i, (A, B, names) in enumerate(dataloader):
            A = A.to(device)  # condition
            B = B.to(device)  # target
            bs = A.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optim_D.zero_grad()

            # Real pairs -> target 1s
            real_logits = D(A, B)
            real_targets = torch.ones_like(real_logits, device=device)
            loss_D_real = adv_criterion(real_logits, real_targets)

            # Fake pairs -> target 0s
            with torch.no_grad():
                B_hat = G(A)
            fake_logits = D(A, B_hat.detach())
            fake_targets = torch.zeros_like(fake_logits, device=device)
            loss_D_fake = adv_criterion(fake_logits, fake_targets)

            loss_D = (loss_D_real + loss_D_fake) * 0.5  # optional 0.5 factor
            loss_D.backward()
            optim_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optim_G.zero_grad()
            B_hat = G(A)
            logits_fake_for_G = D(A, B_hat)

            # Adversarial: want D to classify fake as real (1s)
            targets_for_G = torch.ones_like(logits_fake_for_G, device=device)
            loss_G_adv = adv_criterion(logits_fake_for_G, targets_for_G)

            # L1 reconstruction
            loss_G_l1 = l1_criterion(B_hat, B) * args.lambda_l1

            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            optim_G.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}]  "
                      f"Iter [{i}/{len(dataloader)}]  "
                      f"D: {loss_D.item():.4f}  G_adv: {loss_G_adv.item():.4f}  L1(x{args.lambda_l1}): {loss_G_l1.item():.4f}  G: {loss_G.item():.4f}")

        # Save visuals/checkpoints
        if (epoch + 1) % args.save_every == 0:
            G.eval()
            with torch.no_grad():
                A_vis, B_vis, _ = next(iter(dataloader))
                A_vis = A_vis.to(device)
                B_vis = B_vis.to(device)
                B_hat_vis = G(A_vis)
                save_visuals(A_vis, B_vis, B_hat_vis, "samples_pix2pix", f"epoch_{epoch+1}", max_vis=6)
            torch.save(G.state_dict(), f"checkpoints_pix2pix/G_epoch_{epoch+1}.pth")
            torch.save(D.state_dict(), f"checkpoints_pix2pix/D_epoch_{epoch+1}.pth")

    print("Training complete. Check 'samples_pix2pix' and 'checkpoints_pix2pix'.")

if __name__ == "__main__":
    main()
