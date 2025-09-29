# cycle_gan_toy.py
import os, random, glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from torchvision import transforms

# -----------------------
# 0) Utils & Hyperparams
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = "data"            # expects data/X and data/Y
dir_X = os.path.join(data_root, "X")
dir_Y = os.path.join(data_root, "Y")

img_size = 64
batch_size = 64
epochs = 10

lr = 2e-4
beta1, beta2 = 0.5, 0.999

lambda_cyc = 10.0
lambda_id  = 5.0              # (usually 0.5 * lambda_cyc)

save_every = 10                # save samples every N epochs

os.makedirs("cycle_gan_output/cyclegan_samples", exist_ok=True)
os.makedirs("cycle_gan_output/cyclegan_ckpts", exist_ok=True)

tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # map to [-1,1]
])

# ----------------------
# 1) Unpaired Dataset
# ----------------------
class UnpairedDataset(Dataset):
    def __init__(self, dir_X, dir_Y, transform=None):

        self.x_paths = sorted(sum([glob.glob(os.path.join(dir_X, ext)) for ext in ("*.jpg","*.png","*.jpeg","*.bmp")], []))
        self.y_paths = sorted(sum([glob.glob(os.path.join(dir_Y, ext)) for ext in ("*.jpg","*.png","*.jpeg","*.bmp")], []))
        if len(self.x_paths) == 0 or len(self.y_paths) == 0:
            raise RuntimeError("No images found. Put images under data/X and data/Y")
        self.transform = transform

    def __len__(self):
        return max(len(self.x_paths), len(self.y_paths))

    def __getitem__(self, idx):
        x_path = self.x_paths[idx % len(self.x_paths)]
        y_path = random.choice(self.y_paths)  # unaligned: random Y

        x = Image.open(x_path).convert("RGB")
        y = Image.open(y_path).convert("RGB")
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

# dataset link -> https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset
root = "../dataset/horse_to_zebra"
# dataset = UnpairedDataset(dir_X, dir_Y, tfm)

dataset = UnpairedDataset(os.path.join(root, "trainA"), os.path.join(root, "trainB"), tfm)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# print(f"Dataset sizes: X: {len(dataset.x_paths)}, Y: {len(dataset.y_paths)}")
# exit()

# ----------------------
# 2) Building Blocks
# ----------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),

            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    """
    c7s1-64, d128, d256, R256 x 6, u128, u64, c7s1-3 (small version)
    """
    def __init__(self, in_c=3, out_c=3, ngf=64, n_blocks=6):
        super().__init__()
        layers = []
        # c7s1-64
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(True)
        ]
        # d128, d256
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2, affine=False, track_running_stats=False),
                nn.ReLU(True)
            ]
            mult *= 2
        # Residual blocks
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult)]
        # u128, u64
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2, affine=False, track_running_stats=False),
                nn.ReLU(True)
            ]
            mult //= 2
        # c7s1-3
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_c, kernel_size=7),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN (small)
    C64 - C128 - C256 - C512 - 1
    """
    def __init__(self, in_c=3, ndf=64):
        super().__init__()
        kw, pad = 4, 1
        sequence = [
            nn.Conv2d(in_c, ndf, kernel_size=kw, stride=2, padding=pad),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf*2, kernel_size=kw, stride=2, padding=pad, bias=False),
            nn.InstanceNorm2d(ndf*2, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=kw, stride=2, padding=pad, bias=False),
            nn.InstanceNorm2d(ndf*4, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=kw, stride=1, padding=pad, bias=False),
            nn.InstanceNorm2d(ndf*8, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*8, 1, kernel_size=kw, stride=1, padding=pad)  # logits map
        ]
        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        return self.net(x)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

# ----------------------
# 3) Models
# ----------------------
G_X2Y = ResnetGenerator(in_c=3, out_c=3, ngf=64, n_blocks=6).to(device)
G_Y2X = ResnetGenerator(in_c=3, out_c=3, ngf=64, n_blocks=6).to(device)
D_X   = PatchDiscriminator(in_c=3, ndf=64).to(device)  # real X vs fake X
D_Y   = PatchDiscriminator(in_c=3, ndf=64).to(device)  # real Y vs fake Y

G_X2Y.apply(weights_init)
G_Y2X.apply(weights_init)
D_X.apply(weights_init)
D_Y.apply(weights_init)

# ----------------------
# 4) Losses & Optims
# ----------------------
criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1  = nn.L1Loss()

opt_G = optim.Adam(list(G_X2Y.parameters()) + list(G_Y2X.parameters()), lr=lr, betas=(beta1, beta2))
opt_DX = optim.Adam(D_X.parameters(), lr=lr, betas=(beta1, beta2))
opt_DY = optim.Adam(D_Y.parameters(), lr=lr, betas=(beta1, beta2))

def real_like(t): return torch.ones_like(t, device=device)
def fake_like(t): return torch.zeros_like(t, device=device)

# ----------------------
# 5) Training
# ----------------------
def save_samples(epoch, loader):
    G_X2Y.eval(); G_Y2X.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        fake_y = G_X2Y(x)
        rec_x  = G_Y2X(fake_y)
        fake_x = G_Y2X(y)
        rec_y  = G_X2Y(fake_x)

        # grid: [X, G(X), F(G(X)) | Y, F(Y), G(F(Y))]
        grid = torch.cat([
            (x+1)/2, (fake_y+1)/2, (rec_x+1)/2,
            (y+1)/2, (fake_x+1)/2, (rec_y+1)/2
        ], dim=0)
        vutils.save_image(grid, f"cycle_gan_output/cyclegan_samples/epoch_{epoch:03d}.png", nrow=batch_size)
    G_X2Y.train(); G_Y2X.train()

print(f"Dataset size (batches): {len(loader)}")
for epoch in range(1, epochs+1):
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # -------------------------
        # 5.1) Train D_Y (real Y vs G(X))
        # -------------------------
        opt_DY.zero_grad()
        logits_real_y = D_Y(y)
        loss_DY_real  = criterion_gan(logits_real_y, real_like(logits_real_y))

        fake_y = G_X2Y(x).detach()
        logits_fake_y = D_Y(fake_y)
        loss_DY_fake  = criterion_gan(logits_fake_y, fake_like(logits_fake_y))

        loss_DY = 0.5 * (loss_DY_real + loss_DY_fake)
        loss_DY.backward()
        opt_DY.step()

        # -------------------------
        # 5.2) Train D_X (real X vs F(Y))
        # -------------------------
        opt_DX.zero_grad()
        logits_real_x = D_X(x)
        loss_DX_real  = criterion_gan(logits_real_x, real_like(logits_real_x))

        fake_x = G_Y2X(y).detach()
        logits_fake_x = D_X(fake_x)
        loss_DX_fake  = criterion_gan(logits_fake_x, fake_like(logits_fake_x))

        loss_DX = 0.5 * (loss_DX_real + loss_DX_fake)
        loss_DX.backward()
        opt_DX.step()

        # -------------------------
        # 5.3) Train Generators G_X2Y and G_Y2X
        # -------------------------
        opt_G.zero_grad()

        # Adversarial: G_X2Y aims D_Y(fake_y) -> real
        fake_y = G_X2Y(x)
        logits_fake_y = D_Y(fake_y)
        loss_G_X2Y_adv = criterion_gan(logits_fake_y, real_like(logits_fake_y))

        # Adversarial: G_Y2X aims D_X(fake_x) -> real
        fake_x = G_Y2X(y)
        logits_fake_x = D_X(fake_x)
        loss_G_Y2X_adv = criterion_gan(logits_fake_x, real_like(logits_fake_x))

        # Cycle-consistency
        rec_x = G_Y2X(fake_y)     # F(G(x)) ~ x
        rec_y = G_X2Y(fake_x)     # G(F(y)) ~ y
        loss_cyc_X = criterion_l1(rec_x, x)
        loss_cyc_Y = criterion_l1(rec_y, y)
        loss_cyc = (loss_cyc_X + loss_cyc_Y) * lambda_cyc

        # Identity (optional but helpful for color preservation)
        idt_y = G_X2Y(y)          # should be ~ y
        idt_x = G_Y2X(x)          # should be ~ x
        loss_idt = (criterion_l1(idt_y, y) + criterion_l1(idt_x, x)) * lambda_id

        loss_G = loss_G_X2Y_adv + loss_G_Y2X_adv + loss_cyc + loss_idt
        loss_G.backward()
        opt_G.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(loader)}] "
                  f"DY: {loss_DY.item():.3f} DX: {loss_DX.item():.3f} | "
                  f"G_adv: {(loss_G_X2Y_adv+loss_G_Y2X_adv).item():.3f} "
                  f"cyc: {loss_cyc.item():.3f} id: {loss_idt.item():.3f}")

    if epoch % save_every == 0 or epoch == epochs:
        save_samples(epoch, loader)
        torch.save({
            "G_X2Y": G_X2Y.state_dict(),
            "G_Y2X": G_Y2X.state_dict(),
            "D_X": D_X.state_dict(),
            "D_Y": D_Y.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_DX": opt_DX.state_dict(),
            "opt_DY": opt_DY.state_dict(),
            "epoch": epoch
        }, f"cyclegan_ckpts/ckpt_epoch_{epoch:03d}.pt")

print("Training done.")
