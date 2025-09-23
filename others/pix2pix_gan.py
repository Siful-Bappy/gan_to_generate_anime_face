import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from PIL import Image, ImageFilter
import numpy as np

# Toy Paired Dataset
class ToyPairedDataset(Dataset):
    def __init__(self, length=500, size=64):
        super().__init__()
        self.length = length
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Input A: random black/white "edges"
        A = (np.random.rand(self.size, self.size) > 0.7).astype(np.float32)

        # Target B: blurred & 3-channel version
        img = Image.fromarray((A * 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=2)).convert("RGB")
        B = np.array(img).astype(np.float32) / 255.0

        # Normalize [-1,1]
        A = (A - 0.5) / 0.5
        B = (B - 0.5) / 0.5

        A = torch.tensor(A).unsqueeze(0)       # [1,H,W]
        B = torch.tensor(B).permute(2,0,1)     # [3,H,W]
        return A, B

# Generator (U-Net small)
class UNetGenerator(nn.Module):
    def __init__(self, in_c=1, out_c=3, base_dim=64):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_c, base_dim, 4, 2, 1), 
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, 4, 2, 1), 
            nn.BatchNorm2d(base_dim*2), 
            nn.ReLU(True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*4, 4, 2, 1), 
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim*4, base_dim*2, 4, 2, 1), 
            nn.BatchNorm2d(base_dim*2), 
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim*2, base_dim, 4, 2, 1), 
            nn.BatchNorm2d(base_dim), 
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_dim, out_c, 4, 2, 1), 
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        b = self.bottleneck(d2)
        u1 = self.up1(b)
        u2 = self.up2(u1)
        return self.final(u2)

# Discriminator (PatchGAN)
class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=1, out_c=3, base_dim=64):
        super().__init__()
        C = in_c + out_c
        self.net = nn.Sequential(
            nn.Conv2d(C, base_dim, 4, 2, 1), 
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(base_dim, base_dim*2, 4, 2, 1), 
            nn.BatchNorm2d(base_dim*2), 
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(base_dim*2, base_dim*4, 4, 2, 1), 
            nn.BatchNorm2d(base_dim*4), 
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(base_dim*4, 1, 4, 1, 1)  # PatchGAN logits map
        )

    def forward(self, A, B):
        return self.net(torch.cat([A, B], dim=1))

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ToyPairedDataset(length=200, size=64)
# print(f"Dataset size: {len(dataset)} image pairs")
# exit()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

criterion_adv = nn.BCEWithLogitsLoss()
criterion_l1  = nn.L1Loss()

optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

lambda_l1 = 100
epochs = 25

os.makedirs("pix2pix_samples", exist_ok=True)

# Training Loop
for epoch in range(epochs):
    for i, (A, B) in enumerate(dataloader):
        A, B = A.to(device), B.to(device)

        # Train D
        optimizer_D.zero_grad()
        real_logits = D(A, B)
        real_targets = torch.ones_like(real_logits)
        loss_D_real = criterion_adv(real_logits, real_targets)

        fake_B = G(A).detach()
        fake_logits = D(A, fake_B)
        fake_targets = torch.zeros_like(fake_logits)
        loss_D_fake = criterion_adv(fake_logits, fake_targets)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # Train G
        optimizer_G.zero_grad()
        fake_B = G(A)
        logits_fake = D(A, fake_B)
        loss_G_adv = criterion_adv(logits_fake, torch.ones_like(logits_fake))
        loss_G_l1 = criterion_l1(fake_B, B) * lambda_l1
        loss_G = loss_G_adv + loss_G_l1
        loss_G.backward()
        optimizer_G.step()

        if i % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, "
                  f"G_adv: {loss_G_adv.item():.4f}, L1: {loss_G_l1.item():.4f}")

    # Save sample outputs
    if (epoch + 1) == 25:
        with torch.no_grad():
            sample_A, sample_B = next(iter(dataloader))
            sample_A, sample_B = sample_A.to(device), sample_B.to(device)
            fake_sample_B = G(sample_A)
            grid = torch.cat([
                (sample_A.repeat(1,3,1,1) + 1)/2,   # gray input → RGB
                (fake_sample_B + 1)/2,              # generated
                (sample_B + 1)/2                    # real target
            ])
            vutils.save_image(grid, f"pix2pix_samples/epoch_{epoch+1}.png", nrow=8)
