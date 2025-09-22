import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# Hyperparams
dataroot = "./data"
batch_size = 128
image_size = 64        # DCGAN uses 64x64
nc = 1                 # channels: 1 for MNIST (grayscale)
nz = 100               # latent vector size (noise)
ngf = 64               # feature maps in G
ndf = 64               # feature maps in D
num_epochs = 25
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "dcgan_samples"
ckpt_dir = "dcgan_checkpoints"

# Data: MNIST -> resize to 64x64, normalize to [-1,1]
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # (mean, std) for grayscale
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# print(f"Dataset size: {len(dataset)} images")
# print(f"Number of batches: {len(dataloader)}")
# exit()


# Weight init (DCGAN paper suggested)
def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)

"""
Generator (ConvTranspose2d upsampling)
Input z: (N, nz, 1, 1) -> output: (N, nc, 64, 64)
"""
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            # input Z: (nz) --> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            # (ngf*8) x 4 x 4 --> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # (ngf*4) x 8 x 8 --> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # (ngf*2) x 16 x 16 --> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf) x 32 x 32 --> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z):
        return self.main(z)

"""
Discriminator (Conv2d downsampling with LeakyReLU)
Input: (N, nc, 64, 64) -> output: (N, 1, 1, 1) -> squeeze
"""
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            # (nc) x 64 x 64 --> (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 32 x 32 --> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 16 x 16 --> (ndf*4) x 8 x 8
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 8 x 8 --> (ndf*8) x 4 x 4
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8) x 4 x 4 --> 1 x 1 x 1
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),  # probability of "real"
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(-1, 1)  # (N, 1)

# Instantiate, init, optim, loss
G = Generator(nz, ngf, nc).to(device)
D = Discriminator(nc, ndf).to(device)

G.apply(weights_init_dcgan)
D.apply(weights_init_dcgan)

criterion  = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# fixed noise for eval snapshots
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Training loop (alternating D and G)]
step = 0
for epoch in range(num_epochs):
    for i, (real, _) in enumerate(dataloader):
        real = real.to(device)   # (N, 1, 64, 64)
        bsz  = real.size(0)

        # (1) Update Discriminator
        D.zero_grad()

        # real batch -> label 1
        label_real = torch.ones(bsz, 1, device=device)
        out_real   = D(real)               # (N,1)
        loss_real  = criterion(out_real, label_real)

        # fake batch -> label 0
        noise      = torch.randn(bsz, nz, 1, 1, device=device)
        fake       = G(noise)
        label_fake = torch.zeros(bsz, 1, device=device)
        out_fake   = D(fake.detach())
        loss_fake  = criterion(out_fake, label_fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # (2) Update Generator
        G.zero_grad()
        # generator wants D(fake) -> 1 (fool D)
        out_fake_for_G = D(fake)
        loss_G = criterion(out_fake_for_G, label_real)  # pretend real
        loss_G.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Batch [{i}/{len(dataloader)}] "
                  f"Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")
        step += 1

    if (epoch + 1) == 25:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1) Save model weights
        torch.save(G.state_dict(), f"{ckpt_dir}/G_epoch25.pth")
        torch.save(D.state_dict(), f"{ckpt_dir}/D_epoch25.pth")
        print(f"Saved checkpoints: {ckpt_dir}/G_epoch25.pth and {ckpt_dir}/D_epoch25.pth")

        # 2) Generate exactly 64 samples for a 8x8 grid
        with torch.no_grad():
            fake_samples = G(fixed_noise).detach().cpu()
            vutils.save_image(fake_samples, f"{save_dir}/epoch_{epoch+1:03d}.png",
                          normalize=True, nrow=8)
            print(f"Saved sample grid: {save_dir}/epoch_{epoch+1:03d}.png (8 rows x 8 cols)")

