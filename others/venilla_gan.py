import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os

# -------------------
# 1. Hyperparameters
# -------------------
batch_size = 64
lr = 0.0002
z_dim = 100        # size of noise vector
num_epochs = 50
image_size = 28    # MNIST images are 28x28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# 2. Data (MNIST)
# -------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # scale to [-1, 1]
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=False)

# dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# print(f"Dataset size: {len(dataset)} images")
# exit()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------
# 3. Generator
# -------------------
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.ReLU(True),

            nn.Linear(1024, image_size * image_size),
            nn.Tanh()   # output in [-1,1]
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, 1, image_size, image_size)

# -------------------
# 4. Discriminator
# -------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size * image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()   # output probability
        )

    def forward(self, x):
        x = x.view(-1, image_size * image_size)
        return self.net(x)

# -------------------
# 5. Initialize models
# -------------------
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

# Loss = Binary Cross Entropy
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Create output folder
os.makedirs("venilla_gan_samples", exist_ok=True)

# -------------------
# 6. Training Loop
# -------------------
# for i, (real_imgs, _) in enumerate(dataloader):
#     print(f"Batch {i+1}/{len(dataloader)}")
#     print(f"Real images shape: {real_imgs.shape}")
#     print(f"real image: {real_imgs.size(0)}")
#     break

# real_labels = torch.ones(12, 1).to(device)
# print(f"real_labels: {real_labels.size()}")
# exit()

for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size_curr = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # ---- Train Discriminator ----
        optimizer_D.zero_grad()

        # Real images label = 1
        real_labels = torch.ones(batch_size_curr, 1).to(device)
        # print(f"d: {D(real_imgs).size()}")
        output_real = D(real_imgs)
        loss_real = criterion(output_real, real_labels)

        # Fake images label = 0
        z = torch.randn(batch_size_curr, z_dim).to(device)
        fake_imgs = G(z)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)
        output_fake = D(fake_imgs.detach())
        loss_fake = criterion(output_fake, fake_labels)

        # Total D loss
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # ---- Train Generator ----
        optimizer_G.zero_grad()

        # Generator wants D to output 1 for fakes
        target_labels = torch.ones(batch_size_curr, 1).to(device)
        output = D(fake_imgs)   # discriminator sees fakes
        loss_G = criterion(output, target_labels)

        loss_G.backward()
        optimizer_G.step()

        # Print training stats
        if i % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} "
                  f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Save sample images from generator
    # with torch.no_grad():
    #     sample_z = torch.randn(64, z_dim).to(device)
    #     fake_samples = G(sample_z)
    #     vutils.save_image(fake_samples, f"gan_samples/epoch_{epoch+1}.png", 
    #                       normalize=True, nrow=8)

    # --- Save checkpoints and one visualization at epoch 25 only ---
    if (epoch + 1) == 25:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("gan_samples", exist_ok=True)

        # 1) Save model weights
        torch.save(G.state_dict(), "checkpoints/G_epoch25.pth")
        torch.save(D.state_dict(), "checkpoints/D_epoch25.pth")
        print("Saved checkpoints: checkpoints/G_epoch25.pth and checkpoints/D_epoch25.pth")

        # 2) Generate exactly 24 samples for a 3x8 grid
        with torch.no_grad():
            sample_z = torch.randn(24, z_dim, device=device)   # 24 samples
            fake_samples = G(sample_z)
            # nrow=8 => 8 columns; with 24 images you'll get 3 rows
            vutils.save_image(
                fake_samples, 
                "gan_samples/epoch_25.png",
                normalize=True, nrow=8
            )
            print("Saved sample grid: gan_samples/epoch_25.png (3 rows x 8 cols)")

