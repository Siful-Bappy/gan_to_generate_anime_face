
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Load MNIST (normalized to [0,1] just for visualization)
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# 2. Take the first 10 images
images, labels = zip(*[dataset[i] for i in range(10)])

# 3. Make a row of images
plt.figure(figsize=(15, 2))
for idx, (img, label) in enumerate(zip(images, labels)):
    # Tensor [1,28,28] -> numpy [28,28]
    img = img.squeeze().numpy()
    plt.subplot(1, 10, idx + 1)
    plt.imshow(img, cmap="gray")
    plt.title(str(label))
    plt.axis("off")

plt.tight_layout()
plt.show()
