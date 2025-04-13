import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ====== Config ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#latent_dims = [2048]
epochs = 50
batch_size = 256
learning_rate = 1e-4
output_log_file = "vae_results.txt"
input_dim = 32 * 32 * 3  # CIFAR-10 image shape

# ====== Dataset ======
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ====== CNN Variational Autoencoder ======
class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ====== VAE Loss Function (BCE + KL) ======
def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.1 * kl_div

# ====== Train ======
def train_vae(model, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  -> Loss: {avg_loss:.6f}")

# ====== Evaluate ======
def evaluate(model):
    model.eval()
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)

            # Clamp output to [0, 1] just to be safe
            recon = recon.clamp(0, 1)

            # MSE
            mse = nn.functional.mse_loss(recon, x, reduction='mean').item()
            total_mse += mse

            # PSNR & SSIM (convert to numpy)
            for i in range(x.size(0)):
                img_orig = x[i].cpu().permute(1, 2, 0).numpy()
                img_recon = recon[i].cpu().permute(1, 2, 0).numpy()

                total_psnr += compute_psnr(img_orig, img_recon, data_range=1.0)
                total_ssim += compute_ssim(img_orig, img_recon, data_range=1.0, channel_axis=-1)
                count += 1

    return total_mse / len(test_loader), total_psnr / count, total_ssim / count

# ====== Visualization: Reconstruction Image Grid ======
def get_recon_samples(model, dataset, n=10):
    model.eval()
    origs, recons = [], []
    with torch.no_grad():
        for i in range(n):
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)
            out, _, _ = model(x)
            origs.append(x.squeeze(0).cpu())
            recons.append(out.squeeze(0).cpu())
    return origs, recons

def plot_comparison(orig_train, recon_train, orig_test, recon_test, latent_dim):
    fig, axs = plt.subplots(10, 4, figsize=(20, 8))
    imgs = []
    for o, r in zip(orig_train, recon_train): imgs += [o, r]
    for o, r in zip(orig_test, recon_test): imgs += [o, r]
    for i, ax in enumerate(axs.flat):
        ax.imshow(imgs[i].permute(1, 2, 0))
        ax.axis('off')
    plt.suptitle(f"VAE Reconstructions | Latent dim: {latent_dim}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(f"vae_recon_{latent_dim}.png")
    plt.close()

# ====== Main Loop ======
compression_rates = []
reconstruction_errors = []
psnr_scores = []
ssim_scores = []

with open(output_log_file, "a") as f:
    for latent_dim in latent_dims:
        print(f"\n=== Running VAE with latent_dim = {latent_dim} ===")
        model = ConvVAE(latent_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        train_vae(model, optimizer, epochs)

        mse, psnr, ssim = evaluate(model)
        compression_rate = latent_dim / input_dim
        compression_rates.append(compression_rate)
        reconstruction_errors.append(mse)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

        f.write(f"{latent_dim}: MSE={mse:.6f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}, CR={compression_rate:.6f}\n")
        print(f"  -> MSE: {mse:.6f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, Compression Rate: {compression_rate:.6f}")

        # Visualize reconstructions
        orig_train, recon_train = get_recon_samples(model, train_dataset)
        orig_test, recon_test = get_recon_samples(model, test_dataset)
        plot_comparison(orig_train, recon_train, orig_test, recon_test, latent_dim)

# ====== Final Plot: Compression vs Reconstruction Error ======
plt.figure(figsize=(8, 6))
plt.plot(compression_rates, reconstruction_errors, marker='o', label='MSE')
plt.plot(compression_rates, psnr_scores, marker='x', label='PSNR')
plt.plot(compression_rates, ssim_scores, marker='s', label='SSIM')
plt.xlabel("Compression Rate (latent_dim / 3072)")
plt.ylabel("Metric Value")
plt.title("Compression vs Reconstruction Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compression_tradeoff_vae_metrics.png")

print(" All experiments completed. Results saved to:")
print("  - vae_results.txt")
print("  - vae_recon_{latent_dim}.png")
print("  - compression_tradeoff_vae_metrics.png")
