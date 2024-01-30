# References:
    # https://github.com/AKASHKADEL/dcgan-mnist/blob/master/networks.py

import torch
from torch import nn
from torch.nn import functional as F


def one_hot_encode_label(label, n_classes):
    return torch.eye(n_classes, device=label.device)[label]


class Discriminator(nn.Module):
    def __init__(self, n_classes, hidden_dim=64):
        super().__init__()

        self.n_classes = n_classes

        self.main = nn.Sequential(
            nn.Conv2d(1 + n_classes, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, label):
        ohe_label = one_hot_encode_label(label=label, n_classes=self.n_classes)
        _, _, h, w = image.shape
        x = torch.cat([image, ohe_label[..., None, None].repeat(1, 1, h, w)], dim=1)
        x = self.main(x)
        return x.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim=100, hidden_dim=64):
        super().__init__()

        self.n_classes = n_classes

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + n_classes, hidden_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, latent_vec, label):
        ohe_label = one_hot_encode_label(label=label, n_classes=self.n_classes)
        x = torch.cat([latent_vec, ohe_label], dim=1)
        return self.main(x[..., None, None])


class CGAN(nn.Module):
    def __init__(self, n_classes, latent_dim=100, hidden_dim=64):
        super().__init__()

        self.n_classes = n_classes
        self.latent_dim = latent_dim

        self.D = Discriminator(n_classes=n_classes, hidden_dim=hidden_dim)
        self.G = Generator(n_classes=n_classes, latent_dim=latent_dim, hidden_dim=hidden_dim)

    def sample_latent_vec(self, batch_size, device):
        return torch.randn(size=(batch_size, self.latent_dim), device=device)

    def sample_label(self, batch_size, device):
        return torch.randint(0, self.n_classes, size=(batch_size,), device=device)

    def sample(self, batch_size, device):
        label = self.sample_label(batch_size=batch_size, device=device)
        return self.sample_using_label(label=label)

    def sample_using_label(self, label):
        batch_size = label.size(0)
        latent_vec = self.sample_latent_vec(batch_size=batch_size, device=label.device)
        image = self.G(latent_vec=latent_vec, label=label)
        return image

    def get_D_loss(self, real_image, real_label):
        real_pred = self.D(image=real_image, label=real_label)
        real_label = torch.ones_like(real_pred, device=real_image.device)
        real_loss = F.binary_cross_entropy_with_logits(real_pred, real_label, reduction="mean")

        batch_size = real_image.size(0)
        latent_vec = self.sample_latent_vec(batch_size=batch_size, device=real_image.device)
        fake_label = self.sample_label(batch_size=batch_size, device=real_image.device)
        fake_image = self.G(latent_vec=latent_vec, label=fake_label)
        fake_pred = self.D(image=fake_image, label=fake_label)
        fake_label = torch.zeros_like(fake_pred, device=fake_image.device)
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_label, reduction="mean")
        return (real_loss + fake_loss) / 2

    def get_G_loss(self, batch_size, device):
        latent_vec = self.sample_latent_vec(batch_size=batch_size, device=device)
        fake_label = self.sample_label(batch_size=batch_size, device=device)
        fake_image = self.G(latent_vec=latent_vec, label=fake_label)
        pred = self.D(image=fake_image, label=fake_label)
        label = torch.ones_like(pred, device=fake_image.device)
        return F.binary_cross_entropy_with_logits(pred, label, reduction="mean")

    def get_loss(self, real_image, real_label):
        D_loss = self.get_D_loss(real_image=real_image, real_label=real_label)
        G_loss = self.get_G_loss(batch_size=real_image.size(0), device=real_image.device)
        return D_loss, G_loss


if __name__ == "__main__":
        n_classes = 10
        label = torch.randint(0, 10, size=(4,))
        image = torch.randn((4, 1, 28, 28))
        latent_vec = torch.randn((4, 100))
        G = Generator(n_classes=n_classes)
        D = Discriminator(n_classes=n_classes)
        
        G(latent_vec=latent_vec, label=label).shape
        D(image=image, label=label).shape
