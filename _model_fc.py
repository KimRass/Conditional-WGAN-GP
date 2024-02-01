# References:
    # https://github.com/AKASHKADEL/dcgan-mnist/blob/master/networks.py
    # https://github.com/arturml/mnist-cgan/blob/master/mnist-cgan.ipynb

import torch
from torch import nn


def one_hot_encode_label(label, n_classes):
    return torch.eye(n_classes, device=label.device)[label]


class Discriminator(nn.Module):
    def __init__(self, n_classes, hidden_dim):
        super().__init__()

        self.n_classes = n_classes

        self.label_embed = nn.Embedding(n_classes, n_classes)

        self.layers = nn.Sequential(
            nn.Conv2d(1 + n_classes, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False),
        )
        # self.layers = nn.Sequential(
        #     nn.Linear(794, 1024),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 1),
        # )

    def forward(self, image, label):
        ohe_label = one_hot_encode_label(label=label, n_classes=self.n_classes)
        _, _, h, w = image.shape
        x = torch.cat([image, ohe_label[..., None, None].repeat(1, 1, h, w)], dim=1)
        x = self.layers(x)
        x = x.view(-1, 1).squeeze(1)
        return x

        # emb_label = self.label_embed(label)
        # _, _, h, w = image.shape
        # x = torch.cat([image, emb_label[..., None, None].repeat(1, 1, h, w)], dim=1)
        # x = self.layers(x)
        # x = x.view(-1, 1).squeeze(1)
        # print(x.shape)
        # return x

        # x = image.view(image.size(0), 784)
        # c = self.label_embed(label)
        # x = torch.cat([x, c], 1)
        # x = self.layers(x)
        # x = x.squeeze(1)
        # print(x.shape)
        # return x


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, hidden_dim):
        super().__init__()

        self.n_classes = n_classes
        self.latent_dim = latent_dim

        self.label_embed = nn.Embedding(n_classes, n_classes)

        self.layers = nn.Sequential(
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
            nn.Tanh(),
        )
        # self.layers = nn.Sequential(
        #     nn.Linear(110, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(1024, 784),
        #     nn.Tanh(),
        # )

    def forward(self, latent_vec, label):
        ohe_label = one_hot_encode_label(label=label, n_classes=self.n_classes)
        x = torch.cat([latent_vec, ohe_label], dim=1)
        x = self.layers(x[..., None, None])
        return x

        # emb_label = self.label_embed(label)
        # x = torch.cat([latent_vec, emb_label], dim=1)
        # x = self.layers(x[..., None, None])
        # print(x.shape)
        # return x

        # z = latent_vec.view(latent_vec.size(0), 100)
        # c = self.label_embed(label)
        # x = torch.cat([z, c], 1)
        # x = self.layers(x)
        # x = x.view(x.size(0), 1, 28, 28)
        # print(x.shape)
        # return x


class ConditionalWGANsGP(nn.Module):
    def __init__(self, D, G, D_optim, G_optim):
        super().__init__()

        self.D = D
        self.G = G
        self.D_optim = D_optim
        self.G_optim = G_optim
        self.n_classes = D.n_classes
        self.latent_dim = G.latent_dim

    def sample_latent_vec(self, batch_size, device):
        return torch.randn(size=(batch_size, self.latent_dim), device=device)

    def sample_label(self, batch_size, device):
        return torch.randint(0, self.n_classes, size=(batch_size,), device=device)

    def sample(self, batch_size, device):
        label = self.sample_label(batch_size=batch_size, device=device)
        return self.sample_using_label(label=label)

    def generate_image(self, latent_vec, label):
        image = self.G(latent_vec=latent_vec, label=label)
        return image

    def sample_using_label(self, label):
        batch_size = label.size(0)
        latent_vec = self.sample_latent_vec(batch_size=batch_size, device=label.device)
        image = self.G(latent_vec=latent_vec, label=label)
        return image

    def _get_gradient_penalty(self, real_image, fake_image, label):
        device = real_image.device
        eps = torch.rand((real_image.size(0), 1, 1, 1), device=device)
        inter_image = eps * real_image + (1 - eps) * fake_image
        inter_image.requires_grad = True
        inter_pred = self.D(image=inter_image, label=label)

        real_label = torch.ones_like(inter_pred, device=device)
        grad = torch.autograd.grad(
            outputs=inter_pred, inputs=inter_image, grad_outputs=real_label, create_graph=True, retain_graph=True,
        )[0]
        grad = grad.view(grad.size(0), -1)
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def get_D_loss(self, real_image, label, gp_weight):
        real_pred = self.D(image=real_image, label=label)

        batch_size = real_image.size(0)
        latent_vec = self.sample_latent_vec(batch_size=batch_size, device=real_image.device)
        fake_image = self.G(latent_vec=latent_vec, label=label)
        fake_pred = self.D(image=fake_image.detach(), label=label)
        D_loss1 = -torch.mean(real_pred) + torch.mean(fake_pred)

        gp = self._get_gradient_penalty(
            real_image=real_image, fake_image=fake_image.detach(), label=label,
        )
        D_loss2 = gp_weight * gp
        # print(D_loss1.shape, D_loss2.shape)
        return D_loss1 + D_loss2

    def get_G_loss(self, label):
        latent_vec = self.sample_latent_vec(batch_size=label.size(0), device=label.device)
        fake_image = self.G(latent_vec=latent_vec, label=label)
        fake_pred = self.D(image=fake_image, label=label)
        return -torch.mean(fake_pred)

    def train_single_step(self, real_image, label, gp_weight, n_D_updates, device):
        real_image = real_image.to(device)
        label = label.to(device)

        cum_D_loss = 0
        for _ in range(n_D_updates):
            D_loss = self.get_D_loss(
                real_image=real_image, label=label, gp_weight=gp_weight,
            )
            self.D_optim.zero_grad()
            D_loss.backward()
            self.D_optim.step()

            cum_D_loss += D_loss.item()
        
        G_loss = self.get_G_loss(label=label)
        self.G_optim.zero_grad()
        G_loss.backward()
        self.G_optim.step()
        return cum_D_loss / n_D_updates, G_loss.item()


if __name__ == "__main__":
        n_classes = 10
        label = torch.randint(0, 10, size=(4,))
        image = torch.randn((4, 1, 28, 28))
        latent_vec = torch.randn((4, 100))
        G = Generator(n_classes=n_classes)
        D = Discriminator(n_classes=n_classes)
        
        G(latent_vec=latent_vec, label=label).shape
        D(image=image, label=label).shape
