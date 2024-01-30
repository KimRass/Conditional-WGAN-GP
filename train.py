# References:
    # https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

import torch
from torch.optim import AdamW
from pathlib import Path
import argparse
from tqdm import tqdm

from utils import get_device, set_seed, image_to_grid, save_image
from mnist import get_mnist_dls
from model import CGAN


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--n_epochs", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--lr", type=float, default=0.0005, required=False)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def train_single_step(real_image, real_label, model, D_optim, G_optim, device):
    real_image = real_image.to(device)
    real_label = real_label.to(device)

    D_loss = model.get_D_loss(real_image=real_image, real_label=real_label)
    D_optim.zero_grad()
    D_loss.backward()
    D_optim.step()
    
    G_loss = model.get_G_loss(batch_size=real_image.size(0), device=real_image.device)
    G_optim.zero_grad()
    G_loss.backward()
    G_optim.step()
    return D_loss, G_loss


@torch.no_grad()
def validate(val_dl, model, device):
    model.eval()

    cum_D_loss = 0
    cum_G_loss = 0
    for real_image, real_label in val_dl:
        real_image = real_image.to(device)
        real_label = real_label.to(device)

        D_loss, G_loss = model.get_loss(real_image=real_image, real_label=real_label)
        cum_D_loss += D_loss.item()
        cum_G_loss += G_loss.item()

    model.train()
    return cum_D_loss / len(val_dl), cum_G_loss / len(val_dl),


def train(n_epochs, train_dl, model, D_optim, G_optim, save_dir, device):
    for epoch in range(1, n_epochs + 1):
        cum_D_loss = 0
        cum_G_loss = 0
        for real_image, real_label in tqdm(train_dl, leave=False):
            D_loss, G_loss = train_single_step(
                real_image=real_image,
                real_label=real_label,
                model=model,
                D_optim=D_optim,
                G_optim=G_optim,
                device=device,
            )
            cum_D_loss += D_loss.item()
            cum_G_loss += G_loss.item()
        train_D_loss = cum_D_loss / len(train_dl)
        train_G_loss = cum_G_loss / len(train_dl)

        log = f"""[ {epoch}/{n_epochs} ]"""
        log += f"[ D loss: {train_D_loss:.4f} ]"
        log += f"[ G loss: {train_G_loss:.4f} ]"
        print(log)

        gen_image = model.sample(batch_size=train_dl.batch_size, device=device)
        gen_grid = image_to_grid(gen_image, n_cols=int(train_dl.batch_size ** 0.5))
        save_image(gen_grid, Path(save_dir)/f"epoch_{epoch}.jpg")

        torch.save(model.state_dict(), str(Path(save_dir)/f"epoch_{epoch}.pth"))


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    train_dl, _ = get_mnist_dls(
        data_dir=args.DATA_DIR, batch_size=args.BATCH_SIZE, n_cpus=0,
    )

    model = CGAN(n_classes=10).to(DEVICE)
    D_optim = AdamW(model.D.parameters(), lr=args.LR)
    G_optim = AdamW(model.G.parameters(), lr=args.LR)

    train(
        n_epochs=args.N_EPOCHS,
        train_dl=train_dl,
        model=model,
        D_optim=D_optim,
        G_optim=G_optim,
        save_dir=args.SAVE_DIR,
        device=DEVICE,
    )

if __name__ == "__main__":
    main()
