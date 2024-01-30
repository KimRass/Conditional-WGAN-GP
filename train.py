# References:
    # https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

import torch
from torch.optim import AdamW
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings

from utils import get_device, set_seed, image_to_grid, save_image
from mnist import get_mnist_dls
from model import Discriminator, Generator, CGAN


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--n_epochs", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--lr", type=float, default=0.0005, required=False)
    parser.add_argument("--gp_weight", type=float, default=10, required=False)
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


def train_single_step(real_image, label, model, D_optim, G_optim, gp_weight, device):
    real_image = real_image.to(device)
    label = label.to(device)

    D_loss = model.get_D_loss(real_image=real_image, label=label, gp_weight=gp_weight)
    D_optim.zero_grad()
    D_loss.backward()
    D_optim.step()
    
    G_loss = model.get_G_loss(label=label)
    G_optim.zero_grad()
    G_loss.backward()
    G_optim.step()
    return D_loss, G_loss


def train(n_classes, n_epochs, train_dl, model, D_optim, G_optim, gp_weight, save_dir, device):
    for epoch in range(1, n_epochs + 1):
        cum_D_loss = 0
        cum_G_loss = 0
        for real_image, label in tqdm(train_dl, leave=False):
            D_loss, G_loss = train_single_step(
                real_image=real_image,
                label=label,
                model=model,
                D_optim=D_optim,
                G_optim=G_optim,
                gp_weight=gp_weight,
                device=device,
            )
            cum_D_loss += D_loss.item()
            cum_G_loss += G_loss.item()
        train_D_loss = cum_D_loss / len(train_dl)
        train_G_loss = cum_G_loss / len(train_dl)

        log = f"""[ {epoch}/{n_epochs} ]"""
        log += f"[ D loss: {train_D_loss:.3f} ]"
        log += f"[ G loss: {train_G_loss:.3f} ]"
        print(log)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label = torch.arange(
                n_classes, dtype=torch.int32, device=device,
            ).repeat_interleave(10)
        gen_image = model.sample_using_label(label=label)
        gen_grid = image_to_grid(gen_image, n_cols=n_classes)
        save_image(gen_grid, Path(save_dir)/f"epoch_{epoch}.jpg")

        torch.save(model.G.state_dict(), str(Path(save_dir)/f"epoch_{epoch}.pth"))


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    train_dl, _ = get_mnist_dls(
        data_dir=args.DATA_DIR, batch_size=args.BATCH_SIZE, n_cpus=0,
    )

    N_CLASSES = 10
    D = Discriminator(n_classes=N_CLASSES)
    G = Generator(n_classes=N_CLASSES)
    model = CGAN(D=D, G=G, n_classes=N_CLASSES).to(DEVICE)
    D_optim = AdamW(D.parameters(), lr=args.LR)
    G_optim = AdamW(G.parameters(), lr=args.LR)

    train(
        n_classes=N_CLASSES,
        n_epochs=args.N_EPOCHS,
        train_dl=train_dl,
        model=model,
        D_optim=D_optim,
        G_optim=G_optim,
        gp_weight=args.GP_WEIGHT,
        save_dir=args.SAVE_DIR,
        device=DEVICE,
    )

if __name__ == "__main__":
    main()
