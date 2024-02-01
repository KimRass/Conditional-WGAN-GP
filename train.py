# References:
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/04_gan/03_cgan/cgan.ipynb

import torch
from torch.optim import AdamW
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings

from utils import get_device, set_seed, image_to_grid, save_image
from data import get_mnist_dls
from model import Discriminator, Generator, ConditionalWGANGP


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--n_epochs", type=int, default=50, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--lr", type=float, default=0.0002, required=False)

    parser.add_argument("--d_hidden_dim", type=int, default=32, required=False)
    parser.add_argument("--g_latent_dim", type=int, default=100, required=False)
    parser.add_argument("--g_hidden_dim", type=int, default=32, required=False)
    parser.add_argument("--gp_weight", type=float, default=10, required=False)
    parser.add_argument("--n_d_updates", type=int, default=3, required=False)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def train(n_classes, n_epochs, train_dl, model, gp_weight, save_dir, n_D_updates, device):
    fixed_latent_vec = model.sample_latent_vec(batch_size=n_classes * 10, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed_label = torch.arange(
            n_classes, dtype=torch.int32, device=device,
        ).repeat_interleave(10)

    for epoch in range(1, n_epochs + 1):
        cum_D_loss = 0
        cum_G_loss = 0
        for real_image, label in tqdm(train_dl, leave=False):
            D_loss, G_loss = model.train_single_step(
                real_image=real_image,
                label=label,
                gp_weight=gp_weight,
                device=device,
                n_D_updates=n_D_updates,
            )
            cum_D_loss += D_loss
            cum_G_loss += G_loss
        train_D_loss = cum_D_loss / len(train_dl)
        train_G_loss = cum_G_loss / len(train_dl)

        log = f"""[ {epoch}/{n_epochs} ]"""
        log += f"[ D loss: {train_D_loss:.2f} ]"
        log += f"[ G loss: {train_G_loss:.2f} ]"
        print(log)

        gen_image = model.generate_image(latent_vec=fixed_latent_vec, label=fixed_label)
        gen_grid = image_to_grid(gen_image, mean=0.5, std=0.5, n_cols=n_classes)
        save_image(gen_grid, Path(save_dir)/f"epoch_{epoch}.jpg")

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.G.state_dict(), str(Path(save_dir)/f"epoch_{epoch}.pth"))


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    train_dl, _ = get_mnist_dls(
        data_dir=args.DATA_DIR, batch_size=args.BATCH_SIZE, n_cpus=0,
    )

    N_CLASSES = 10
    D = Discriminator(n_classes=N_CLASSES, hidden_dim=args.D_HIDDEN_DIM).to(DEVICE)
    G = Generator(
        n_classes=N_CLASSES, latent_dim=args.G_LATENT_DIM, hidden_dim=args.G_HIDDEN_DIM,
    ).to(DEVICE)
    D_optim = AdamW(D.parameters(), lr=args.LR)
    G_optim = AdamW(G.parameters(), lr=args.LR)
    model = ConditionalWGANGP(D=D, G=G, D_optim=D_optim, G_optim=G_optim).to(DEVICE)

    train(
        n_classes=N_CLASSES,
        n_epochs=args.N_EPOCHS,
        train_dl=train_dl,
        model=model,
        gp_weight=args.GP_WEIGHT,
        save_dir=args.SAVE_DIR,
        n_D_updates=args.N_D_UPDATES,
        device=DEVICE,
    )

if __name__ == "__main__":
    main()
