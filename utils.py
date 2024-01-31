import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import os
import numpy as np
import random


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def denorm(tensor, mean, std):
    return TF.normalize(
        tensor, mean=- np.array(mean) / np.array(std), std=1 / np.array(std),
    )


def image_to_grid(image, mean, std, n_cols, padding=1):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor, mean=mean, std=std)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=padding)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def to_pil(img):
    if not isinstance(img, Image.Image):
        image = Image.fromarray(img)
        return image
    else:
        return img


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(image).save(str(path), quality=100)
