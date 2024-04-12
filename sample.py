import sys
sys.path.insert(0, "/home/jbkim/Desktop/workspace/Conditional-WGAN-GP")
import torch

from utils import get_device
from model import ConditionalWGANGP, Generator


DEVICE = get_device()
MODEL_PARAMS = "ML-API/assignment3/resources/cwgan_gp_mnist.pth"
state_dict = torch.load(str(MODEL_PARAMS), map_location=DEVICE)
# model = ConditionalWGANGP()
model = Generator(n_classes=10, latent_dim=100, hidden_dim=32).to(DEVICE)
model.load_state_dict(state_dict)