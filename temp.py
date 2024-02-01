import torch

state_dict = torch.load("/Users/jongbeomkim/Documents/datasets/cgan/temp/epoch_8.pth")
state_dict.keys()

state_dict["main.0.weight"].shape, state_dict["main.1.weight"].shape, state_dict["main.9.weight"].shape