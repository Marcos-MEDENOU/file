import torch
import torch.nn as nn


model = torch.load("save/saved_weights.pt")


model.eval()
