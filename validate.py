import torch
from loader import lyft_loader


def run(model, eval_params, dataloader):
    # TODO: Get the validation dataloader Implement the validation function
    model.eval()

    with torch.no_grad():
        pass