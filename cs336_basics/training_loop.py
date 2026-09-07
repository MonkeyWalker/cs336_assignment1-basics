import numpy as np
import torch





def save_checkpoint(model, optimizer, iteration, out):
    save_dict = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(save_dict, out)



def load_checkpoint(src, model, optimizer) -> int:

    saved_dict =  torch.load(src)
    model.load_state_dict(saved_dict["model"])
    optimizer.load_state_dict(saved_dict["optimizer"])
    return saved_dict["iteration"]

