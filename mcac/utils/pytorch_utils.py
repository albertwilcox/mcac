import torch
import numpy as np


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def setup(idx=0):
    global TORCH_DEVICE
    if idx < 0:
        TORCH_DEVICE = torch.device('cpu')
    else:
        TORCH_DEVICE = torch.device('cuda', idx) if torch.cuda.is_available() else torch.device('cpu')
    print('Using', TORCH_DEVICE)


def torchify(*args, cls=torch.FloatTensor):
    out = []
    for x in args:
        if type(x) is not torch.Tensor and type(x) is not np.ndarray:
            x = np.array(x)
        if type(x) is not torch.Tensor:
            x = cls(x)
        out.append(x.to(TORCH_DEVICE))
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def numpify(*args):
    out = []
    for x in args:
        if x is None:
            out.append(x)
        else:
            out.append(x.detach().cpu().numpy())
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def soft_update(module, module_targ, tau=0.995):
    with torch.no_grad():
        for p, p_targ in zip(module.parameters(), module_targ.parameters()):
            p_targ.data.mul_(tau)
            p_targ.data.add_((1 - tau) * p.data)


def hard_update(module, module_targ):
    with torch.no_grad():
        for p, p_targ in zip(module.parameters(), module_targ.parameters()):
            p_targ.data.copy_(p.data)
