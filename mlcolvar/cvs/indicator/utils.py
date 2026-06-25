import torch
from torch import nn
import copy
from mlcolvar.cvs.indicator import IndicatorProduction

__all__ = ["IndicatorBiasModel"]

class IndicatorBiasModel(torch.nn.Module):
    """Gradient-norm-based bias model for estimating the PLUMED LAMBDA parameter.

    bias(x) = -l * (log(||grad q(x)||^2 + e) - log(e))
    """

    def __init__(self, input_model, e=1e-6, l=1):
        super().__init__()
        self.input_model = input_model
        self.l = l
        if not isinstance(e, torch.Tensor):
            e = torch.tensor([e], dtype=torch.float32)
        self.e = e.to("cpu")

    def forward(self, x):
        x = x.detach().float().requires_grad_(True)
        q = self.input_model(x)
        grad_outputs = torch.ones_like(q)
        grads = torch.autograd.grad(q, x, grad_outputs, retain_graph=True)[0]
        grads_sq = torch.sum(torch.pow(grads, 2), dim=1)
        return -self.l * (torch.log(grads_sq + self.e) - torch.log(self.e))
    