from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss.generator_loss import GeneratorLoss
from mlcolvar.cvs.generator import compute_eigenfunctions
from mlcolvar.core.loss.utils.smart_derivatives import SmartDerivatives
from mlcolvar.core.loss.utils.vjp_derivatives import VJPDerivatives
from typing import Union, Tuple
import lightning
import torch

__all__ = ["IndicatorTraining", "IndicatorProduction"]

class IndicatorTraining(BaseCV, lightning.LightningModule):
    """Train an indicator function to be 1 inside the known domain and different from 1 outside.
        We obtain this indicator from the Generator framework, learning only the first eigenfunction of the Fokker-Planck operator.
    """

    DEFAULT_BLOCKS = ["nn"]

    def __init__(self, layers, eta, alpha=20, friction=None, descriptors_derivatives: Union[SmartDerivatives, VJPDerivatives, torch.Tensor] = None, options=None, **kwargs):
        """Initialize and Indicator for training.

        Parameters
        ----------
        layers : list of int
            List specifying the number of neurons in each layer of the neural network.
        eta : float
            Hyperparameter for the shift to define the resolvent, i.e., $(\eta I-_mathcal{L})^{-1}$
        alpha : int, optional
            Regularization parameter, by default 20
        friction : float, optional
            Langevin friction coefficients, by default None
        descriptors_derivatives : Union[SmartDerivatives, VJPDerivatives, torch.Tensor], optional
            Derivatives of descriptors wrt atomic positions (if used) to speed up calculation of gradients, by default None. 
            Can be either:
                - A `SmartDerivatives` object to save both memory and time, see also mlcolvar.core.loss.committor_loss.SmartDerivatives
                - A `VJPDerivatives` object to save both memory and time, see also mlcolvar.core.loss.utils.vjp_derivatives.VJPDerivatives
                - A torch.Tensor with the derivatives to save time, memory-wise could be less efficient
        options : dict, optional
            Additional options for the neural network, by default None
        """
        super().__init__(model=layers, **kwargs)
        self.loss_fn = GeneratorLoss(eta=eta, alpha=alpha, friction=friction, r=1, descriptors_derivatives=descriptors_derivatives)
        self.r = 1
        self.eta = eta
        self.friction = friction
        self.cell = None
        self.evecs = None
        self.evals = None
        options = self.parse_options(options or {})
        o = "nn"
        if "activation" not in options[o]:
            options[o]["activation"] = "tanh"
        self.nn = FeedForward(layers, **options[o])

    def compute_eigenfunctions(self, dataset, friction=None, eta=None, cell=None,
                               tikhonov_reg=1e-4, recompute=False):
        if friction is None:
            friction = self.friction
        if eta is None:
            eta = self.eta
        if cell is None:
            cell = self.cell
        if recompute or self.evecs is None:
            dataset["data"].requires_grad_(True)
            output = self.forward(dataset["data"])

            # Check for descriptor derivatives in dataset regardless of container type.
            try:
                desc_derivs = dataset["derivatives"]
            except (KeyError, IndexError):
                desc_derivs = None
            eigenfunctions, evals, evecs = compute_eigenfunctions(
                input=dataset["data"],
                output=output,
                weights=dataset["weights"],
                r=self.r,
                eta=eta,
                friction=friction,
                tikhonov_reg=tikhonov_reg,
                descriptors_derivatives=desc_derivs,
            )
            self.evals = evals
            self.evecs = evecs
            return eigenfunctions, evals, evecs
        else:
            eigenfunctions = self.forward(dataset["data"]) @ self.evecs.real
            return eigenfunctions, self.evals, self.evecs

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.nn(x))

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        torch.set_grad_enabled(True)
        # =================get data===================
        x = train_batch["data"]
        # check data are have shape (n_data, -1)
        x = x.reshape((x.shape[0], -1))

        x.requires_grad = True

        weights = train_batch["weights"]
        if "derivatives" in train_batch.keys():
            derivatives = train_batch["derivatives"]
        else:
            derivatives = None

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        q = self.forward(x)
        # ===================loss=====================
        if self.training:
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights, derivatives)
        else:
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights, derivatives)
        # ====================log=====================
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_ef, on_epoch=True)
        self.log(f"{name}_loss_ortho", loss_ortho, on_epoch=True)
        return loss
    

class IndicatorProduction(BaseCV, lightning.LightningModule):
    """
    Trivial Indicator CV: forward_cv = exp(-nn(x)).

    NN weights are copied from a trained IndicatorTraining.
    Eigenvector coefficients are stored but the NN already encodes the right direction.
    Used for lightweight torchscript export.
    >>> trivial = IndicatorProduction(
    >>>    layers=layers, eta=eta, r=r, alpha=alpha,
    >>>    friction=friction.cpu(), coeffs=coeffs[:, 0]
    >>> ).to("cpu").to(torch.float32)

    >>> # Copy NN weights from trained model
    >>> trivial.nn = copy.deepcopy(model.nn).to("cpu").to(torch.float32)
    """

    DEFAULT_BLOCKS = ["nn"]

    def __init__(self, layers, eta, alpha=20, friction=None, options=None, coeffs=None, **kwargs):
        super().__init__(model=layers, **kwargs)
        self.loss_fn = GeneratorLoss(eta=eta, alpha=alpha, friction=friction, r=1)
        options = self.parse_options(options or {})
        o = "nn"
        if "activation" not in options[o]:
            options[o]["activation"] = "tanh"
        self.nn = FeedForward(layers, **options[o])
        self.coeffs = coeffs

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.nn(x))