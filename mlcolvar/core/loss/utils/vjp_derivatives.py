import gc
from typing import Optional, Union

import torch

from mlcolvar.data import DictDataset
from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import sanitize_positions_shape
from mlcolvar.core.loss.utils.smart_derivatives import create_smart_dataset

__all__ = ["VJPDerivatives"]


class VJPDerivatives(torch.nn.Module):
    """
    On-demand, memory-vjp derivatives via vector-Jacobian products (VJP).

    Contract:
    - setup(dataset, descriptor_function, n_atoms, ...) -> returns a dataset with descriptors as data and a ref_idx mapping
    - forward(right, ref_idx) -> returns d(out)/d(pos) by computing VJP per batch element

    Key differences vs Smart/vjp:
    - Does not precompute/store d(desc)/d(pos). Instead, at forward time, for the requested indices it computes
      descriptor values and uses autograd to evaluate sum(desc * right) VJP w.r.t. positions. This keeps memory at O(batch).
    - It preserves gradient flow to `right` by building the computational graph (create_graph=True) during VJP.
    """

    def __init__(
        self,
        setup_device: str = "cpu",
        desc_dtype: Optional[torch.dtype] = None,
        retain_graph: bool = True,
    ):
        super().__init__()
        self.setup_device = setup_device
        self.desc_dtype = desc_dtype
        # Some training loops (e.g., Lightning closures, gradient accumulation, or multiple backward passes)
        # may backward through the same graph more than once. When True, retain the inner VJP graph to avoid
        # "Trying to backward through the graph a second time" errors. For memory saving, set to False if
        # your training loop guarantees a single backward per forward.
        self.retain_graph = retain_graph
        self._check_setup = False
        self._descriptor_device: Optional[torch.device] = None

    def setup(
        self,
        dataset: DictDataset,
        descriptor_function: Transform,
        n_atoms: int,
        separate_boundary_dataset: bool = False,
        positions_noise: float = 0.0,
        descriptors_batch_size: Optional[int] = None,
    ) -> DictDataset:
        """Prepare the VJP backend: cache variational positions and build the descriptor dataset.

        Returns a DictDataset like Smart/vjp with:
        - data: descriptors
        - labels, weights: copied from input
        - ref_idx: mapping from dataset indices to variational subset [0..N_var)
        """
        self.n_atoms = n_atoms
        self.descriptor_function: Transform = descriptor_function

        # optional jitter (to avoid perfectly identical coords that cause zero grads)
        if positions_noise > 0:
            dataset["data"] = dataset["data"] + torch.rand_like(dataset["data"]) * positions_noise

        pos = dataset["data"]
        labels = dataset["labels"]
        pos = sanitize_positions_shape(pos=pos, n_atoms=n_atoms)[0]
        device = pos.device

        if separate_boundary_dataset:
            mask_var = labels.squeeze() > 1
            if mask_var.sum() == 0:
                raise ValueError(
                    "No points left after separating boundary and variational datasets.\n"
                    "If using only unbiased data: set separate_boundary_dataset=False or don't use VJPDerivatives."
                )
        else:
            mask_var = torch.ones_like(labels.squeeze(), dtype=torch.bool)

        # cache variational positions on the chosen setup_device (CPU by default)
        self._var_pos = pos[mask_var].detach().to(self.setup_device)

        # Build descriptors for dataset (streamed to limit memory)
        if descriptors_batch_size in (None, -1):
            batch_size = int(self._var_pos.shape[0])
        else:
            batch_size = int(descriptors_batch_size)
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        desc_chunks = []
        n_batches = (self._var_pos.shape[0] + batch_size - 1) // batch_size
        for b in range(n_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, self._var_pos.shape[0])
            if s >= e:
                continue
            batch_pos = self._var_pos[s:e].to(device)
            with torch.no_grad():
                batch_desc = descriptor_function(batch_pos)
            desc_chunks.append(batch_desc.detach().to("cpu"))
            del batch_pos, batch_desc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(desc_chunks) == 0:
            raise RuntimeError("Descriptor function returned empty output for all batches.")

        desc_var = torch.cat(desc_chunks, dim=0)

        # If separate_boundary_dataset, compute descriptors for boundary too to fill full dataset
        if separate_boundary_dataset:
            # Batched computation for boundary frames to avoid a single giant forward
            n_bound = int((~mask_var).sum().item())
            if n_bound > 0:
                bound_chunks = []
                n_batches_bound = (n_bound + batch_size - 1) // batch_size
                for b in range(n_batches_bound):
                    s, e = b * batch_size, min((b + 1) * batch_size, n_bound)
                    if s >= e:
                        continue
                    batch_pos = pos[~mask_var][s:e].to(device)
                    with torch.no_grad():
                        batch_desc = descriptor_function(batch_pos)
                    bound_chunks.append(batch_desc.detach().to("cpu"))
                    del batch_pos, batch_desc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if len(bound_chunks) == 0:
                    # No boundary descriptors produced (degenerate), fall back to zeros of the same feature size as var
                    feat = int(desc_var.shape[-1])
                    desc_not_var = torch.zeros((0, feat), dtype=desc_var.dtype)
                else:
                    desc_not_var = torch.cat(bound_chunks, dim=0)

                # Assemble full descriptor tensor on CPU to match dataset order
                feat = int((desc_var.shape[-1] if desc_var.numel() > 0 else desc_not_var.shape[-1]))
                desc = torch.zeros((len(dataset), feat), dtype=(desc_var.dtype if desc_var.numel() > 0 else desc_not_var.dtype))
                desc[mask_var] = desc_var
                if n_bound > 0:
                    desc[~mask_var] = desc_not_var
            else:
                # No boundary frames: create full-sized tensor and place variational descriptors
                feat = int(desc_var.shape[-1])
                desc = torch.zeros((len(dataset), feat), dtype=desc_var.dtype)
                desc[mask_var] = desc_var
        else:
            desc = desc_var

        if self.desc_dtype is not None and desc.numel() > 0:
            desc = desc.to(self.desc_dtype)

        vjp_dataset = create_smart_dataset(desc=desc.detach().to(device), dataset=dataset, separate_boundary_dataset=separate_boundary_dataset)

        # remember how many variational entries we have for bounds checking
        self._n_var = int(self._var_pos.shape[0])
        self._check_setup = True
        return vjp_dataset

    def _ensure_descriptor_device(self, device: torch.device):
        # Move descriptor function to device lazily if it supports .to
        if getattr(self.descriptor_function, "to", None) is not None:
            # only if needed
            if self._descriptor_device != device:
                try:
                    self.descriptor_function = self.descriptor_function.to(device)
                except Exception:
                    pass
                self._descriptor_device = device

    def forward(self, x: torch.Tensor, ref_idx: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """Apply chain rule d(out)/d(pos) = J_{desc,pos}^T @ d(out)/d(desc) via autograd VJP.

        Supports both single-output and multi-output right-hand shapes:
        - x: (B, D)
        - x: (B, D, O)
        Returns:
        - (B, n_atoms, 3) or (B, n_atoms, 3, O)
        """
        if not self._check_setup:
            raise RuntimeError("VJPDerivatives must be setup() before calling forward().")

        if ref_idx is None:
            ref_idx = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        else:
            ref_idx = ref_idx.to(x.device).long()

        if ref_idx.numel() == 0 or x.size(0) == 0:
            if x.dim() == 2:
                return torch.zeros((0, self.n_atoms, 3), dtype=x.dtype, device=x.device)
            else:
                return torch.zeros((0, self.n_atoms, 3, x.shape[-1]), dtype=x.dtype, device=x.device)

        # gather the relevant variational positions
        if (ref_idx.min() < 0) or (ref_idx.max() >= self._n_var):
            raise IndexError("ref_idx contains indices outside the variational subset range.")

        pos_sub = self._var_pos[ref_idx.cpu()].to(x.device)
        pos_sub = pos_sub.detach().requires_grad_(True)

        # ensure descriptor function and buffers are on the correct device
        self._ensure_descriptor_device(x.device)

        desc_sub = self.descriptor_function(pos_sub)
        # Ensure dtype/device alignment with the incoming right-hand tensor
        if desc_sub.dtype != x.dtype:
            desc_sub = desc_sub.to(dtype=x.dtype)

        if x.dim() == 2:
            # single-output case: VJP with weight x
            s = (desc_sub * x).sum()
            # Retain graph based on configuration to support potential multiple backward passes.
            grad_pos = torch.autograd.grad(s, pos_sub, retain_graph=self.retain_graph, create_graph=True)[0]
            return grad_pos

        # multi-output case: loop over outputs to retain differentiability w.r.t x
        outs = []
        O = x.shape[-1]
        for i in range(O):
            s_i = (desc_sub * x[:, :, i]).sum()
            # Retain graph either until the last output (for within-call reuse) or fully if configured.
            g_i = torch.autograd.grad(
                s_i,
                pos_sub,
                retain_graph=(self.retain_graph or i < O - 1),
                create_graph=True,
            )[0]
            outs.append(g_i)
        out = torch.stack(outs, dim=-1)
        return out


def _test_vjp_derivatives_minimal():
    """Minimal parity check against autograd for a tiny system and single/multi-output right."""
    from mlcolvar.core.transform import PairwiseDistances

    torch.manual_seed(0)
    n_atoms = 3
    cell = torch.Tensor([3.1])
    ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms, PBC=True, cell=cell, scaled_coords=False)

    # Build a small dataset
    pos = torch.randn(6, n_atoms * 3)
    labels = torch.tensor([2, 2, 0, 3, 1, 2])  # mix boundary/var labels
    weights = torch.ones_like(labels)
    dataset = DictDataset({"data": pos, "labels": labels, "weights": weights})

    vjp = VJPDerivatives()
    vjp_dataset = vjp.setup(dataset=dataset, descriptor_function=ComputeDescriptors, n_atoms=n_atoms, separate_boundary_dataset=True)

    # pick a batch of the variational subset using ref_idx
    mask_var = labels > 1
    ref_idx = vjp_dataset["ref_idx"][mask_var]

    # ground truth via explicit autograd
    pos_var = pos.view(-1, n_atoms, 3)[mask_var].detach().requires_grad_(True)
    desc = ComputeDescriptors(pos_var)
    right = torch.randn(desc.shape[0], desc.shape[1])
    s = (desc * right).sum()
    gt = torch.autograd.grad(s, pos_var)[0]

    # VJPDerivatives result
    got = vjp(right, ref_idx)
    assert torch.allclose(got, gt, atol=1e-6)

    # multi-output: recompute graph to avoid reuse issues
    pos_var2 = pos.view(-1, n_atoms, 3)[mask_var].detach().requires_grad_(True)
    desc2 = ComputeDescriptors(pos_var2)
    right2 = torch.randn(desc2.shape[0], desc2.shape[1], 2)
    s2a = (desc2 * right2[:, :, 0]).sum()
    s2b = (desc2 * right2[:, :, 1]).sum()
    gt2a = torch.autograd.grad(s2a, pos_var2, retain_graph=True)[0]
    gt2b = torch.autograd.grad(s2b, pos_var2)[0]
    gt2 = torch.stack([gt2a, gt2b], dim=-1)
    got2 = vjp(right2, ref_idx)
    assert torch.allclose(got2, gt2, atol=1e-6)


def test_train_with_vjp_derivatives():
    from mlcolvar.core.transform import PairwiseDistances
    from mlcolvar.data import DictModule, DictDataset
    from mlcolvar.cvs import Committor, Generator
    from mlcolvar.cvs.committor.utils import initialize_committor_masses
    from mlcolvar.core.loss.utils.vjp_derivatives import VJPDerivatives
    from mlcolvar.explain.sensitivity import sensitivity_analysis

    import lightning

    # committor
    # full atoms with all distances
    n_atoms = 10
    pos = torch.Tensor([[ 1.4970,  1.3861, -0.0273, -1.4933,  1.5070, -0.1133, -1.4473, -1.4193,
                        -0.0553,  1.4940,  1.4990, -0.2403,  1.4780, -1.4173, -0.3363, -1.4243,
                        -1.4093, -0.4293,  1.3530, -1.4313, -0.4183,  1.3060,  1.4750, -0.4333,
                        1.2970, -1.3233, -0.4643,  1.1670, -1.3253, -0.5354]])
    
    pos = pos.repeat(200, 1)
    labels = torch.arange(0, 5, dtype=torch.float32).unsqueeze(-1).repeat(40,1).sort()[0]
    weights = torch.ones_like(labels)
    atomic_masses = initialize_committor_masses(atom_types=[0, 0, 1, 2, 0, 0, 0, 1, 2, 0], 
                                            masses=[12.011, 15.999, 14.007])

    dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': weights})

    cell = torch.Tensor([3.0233])

    ComputeDescriptors = PairwiseDistances(n_atoms=n_atoms,
                            PBC=True,
                            cell=cell,
                            scaled_coords=False,
                            slicing_pairs=None)
    
    vjp_derivatives = VJPDerivatives(desc_dtype=torch.float32)
    vjp_dataset = vjp_derivatives.setup(dataset=dataset, 
                                        descriptor_function=ComputeDescriptors,
                                        n_atoms=n_atoms,
                                        separate_boundary_dataset=True,
                                        descriptors_batch_size=25)
    
    datamodule = DictModule(dataset=vjp_dataset, lengths=[0.8, 0.2], batch_size=80)
    
    model = Committor(layers=[45, 10, 1],
                      atomic_masses=atomic_masses,
                      alpha=1,
                      separate_boundary_dataset=True,
                      descriptors_derivatives=vjp_derivatives 
                      )
    
    trainer = lightning.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    
    trainer.fit(model, datamodule)

    # check that sensitivity works
    sensitivity_analysis(model=model, dataset=vjp_dataset)

    # Generator
    kT = 2.49432

    # create friction tensor
    #### This part should be made easier using committor utils TODO
    masses = torch.Tensor([ 12.011, 12.011, 15.999, 14.0067, 12.011, 12.011, 12.011, 15.999, 14.0067, 12.011])
    gamma = 1 / 0.05
    friction = kT / (gamma*masses)
    ref_weights = torch.ones(len(pos))

    dataset = DictDataset({'data' : pos, 'labels' : labels, 'weights': ref_weights})


if __name__ == "__main__":
    _test_vjp_derivatives_minimal()
    test_train_with_vjp_derivatives()
