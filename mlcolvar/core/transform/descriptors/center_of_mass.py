import torch
import numpy as np

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import compute_distances_pairs, _resolve_descriptor_cell, sanitize_cell_shape, sanitize_positions_shape

from typing import List, Union

__all__ = ["CenterOfMass"]


def _normalize_groups(groups: Union[list, np.ndarray, torch.Tensor]) -> List[List[int]]:
    """Normalize `groups` to a list of list of ints.
    Accepts either a single flat group of atom indices or a list/array of groups.
    """
    if isinstance(groups, torch.Tensor):
        groups = groups.tolist()
    elif isinstance(groups, np.ndarray):
        groups = groups.tolist()
    else:
        groups = list(groups)

    if len(groups) == 0:
        raise ValueError("`groups` cannot be empty!")

    # Detect whether we were given a single flat group (list of ints) or multiple groups
    first = groups[0]
    if isinstance(first, torch.Tensor):
        first = first.tolist()
    if isinstance(first, np.ndarray):
        first = first.tolist()
    is_flat_group = not isinstance(first, (list, tuple))

    if is_flat_group:
        groups = [groups]

    normalized = []
    for group in groups:
        if isinstance(group, torch.Tensor):
            group = group.tolist()
        elif isinstance(group, np.ndarray):
            group = group.tolist()
        else:
            group = list(group)
        if len(group) == 0:
            raise ValueError("Each group must contain at least one atom!")
        normalized.append([int(a) for a in group])
    return normalized


class CenterOfMass(Transform):
    """
    Center(s) of mass of a group (or multiple groups) of atoms from their positions and masses.
    Handles Periodic Boundary Conditions by unwrapping each group's atoms with respect to a
    reference atom (the first one in the group) using the minimum image convention, so that
    groups split across the periodic boundary are still assigned a well defined center of mass.
    """

    def __init__(self,
                 groups: Union[list, List[list], np.ndarray, torch.Tensor],
                 masses: Union[list, np.ndarray, torch.Tensor],
                 n_atoms: int,
                 PBC: bool,
                 cell: Union[float, list, None] = None,
                 scaled_coords: bool = False) -> torch.Tensor:
        """Initialize a center of mass object.
        Can compute a single center of mass or multiple centers of mass based on the `groups` key.
        The cell size to be used for PBC and/or scaled coordinates needs to be provided.
        This can be done in one of two ways, exclusively:
        - Fixed cell (e.g., NVT simulations), at initialization, using the `cell` keyword, for a fixed cell only.
          This mode supports torchscript of the preprocessing module and can be used with the `PLUMED` interface.
        - Varying cells, at runtime (e.g., NPT simulations), using the `cell` entry in the `forward` method or adding the `cell` data in the dataset used for training.
          This mode **doesn't** support torchscript of the preprocessing module, as it is not supported in the `PLUMED` interface.

        Parameters
        ----------
        groups : Union[list, List[list], np.ndarray, torch.Tensor]
            Zero-based indices of the atoms belonging to the group(s) for which the center(s) of mass are computed.
            It can be:
            - A single flat list/array of atom indices: [a1, a2, ...] for a single center of mass
            - A list of lists/arrays of atom indices: [[a1, a2, ...], [b1, b2, ...], ...] for multiple centers of mass
        masses : Union[list, np.ndarray, torch.Tensor]
            Masses of all the `n_atoms` atoms in the system, indexed consistently with `groups`.
        n_atoms : int
            Number of atoms in the positions tensor used in the forward.
        PBC : bool
            Switch for Periodic Boundary Conditions use
        cell : Union[float, list, None]
            Dimensions of the real cell for fixed cell mode, orthorombic-like cells only.
            For varying cell mode, this argument must be left as None and the cell must be provided at runtime.
            Note that only fixed cell mode supports torchscript of the preprocessing module.
        scaled_coords : bool, optional
            Switch for coordinates scaled on cell's vectors use, by default False

        Returns
        -------
        torch.Tensor
            Center(s) of mass. Shape: [batch_size, n_groups * 3]
        """
        groups = _normalize_groups(groups)
        n_groups = len(groups)

        super().__init__(in_features=int(n_atoms * 3), out_features=int(n_groups * 3))

        masses = torch.as_tensor(masses, dtype=torch.float64).to(torch.get_default_dtype())
        if masses.numel() != n_atoms:
            raise ValueError(f"`masses` must contain exactly {n_atoms} elements (n_atoms), found {masses.numel()}.")
        if torch.any(masses <= 0):
            raise ValueError("All the `masses` must be strictly positive.")

        # Build, for every group, the (reference_atom, atom) pairs used to compute the PBC-safe
        # displacement of each atom in the group with respect to a reference atom (the first of the group).
        pairs = []
        pair_group_id = []
        pair_weight = []
        ref_indices = []
        for group_idx, group in enumerate(groups):
            ref_atom = group[0]
            ref_indices.append(ref_atom)
            for atom in group:
                pairs.append([ref_atom, atom])
                pair_group_id.append(group_idx)
                pair_weight.append(masses[atom])

        self.n_atoms = n_atoms
        self.n_groups = n_groups
        self.PBC = PBC
        self.scaled_coords = scaled_coords

        self.register_buffer('_pairs', torch.tensor(pairs, dtype=torch.long))
        self.register_buffer('_pair_group_id', torch.tensor(pair_group_id, dtype=torch.long))
        self.register_buffer('_pair_weight', torch.stack(pair_weight))
        self.register_buffer('_ref_indices', torch.tensor(ref_indices, dtype=torch.long))

        total_mass = torch.zeros(n_groups, dtype=self._pair_weight.dtype)
        total_mass.index_add_(0, self._pair_group_id, self._pair_weight)
        self.register_buffer('_total_mass', total_mass)

        default_cell = None if cell is None else sanitize_cell_shape(cell)
        self.register_buffer("default_cell", default_cell)

    def compute_center_of_mass(self, pos, cell=None):
        cell = _resolve_descriptor_cell(runtime_cell=cell,
                                       default_cell=self.default_cell,
                                       require_cell=self.PBC or self.scaled_coords,
                                    )
        pos_sanitized, batch_size = sanitize_positions_shape(pos, self.n_atoms)
        device = pos.device

        # PBC-safe displacement of every atom in a group w.r.t. its group's reference atom.
        # Shape: [batch_size, 3, n_pairs]
        disp = compute_distances_pairs(pos=pos,
                                       n_atoms=self.n_atoms,
                                       PBC=self.PBC,
                                       cell=cell,
                                       scaled_coords=self.scaled_coords,
                                       slicing_pairs=self._pairs,
                                       vector=True)

        pair_weight = self._pair_weight.to(disp.dtype)
        total_mass = self._total_mass.to(disp.dtype)

        weighted_disp = disp * pair_weight.view(1, 1, -1)

        # scatter-add the weighted displacements of each group's atoms into their group's slot
        com_disp = torch.zeros(batch_size, 3, self.n_groups, device=device, dtype=weighted_disp.dtype)
        com_disp.index_add_(2, self._pair_group_id, weighted_disp)
        com_disp = com_disp / total_mass.view(1, 1, -1)
        com_disp = com_disp.transpose(1, 2)  # [batch_size, n_groups, 3]

        ref_pos = pos_sanitized[:, self._ref_indices, :]  # [batch_size, n_groups, 3]
        com = ref_pos + com_disp
        return com.reshape(batch_size, -1)

    def forward(self, x: torch.Tensor, cell: Union[float, list, torch.Tensor] = None):
        x = self.compute_center_of_mass(x, cell=cell)
        return x


def test_center_of_mass():
    # 4 atoms, simple non-PBC test: two independent groups
    pos = torch.Tensor([[0.0, 0.0, 0.0,
                         2.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 3.0, 0.0]])
    pos.requires_grad = True

    # equal masses -> arithmetic mean
    masses = [1.0, 1.0, 1.0, 1.0]
    model = CenterOfMass(groups=[[0, 1], [2, 3]], masses=masses, n_atoms=4, PBC=False)
    out = model(pos)
    assert out.shape == (1, 6)
    ref = torch.Tensor([[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
    assert torch.allclose(out, ref, atol=1e-5)
    out.sum().backward()

    # single flat group (backward compatible input format)
    model = CenterOfMass(groups=[0, 1], masses=masses, n_atoms=4, PBC=False)
    out = model(pos)
    assert out.shape == (1, 3)
    assert torch.allclose(out, ref[:, :3], atol=1e-5)

    # weighted masses
    masses_weighted = [1.0, 3.0, 1.0, 1.0]
    model = CenterOfMass(groups=[[0, 1]], masses=masses_weighted, n_atoms=4, PBC=False)
    out = model(pos)
    # weighted mean along x: (1*0 + 3*2) / 4 = 1.5
    assert torch.allclose(out, torch.Tensor([[1.5, 0.0, 0.0]]), atol=1e-5)

    # mismatched masses length raises
    try:
        CenterOfMass(groups=[[0, 1]], masses=[1.0, 1.0, 1.0], n_atoms=4, PBC=False)
        raise AssertionError("Expected ValueError for wrong number of masses.")
    except ValueError as e:
        assert "masses" in str(e)

    # ---------------- PBC unwrapping test ----------------
    # A 2-atom "molecule" straddling the periodic boundary along x.
    cell = torch.Tensor([1.0, 1.0, 1.0])
    pos_pbc = torch.Tensor([[0.1, 0.0, 0.0,
                            0.9, 0.0, 0.0]])
    pos_pbc.requires_grad = True
    model = CenterOfMass(groups=[[0, 1]], masses=[1.0, 1.0], n_atoms=2, PBC=True, cell=cell)
    out = model(pos_pbc)
    # unwrapped: atom 1 is really at -0.1 relative to atom 0 -> COM x = 0.0
    assert torch.allclose(out, torch.Tensor([[0.0, 0.0, 0.0]]), atol=1e-5)
    out.sum().backward()

    # naive (non-PBC-aware) average would give the wrong result, sanity check that they differ
    naive_com = pos_pbc.detach().reshape(1, 2, 3).mean(dim=1)
    assert not torch.allclose(out, naive_com, atol=1e-3)

    # ---------------- scaled coordinates ----------------
    pos_scaled = (torch.clone(pos_pbc.detach()).reshape(1, 2, 3) / cell).reshape(1, 6)
    pos_scaled.requires_grad = True
    model = CenterOfMass(groups=[[0, 1]], masses=[1.0, 1.0], n_atoms=2, PBC=True, cell=cell, scaled_coords=True)
    out = model(pos_scaled)
    assert torch.allclose(out, torch.Tensor([[0.0, 0.0, 0.0]]), atol=1e-5)
    out.sum().backward()

    # runtime cell is allowed only when init cell is None
    model = CenterOfMass(groups=[[0, 1]], masses=[1.0, 1.0], n_atoms=2, PBC=True, cell=None)
    _ = model(pos_pbc, cell=cell)
    model = CenterOfMass(groups=[[0, 1]], masses=[1.0, 1.0], n_atoms=2, PBC=True, cell=cell)
    try:
        _ = model(pos_pbc, cell=cell)
        raise AssertionError("Expected ValueError when passing `cell` both at init and runtime.")
    except ValueError as e:
        assert "provided at initialization" in str(e)

    # ---------------- batched varying cell ----------------
    scales = torch.tensor([0.9, 1.0, 1.1], dtype=pos_pbc.dtype)
    cell_batched = torch.stack([cell * s for s in scales], dim=0)
    pos_pbc_batched = torch.cat([pos_pbc.detach() * s for s in scales], dim=0).clone().detach().requires_grad_(True)

    model = CenterOfMass(groups=[[0, 1]], masses=[1.0, 1.0], n_atoms=2, PBC=True, cell=None)
    out = model(pos_pbc_batched, cell=cell_batched)
    ref_batched_single = torch.cat(
        [model(pos_pbc_batched[i:i+1], cell=cell_batched[i]) for i in range(len(scales))],
        dim=0,
    )
    assert out.shape == (3, 3)
    assert torch.allclose(out, ref_batched_single, atol=1e-4)
    out.sum().backward()
