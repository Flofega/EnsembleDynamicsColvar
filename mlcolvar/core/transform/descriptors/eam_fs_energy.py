import torch
from typing import List, Tuple, Optional, Dict, Union
from mlcolvar.core.transform import Transform
import numpy as np

__all__ = ["EAM_FS"]

def _to_cell_matrix_single(cell_entry, device=None, dtype=None):
    """Convert a single cell representation to a 3x3 lattice matrix.

    Supported formats:
    - (3,) lengths: [Lx, Ly, Lz] -> diag(Lx, Ly, Lz)
    - (6,) LAMMPS bounds: [xlo, xhi, ylo, yhi, zlo, zhi] -> diag(xhi-xlo, yhi-ylo, zhi-zlo)
    - (9,) flattened 3x3 (row-major) -> reshape to (3,3)
    - (3,3) lattice matrix (rows are lattice vectors)
    """
    tt = torch.as_tensor(cell_entry, device=device, dtype=dtype if dtype is not None else torch.get_default_dtype())
    if tt.ndim == 2 and tt.shape == (3, 3):
        return tt
    flat = tt.view(-1)
    n = flat.numel()
    if n == 3:
        return torch.diag(flat)
    elif n == 6:
        xlo, xhi, ylo, yhi, zlo, zhi = flat.tolist()
        L = torch.tensor([xhi - xlo, yhi - ylo, zhi - zlo], device=tt.device, dtype=tt.dtype)
        return torch.diag(L)
    elif n == 9:
        return flat.view(3, 3)
    else:
        raise ValueError("Unsupported cell format; expected 3-lengths, 6-bounds, 9-matrix or 3x3")

def _prepare_cell_matrix(cell, B, device, dtype):
    """Prepare lattice matrix/matrices for batch.

    If `cell` is a per-frame list/array of length B, stack into (B,3,3),
    otherwise return a single (3,3) matrix broadcastable across batch.
    """
    # per-batch list/array
    if isinstance(cell, (list, tuple)) and len(cell) > 0 and isinstance(cell[0], (list, tuple, np.ndarray, torch.Tensor)):
        if len(cell) != B:
            raise ValueError("Per-batch cell list/array must have length equal to batch size")
        mats = [ _to_cell_matrix_single(cell[i], device=device, dtype=dtype) for i in range(B) ]
        return torch.stack(mats, dim=0)  # (B,3,3)
    # torch tensor with leading batch dim
    if isinstance(cell, torch.Tensor) and cell.ndim >= 2 and cell.shape[0] == B:
        # convert each entry to 3x3 then stack
        mats = [ _to_cell_matrix_single(cell[i], device=device, dtype=dtype) for i in range(B) ]
        return torch.stack(mats, dim=0)
    # single cell entry
    return _to_cell_matrix_single(cell, device=device, dtype=dtype)  # (3,3)

def _min_image_general(dr: torch.Tensor, cell_mat: torch.Tensor):
    """Apply minimum image using lattice matrices.

    dr: (B, N, N, 3)
    cell_mat: (3,3) or (B,3,3) with rows as lattice vectors (Cartesian)
    """
    B = dr.shape[0]
    if cell_mat.ndim == 2:
        cell_mat = cell_mat.unsqueeze(0).expand(B, -1, -1)
    inv = torch.linalg.inv(cell_mat)  # (B,3,3)
    # reshape for batched matmul
    drb = dr.view(B, -1, 3)
    s = torch.bmm(drb, inv)  # fractional coordinates
    s_wrapped = s - torch.round(s)
    dr_min = torch.bmm(s_wrapped, cell_mat).view(B, dr.shape[1], dr.shape[2], 3)
    return dr_min

def _pairwise_displacements(pos: torch.Tensor, box):
    rij = pos.unsqueeze(2) - pos.unsqueeze(1)
    if box is not None:
        B = pos.shape[0]
        cell_mat = _prepare_cell_matrix(box, B, device=pos.device, dtype=pos.dtype)
        rij = _min_image_general(rij, cell_mat)
    return rij


class EAM_FS(Transform):
    """Descriptor that reads an EAM/FS single-element file and evaluates energies mirroring the LAMMPS implementation (PyTorch/autograd compatible).
    Supports NPT simulations.
    Notes:
    - Reads .eam.fs files with single element support (View Lammps doc for file format specs).
    - Uses linear interpolation on uniform grids (embedding F(rho), rho(r), z2(r)=r*phi(r)) this introduces a small but usually negligable error compared to the correct/lammps implementation.
    - Vectorized over batches and atoms. Expects positions shaped (B, N, 3).
    - Returns per-atom energies with shape (B, N, 1).

    Assumptions / differences vs PLUMED C++:
    - The potential grids are treated as fixed tensors (no trainable params).
    - Interpolation is implemented with torch ops so gradients flow w.r.t. positions.
    - Supports orthorhombic and triclinic boxes: pass 3-lengths, 6 LAMMPS bounds, 9-matrix, 3x3, or per-batch lists.
    """

    def __init__(self, filename, n_atoms: int, cell, r_scale: float = 1.0, e_scale: float = 1.0):
        super().__init__(in_features=int(n_atoms * 3), out_features=n_atoms)
        self.n_atoms = int(n_atoms)
        # Register cell as buffer so it moves with .to(device)
        if isinstance(cell, torch.Tensor):
            self.register_buffer('cell', cell.clone())
        elif cell is not None:
            self.register_buffer('cell', torch.tensor(cell, dtype=torch.float32))
        else:
            self.register_buffer('cell', None)
        self.r_scale = float(r_scale)
        self.e_scale = float(e_scale)

        # placeholders to be set by read
        self.nrho = None
        self.nr = None
        self.drho = None
        self.dr = None
        self.cut = None

        # raw numpy arrays (then converted to torch tensors on demand)
        self.frhonp = None
        self.rhornp = None
        self.z2rnp = None

        # read the file and build grids
        self._read_fs_file(filename)

        # convert to torch tensors with default dtype
        dtype = torch.get_default_dtype()
        self.frho = torch.as_tensor(self.frhonp, dtype=dtype)
        self.rhor = torch.as_tensor(self.rhornp, dtype=dtype)
        self.z2r = torch.as_tensor(self.z2rnp, dtype=dtype)

        # cutoff in MD units (file cutoff is in FS units)
        self.cutoff_md = float(self.cut) / self.r_scale

    def _read_fs_file(self, path_or_file):
        # Read lines; accept file object or path
        if hasattr(path_or_file, "read"):
            lines = path_or_file.readlines()
        else:
            with open(path_or_file, "r") as fh:
                lines = fh.readlines()

        if len(lines) < 6:
            raise ValueError("EAM/FS file seems too short")

        # Skip first 3 header lines
        # Line 4: elements line: nelements and names
        elem_line = lines[3].strip()
        parts = elem_line.split()
        if len(parts) < 1:
            raise ValueError("EAM_FS: invalid elements line")
        nelements = int(parts[0])
        if nelements != 1:
            raise ValueError("EAM_FS: only single-element FS files supported in this implementation")
        # names = parts[1:1+nelements]  # unused but parsed

        # Line 5: grid line
        grid_line = lines[4].strip()
        gparts = grid_line.split()
        if len(gparts) < 5:
            raise ValueError("EAM_FS: grid line missing entries")
        self.nrho = int(gparts[0])
        self.drho = float(gparts[1])
        self.nr = int(gparts[2])
        self.dr = float(gparts[3])
        self.cut = float(gparts[4])

        # Line 6: element info line — skip entirely (may contain strings like 'bcc')
        # Array values start from line 7 onward
        array_lines = lines[6:]

        # Collect numeric tokens robustly, ignoring non-numeric tokens
        numeric_vals = []
        for L in array_lines:
            for tok in L.split():
                try:
                    numeric_vals.append(float(tok))
                except ValueError:
                    # ignore non-numeric tokens (e.g., 'bcc') if present
                    continue

        n_needed = self.nrho + self.nr + self.nr
        if len(numeric_vals) < n_needed:
            raise ValueError("EAM_FS: not enough numeric table values after header")

        import numpy as np
        self.frhonp = np.array(numeric_vals[0:self.nrho], dtype=np.float64)
        self.rhornp = np.array(numeric_vals[self.nrho:self.nrho + self.nr], dtype=np.float64)
        self.z2rnp = np.array(numeric_vals[self.nrho + self.nr:self.nrho + 2 * self.nr], dtype=np.float64)

    @staticmethod
    def _interp_uniform(y: torch.Tensor, dx: float, x: torch.Tensor):
        """Uniform-grid linear interpolation with simple end-segment extrapolation.

        y: 1D tensor of length n
        dx: grid spacing (float)
        x: arbitrary-shaped tensor of query points (same device/dtype as y)

        returns: tensor of same shape as x with interpolated values
        """
        # ensure shapes/dtypes on same device
        device = y.device
        dtype = y.dtype
        x = x.to(device=device, dtype=dtype)

        n = y.numel()
        if n < 2:
            return y.view(1)[0] * torch.ones_like(x)

        t = x / float(dx)
        i = torch.floor(t).to(torch.long)
        # clamp index to valid interval [0, n-2]
        i_clamped = torch.clamp(i, 0, n - 2)
        # fractional part (can be <0 for negative x, and >1 for extrapolation beyond last bin)
        frac = t.to(dtype) - i_clamped.to(dtype)

        # gather y[i] and y[i+1]
        i_flat = i_clamped.view(-1)
        y0 = y[i_flat].view(i_clamped.shape)
        y1 = y[(i_flat + 1)].view(i_clamped.shape)

        val = y0 + (y1 - y0) * frac
        return val

    def calculate(self, config: torch.Tensor, cell: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute per-atom EAM/FS energies for a batch of configurations.

        Parameters
        ----------
        config : torch.Tensor
            Positions shaped (B, N, 3) where N == self.n_atoms.
        cell : torch.Tensor, optional
            Runtime cell for NPT simulations. If provided, overrides self.cell.
            Can be: (3,), (6,), (9,), (3,3), or per-batch (B, ...).

        Returns
        -------
        torch.Tensor
            Per-atom energies shaped (B, N, 1).
        """
        if config.dim() != 3 or config.shape[1] != self.n_atoms or config.shape[2] != 3:
            raise ValueError(f"EAM_FS.calculate expects (B, N, 3) with N={self.n_atoms}")

        device = config.device
        dtype = config.dtype

        # Use runtime cell if provided, otherwise fall back to self.cell
        box = cell if cell is not None else self.cell

        # pairwise displacements using per-batch box support
        rij = _pairwise_displacements(config, box)
        r = torch.linalg.norm(rij, dim=-1).clamp_min(0.0)  # shape (B, N, N)

        # mask neighbors: exclude self (r>0) and within cutoff in MD units
        mask = (r > 0.0) & (r < float(self.cutoff_md))

        # FS distances
        r_fs = r * float(self.r_scale)

        # interpolate electron density contribution rho(r_fs)
        rho_contrib = self._interp_uniform(self.rhor.to(device=device, dtype=dtype), float(self.dr), r_fs)
        # zero out outside-mask
        rho_contrib = rho_contrib * mask.to(dtype)
        # per-atom electron density: sum over neighbors (dim=2)
        rhos_pa = rho_contrib.sum(dim=2)  # shape (B, N)

        # embedded energy per atom: F(rho_i)
        Eis_embed_fs = self._interp_uniform(self.frho.to(device=device, dtype=dtype), float(self.drho), rhos_pa)
        Eis_embed = self.e_scale * Eis_embed_fs

        # pair term (phi = z2 / r_fs); get z2(r_fs)
        # avoid dividing by zero by masking
        z2 = self._interp_uniform(self.z2r.to(device=device, dtype=dtype), float(self.dr), r_fs)
        # safe r_fs for division (where mask is False set arbitrary nonzero to avoid nan)
        rfs_safe = r_fs.clone()
        rfs_safe[~mask] = 1.0
        phi = z2 / rfs_safe
        phi = phi * mask.to(dtype)
        # half-per-atom contribution (pair split)
        val_pair = 0.5 * self.e_scale * phi
        phis_pa = val_pair.sum(dim=2)

        # total per-atom energy
        Eis = Eis_embed + phis_pa
        return Eis.view(config.shape[0], self.n_atoms, 1)

    def total_energy(self, config: torch.Tensor, cell: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute total energy for a batch of configurations.

        Parameters
        ----------
        config : torch.Tensor
            Positions shaped (B, N, 3) or (B, N*3) where N == self.n_atoms.
        cell : torch.Tensor, optional
            Runtime cell for NPT simulations. If provided, overrides self.cell.

        Returns
        -------
        torch.Tensor
            Total energy per configuration, shape (B,).
        """
        # Handle flattened input
        if config.dim() == 2:
            config = config.view(config.shape[0], self.n_atoms, 3)
        
        per_atom = self.calculate(config, cell=cell)  # (B, N, 1)
        return per_atom.sum(dim=(1, 2))  # (B,)

    def forward(self, pos: torch.Tensor, cell: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional runtime cell for NPT."""
        return self.calculate(pos, cell=cell)
