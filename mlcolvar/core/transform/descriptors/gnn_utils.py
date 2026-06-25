import numpy as np
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from mlcolvar.core.transform.descriptors.utils import sanitize_positions_shape

def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Function that sums the segments of a matrix. Each row has a non-unique ID and all rows with the same ID are summed such that a matrix with the number of rows equal to the number of unique IDs is obtained.

    Parameters
    ----------
    data: torch.Tensor
        A tensor that contains the data that is to be summed.
    segment_ids: torch.Tensor
        An array that has the same number of entries as data has rows which indicates which rows shall be summed.
    num_segments: int
        This is the number of unique IDs, i.e. the dimensionality of the resulting tensor.
        
    Returns
    -------
    torch.Tensor
        Returns a tensor shaped num_segments x data.size(1) containing all the segment sums.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_zeros(result_shape)  # Init empty result tensor.
    segment_ids_exp = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    # Use non-inplace scatter_add (returns new tensor, better for 2nd derivatives)
    result = result.scatter_add(0, segment_ids_exp, data)
    return result



def soft_weighted_unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, seg_weights: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Function that sums the segments of a matrix. Each row has a non-unique ID and all rows with the same ID are summed such that a matrix with the number of rows equal to the number of unique IDs is obtained.

    This version is numerically stable for second derivatives (Hessian computation).

    Parameters
    ----------
    data: torch.Tensor
        A tensor that contains the data that is to be summed.
    segment_ids: torch.Tensor
        An array that has the same number of entries as data has rows which indicates which rows shall be summed.
    seg_weights: torch.Tensor
        An array of weights corresponding to each segment.
    num_segments: int
        This is the number of unique IDs, i.e. the dimensionality of the resulting tensor.
        
    Returns
    -------
    torch.Tensor
        Returns a tensor shaped num_segments x data.size(1) containing all the segment sums.
    """
    # Numerical stabilisation for 2nd derivatives:
    # 1. Clamp logits to prevent exp overflow
    # 2. Use softmax with numerical stability (subtract max per segment)
    # 3. Add small epsilon to denominator instead of using torch.where
    
    sw = seg_weights.squeeze(-1)  # (E,)
    eps = 1e-8  # Small epsilon for numerical stability
    
    # Clamp to prevent exp overflow
    sw = torch.clamp(sw, min=-50.0, max=50.0)
    exp_weights = torch.exp(sw)  # (E,)

    # Sum of exp weights per segment for normalisation
    # Use zeros_like to maintain gradient flow
    w_sum = data.new_zeros((num_segments,))
    if segment_ids.numel() > 0:
        w_sum = w_sum.scatter_add(0, segment_ids, exp_weights)
    
    # CRITICAL: Use smooth regularization instead of torch.where
    # torch.where creates discontinuities that break 2nd derivatives
    # Adding eps ensures we never divide by exactly zero
    w_sum_safe = w_sum + eps  # Now always > 0

    # Aggregate weighted messages  
    result = data.new_zeros((num_segments, data.size(1)))
    if segment_ids.numel() > 0:
        # Normalize weights using safe denominator
        norm_weights = exp_weights / w_sum_safe[segment_ids]  # (E,)
        
        # Weighted aggregation using non-inplace scatter_add
        segment_ids_exp = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result = result.scatter_add(0, segment_ids_exp, data * norm_weights.unsqueeze(-1))
    
    return result


def _sanitize_cell_local(cell, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Device-aware cell sanitization that keeps tensor on correct device.
    
    This replaces mlcolvar's sanitize_cell_shape for TorchScript compatibility.
    Creates cell tensor directly on the target device to avoid device mismatches.
    
    Parameters
    ----------
    cell : float, list, or torch.Tensor
        Cell specification in various formats
    device : torch.device
        Target device
    dtype : torch.dtype
        Target dtype
        
    Returns
    -------
    torch.Tensor
        Cell as shape (3,) [Lx, Ly, Lz] on target device
    """
    # Handle tensor input
    if isinstance(cell, torch.Tensor):
        cell = cell.to(device=device, dtype=dtype)
        
        # Flatten to 1D if needed for consistent handling
        if cell.dim() == 0:
            # Scalar -> cubic
            return cell.expand(3).clone()
        
        cell_flat = cell.flatten()
        numel = cell_flat.numel()
        
        if numel == 1:
            return cell_flat.expand(3).clone()
        elif numel == 3:
            return cell_flat
        elif numel == 6:
            # LAMMPS bounds
            Lx = cell_flat[1] - cell_flat[0]
            Ly = cell_flat[3] - cell_flat[2]
            Lz = cell_flat[5] - cell_flat[4]
            return torch.stack([Lx, Ly, Lz])
        elif numel == 9:
            # 3x3 matrix -> diagonal
            mat = cell_flat.view(3, 3)
            return torch.diag(mat)
        else:
            raise ValueError(f"Unsupported cell tensor with {numel} elements")
    
    # Handle scalar
    elif isinstance(cell, (int, float)):
        return torch.tensor([cell, cell, cell], device=device, dtype=dtype)
    
    # Handle list/tuple
    elif isinstance(cell, (list, tuple)):
        cell_t = torch.tensor(cell, device=device, dtype=dtype)
        return _sanitize_cell_local(cell_t, device, dtype)
    
    else:
        raise ValueError(f"Unsupported cell type: {type(cell)}")
    

def _apply_pbc_distances(dist_components, pbc_cell):
    """Apply PBC corrections to distance components.
    
    Uses device-aware operations to avoid TorchScript device mismatches.
    All scalar constants are created as tensors on the same device as inputs.
    """
    device = dist_components.device
    dtype = dist_components.dtype
    
    # Ensure pbc_cell is on same device/dtype
    pbc_cell = pbc_cell.to(device=device, dtype=dtype)
    
    # Create scalar constants on correct device (critical for TorchScript)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    two = torch.tensor(2.0, device=device, dtype=dtype)
    
    # Compute PBC shifts using device-aware operations
    # For each dimension d: shift = round(dist / L) * L
    # This uses: round(x) = floor(x + 0.5) = trunc(x + sign(x)*0.5)
    
    # Compute shifts for each dimension
    # shifts = trunc(dist / (L/2)) then adjust
    # final_shift = trunc((shifts + sign(shifts)) / 2) * L
    
    # Extract cell lengths for broadcasting (reshape to [1, 3, 1, 1] for batched ops)
    Lx = pbc_cell[0].reshape(1, 1, 1)
    Ly = pbc_cell[1].reshape(1, 1, 1)  
    Lz = pbc_cell[2].reshape(1, 1, 1)
    half_Lx = Lx / two
    half_Ly = Ly / two
    half_Lz = Lz / two
    
    # dist_components shape: (B, 3, N, N)
    # Apply PBC separately for each dimension
    dx = dist_components[:, 0:1, :, :]
    dy = dist_components[:, 1:2, :, :]
    dz = dist_components[:, 2:3, :, :]
    
    # PBC correction: d_wrapped = d - round(d/L)*L
    # where round(x) = trunc(x + sign(x)*0.5)
    def wrap_dim(d, L, half_L):
        # Compute number of cell crossings
        n = torch.div(d, half_L, rounding_mode='trunc')
        # Adjust for proper rounding
        n = torch.div(n + torch.sign(n) * one, two, rounding_mode='trunc')
        # Compute shift
        return d - n * L
    
    dx_wrapped = wrap_dim(dx, Lx, half_Lx)
    dy_wrapped = wrap_dim(dy, Ly, half_Ly)
    dz_wrapped = wrap_dim(dz, Lz, half_Lz)
    
    # Concatenate back
    dist_components = torch.cat([dx_wrapped, dy_wrapped, dz_wrapped], dim=1)
    
    return dist_components


def compute_distances_matrix_safe(pos: torch.Tensor,
                                   n_atoms: int,
                                   PBC: bool,
                                   cell: Union[float, list, torch.Tensor],
                                   vector: bool = False,
                                   scaled_coords: bool = False,
                                   eps: float = 1e-8,
                                  ) -> torch.Tensor:
    """Compute pairwise distances matrix with numerical stability for 2nd derivatives. Relevant for computing Committor-type biases.
    
    This is a modified version of compute_distances_matrix that adds
    epsilon regularization inside sqrt to prevent NaN in second derivatives when
    atoms are very close together.
    
    Parameters
    ----------
    pos : torch.Tensor
        Positions shape (batch, n_atoms, 3) or (batch, n_atoms*3)
    n_atoms : int
        Number of atoms
    PBC : bool
        Use periodic boundary conditions
    cell : Union[float, list, torch.Tensor]
        Cell dimensions
    vector : bool
        Return vector distances instead of scalar
    scaled_coords : bool
        Coordinates are scaled to [0,1]
    eps : float
        Small value to add inside sqrt for numerical stability (default 1e-8)
        
    Returns
    -------
    torch.Tensor
        Distance matrix (batch, n_atoms, n_atoms)
    """
    pos, batch_size = sanitize_positions_shape(pos, n_atoms)

    _device = pos.device
    _dtype = pos.dtype
    
    # Use local cell sanitization to avoid device mismatches
    # This creates tensors directly on the correct device
    cell = _sanitize_cell_local(cell, _device, _dtype)

    if scaled_coords:
        pbc_cell = torch.tensor([1., 1., 1.], device=_device, dtype=_dtype)
    else:
        pbc_cell = cell
    
    pos = torch.reshape(pos, (batch_size, n_atoms, 3))
    pos = torch.transpose(pos, 1, 2)
    pos = pos.reshape((batch_size, 3, n_atoms))

    pos_expanded = torch.tile(pos, (1, 1, n_atoms)).reshape(batch_size, 3, n_atoms, n_atoms)
    dist_components = pos_expanded - torch.transpose(pos_expanded, -2, -1)

    if PBC:
        dist_components = _apply_pbc_distances(dist_components=dist_components, pbc_cell=pbc_cell)

    if scaled_coords:
        dist_components = torch.einsum('bijk,i->bijk', dist_components, cell)

    if vector: 
        return dist_components
    else:
        # Sum squared components
        dist_sq = torch.sum(torch.pow(dist_components, 2), 1)  # (batch, n_atoms, n_atoms)
        
        # Add epsilon INSIDE sqrt to prevent NaN in 2nd derivatives
        # sqrt(r² + eps) has bounded 2nd derivatives even when r→0
        # For diagonal (self-distance), we set to 0 after
        dist = torch.sqrt(dist_sq + eps)
        
        # Zero out diagonal (self-distances should be exactly 0, not sqrt(eps))
        diag_mask = torch.eye(n_atoms, dtype=torch.bool, device=_device).unsqueeze(0).expand(batch_size, -1, -1)
        dist = dist.masked_fill(diag_mask, 0.0)
        
        return dist
    

def build_graphs_for_positions(
    pos: torch.Tensor,
    n_atoms: int,
    PBC: bool,
    cell: Union[float, List[float], np.ndarray],
    cutoff: float,
) -> List[torch.Tensor]:
    """Precompute cutoff graphs (edge_index) for each structure in a batch of positions.

    Parameters
    ----------
    pos : torch.Tensor
        Positions tensor of shape (B, n_atoms, 3) or (B, n_atoms*3).
    n_atoms : int
        Number of atoms per structure.
    PBC : bool
        Whether to apply periodic boundary conditions.
    cell : float | List[float] | np.ndarray
        Cell dimensions used for PBC. Accepts multiple formats:
        - Scalar: cubic box with this side length
        - [Lx, Ly, Lz]: orthorhombic box lengths (length 3)
        - [xlo, xhi, ylo, yhi, zlo, zhi]: LAMMPS-style bounds (length 6)
    cutoff : float
        Distance cutoff for connecting edges.

    Returns
    -------
    List[torch.Tensor]
        A list of length B, where each element is a (2, E_i) LongTensor with local indices [0..n_atoms-1].
    """
    pos, _ = sanitize_positions_shape(pos=pos, n_atoms=n_atoms)
    B, N, D = pos.shape
    if N != n_atoms or D != 3:
        raise ValueError(f"Expected positions of shape (B, {n_atoms}, 3), got {tuple(pos.shape)}")

    # Convert LAMMPS bounds or other formats to box lengths [Lx, Ly, Lz]
    box = _sanitize_cell_local(cell, device=pos.device, dtype=pos.dtype)
    
    Dmat = compute_distances_matrix_safe(pos=pos, n_atoms=n_atoms, PBC=PBC, cell=box, scaled_coords=False)
    self_mask = ~torch.eye(n_atoms, dtype=torch.bool, device=pos.device).unsqueeze(0).expand(B, -1, -1)
    cutmask = (Dmat > 0) & (Dmat <= float(cutoff)) & self_mask

    graphs: List[torch.Tensor] = []
    for b in range(B):
        idx = cutmask[b].nonzero(as_tuple=False)
        if idx.numel() == 0:
            graphs.append(torch.zeros((2, 0), dtype=torch.long))
        else:
            row = idx[:, 0].to(torch.long).cpu()
            col = idx[:, 1].to(torch.long).cpu()
            graphs.append(torch.stack([row, col], dim=0))
    return graphs


@torch.no_grad()
def build_graphs_for_dataset(
    dataset,
    n_atoms: int,
    PBC: bool,
    cell: Union[float, List[float], np.ndarray],
    cutoff: float,
    per_frame_bounds: Optional[Union[List, np.ndarray]] = None,
) -> List[torch.Tensor]:
    """Precompute graphs for all structures in a DictDataset-like object.

    The `dataset` is expected to have a 'data' key with positions. Returns a list of edge_index tensors
    (2, E_i) with local indices per structure. You can then assign it back as dataset['graph'] = graphs.
    
    Parameters
    ----------
    dataset : DictDataset-like
        Dataset with 'data' key containing positions
    n_atoms : int
        Number of atoms per structure
    PBC : bool
        Whether to apply periodic boundary conditions
    cell : float | List[float] | np.ndarray
        Cell dimensions used for PBC if per_frame_bounds is None.
        Accepts multiple formats (see build_graphs_for_positions).
    cutoff : float
        Distance cutoff for connecting edges
    per_frame_bounds : List | np.ndarray, optional
        Per-frame cell bounds for NPT simulations. Shape (n_frames, 6) where each
        row is [xlo, xhi, ylo, yhi, zlo, zhi], OR shape (n_frames, 3) for box lengths.
        If provided, overrides the `cell` argument.
        
    Returns
    -------
    List[torch.Tensor]
        A list of length B, where each element is a (2, E_i) LongTensor with local indices.
    """
    pos = dataset["data"]
    n_frames = len(pos)
    
    if per_frame_bounds is not None:
        # NPT mode: build graphs frame-by-frame with varying cell
        graphs = []
        for i in range(n_frames):
            frame_pos = pos[i:i+1]  # Keep batch dimension
            frame_cell = per_frame_bounds[i]
            frame_graphs = build_graphs_for_positions(
                pos=frame_pos, n_atoms=n_atoms, PBC=PBC, cell=frame_cell, cutoff=cutoff
            )
            graphs.extend(frame_graphs)
        return graphs
    else:
        # NVT mode: single cell for all frames
        return build_graphs_for_positions(pos=pos, n_atoms=n_atoms, PBC=PBC, cell=cell, cutoff=cutoff)



class CurvatureRegularizationLoss(nn.Module):
    """Curvature Regularization (CR) loss for correlating model gradient space with potential energy space.
    
    The loss encourages the model to learn physically meaningful paths through phase space by correlating
    local curvature in descriptor space with curvature in potential energy space:
    
    L_CR = Σ_{l=1}^{L} α^l · |log|Δh_l| - β·ΔV_l|
    
    where:
    - Δh_l = ||h_{l} - h_{l-1}||_2 is the change in descriptor space
    - ΔV_l = |V_{l} - V_{l-1}| is the change in potential energy
    - α is a decay factor for distant steps
    - β scales the energy contribution
    
    The direction of movement is towards a reference state (e.g., state B),
    defined as: direction = h_reference - h_current (normalized).

    When using this feature please cite:

    ////////////////////////////////////////////////
    J. Chem. Phys. 14 March 2026; 164 (10): 104115. 
    https://doi.org/10.1063/5.0311722
    ////////////////////////////////////////////////
    
    Parameters
    ----------
    energy_calculator : nn.Module
        Energy calculator (e.g., EAM_FS) with a total_energy(pos) method
    reference_state : torch.Tensor
        Mean representation of the target state (e.g., state B), shape (out_features,)
    n_atoms : int
        Number of atoms in the system
    step_scale : float
        Scale factor for position steps (in same units as positions, e.g., Angstroms)
    n_steps : int
        Number of steps along the path (L)
    decay_alpha : float
        Decay factor for distant steps (α)
    beta : float
        Energy scaling factor (β)
    max_loss : float
        Maximum loss value per step to prevent exploding gradients
    eps : float
        Small value for numerical stability in log
    """
    
    def __init__(
        self,
        energy_calculator: nn.Module,
        reference_state: torch.Tensor,
        n_atoms: int,
        step_scale: float = 0.05,
        n_steps: int = 5,
        decay_alpha: float = 0.9,
        beta: float = 1.0,
        max_loss: float = 10.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.energy_calculator = energy_calculator
        self.register_buffer('reference_state', reference_state)
        self.n_atoms = int(n_atoms)
        self.step_scale = float(step_scale)
        self.n_steps = int(n_steps)
        self.decay_alpha = float(decay_alpha)
        self.beta = float(beta)
        self.max_loss = float(max_loss)
        self.eps = float(eps)
    
    def _bounds_to_cell(self, bounds: torch.Tensor) -> torch.Tensor:
        """Convert LAMMPS-style bounds to cell dimensions.
        
        Parameters
        ----------
        bounds : torch.Tensor
            Shape (B, 6) for LAMMPS format [xlo, xhi, ylo, yhi, zlo, zhi]
            or (B, 3) already as cell dimensions [Lx, Ly, Lz]
            or (B, 9) / (B, 3, 3) for full cell matrix
            
        Returns
        -------
        torch.Tensor
            Cell dimensions shape (B, 3) as [Lx, Ly, Lz]
        """
        if bounds.dim() == 1:
            bounds = bounds.unsqueeze(0)
        
        B = bounds.shape[0]
        
        if bounds.shape[-1] == 6:
            # LAMMPS format: [xlo, xhi, ylo, yhi, zlo, zhi]
            Lx = bounds[:, 1] - bounds[:, 0]
            Ly = bounds[:, 3] - bounds[:, 2]
            Lz = bounds[:, 5] - bounds[:, 4]
            return torch.stack([Lx, Ly, Lz], dim=-1)  # (B, 3)
        elif bounds.shape[-1] == 3:
            # Already cell dimensions
            return bounds
        elif bounds.shape[-1] == 9:
            # Flattened 3x3 matrix, extract diagonal
            cell_mat = bounds.view(B, 3, 3)
            return torch.diagonal(cell_mat, dim1=-2, dim2=-1)  # (B, 3)
        elif bounds.dim() == 3 and bounds.shape[-2:] == (3, 3):
            # Full 3x3 cell matrix, extract diagonal
            return torch.diagonal(bounds, dim1=-2, dim2=-1)  # (B, 3)
        else:
            raise ValueError(f"Unsupported bounds shape: {bounds.shape}. "
                           f"Expected (B, 6), (B, 3), (B, 9), or (B, 3, 3)")
    
    def _compute_direction(self, current_state: torch.Tensor) -> torch.Tensor:
        """Compute normalized direction vector towards reference state.
        
        Parameters
        ----------
        current_state : torch.Tensor
            Current histogram predictions, shape (B, out_features)
            
        Returns
        -------
        torch.Tensor
            Normalized direction vectors, shape (B, out_features)
        """
        # Direction: reference - current (pointing towards reference)
        ref = self.reference_state.to(current_state.device)
        direction = ref.unsqueeze(0) - current_state  # (B, out_features)
        
        # Normalize to unit vectors
        norm = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=self.eps)
        direction_normalized = direction / norm
        
        return direction_normalized
    
    def forward(
        self,
        descriptor: nn.Module,
        positions: torch.Tensor,
        bounds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the CR loss for a batch of configurations.
        
        Parameters
        ----------
        descriptor : nn.Module
            The GNN descriptor model
        positions : torch.Tensor
            Input positions, shape (B, n_atoms*3) or (B, n_atoms, 3)
        bounds : torch.Tensor, optional
            Per-batch cell/box information for NPT simulations.
            Shape: (B, 3), (B, 6), (B, 9), (B, 3, 3), or broadcastable.
            Passed to both descriptor and energy_calculator.
            
        Returns
        -------
        torch.Tensor
            Mean CR loss over the batch (scalar)
        """
        device = positions.device
        B = positions.shape[0]
        
        # Convert bounds from LAMMPS format to cell dimensions if needed
        cell = None
        if bounds is not None:
            cell = self._bounds_to_cell(bounds)  # (B, 3)
        
        # Ensure positions are flattened (B, n_atoms*3) and require gradients
        if positions.dim() == 3:
            pos_flat = positions.view(B, -1)
        else:
            pos_flat = positions
        
        # We need gradients w.r.t. positions for stepping
        pos = pos_flat.detach().requires_grad_(True)
        
        # Always recompute h from pos to establish gradient path
        # Pass cell (converted from bounds) to descriptor for NPT support
        h = descriptor(pos, cell=cell)
        
        # Compute initial energy with cell for NPT support
        pos_3d = pos.view(B, self.n_atoms, 3)
        V = self.energy_calculator.total_energy(pos_3d, cell=cell)  # (B,)
        
        # Initialize loss accumulator
        loss_total = torch.zeros(B, device=device, dtype=pos.dtype)
        
        for step in range(1, self.n_steps + 1):
            # Compute direction towards reference in state space
            direction = self._compute_direction(h)  # (B, out_features)
            
            # Compute gradient of state w.r.t. positions
            # We want to move positions such that h moves towards reference
            # d(h)/d(pos) @ direction gives the position update direction
            
            # Use the dot product of state with direction as the scalar to differentiate
            # This projects the state change onto the desired direction
            objective = (h * direction).sum(dim=-1).sum()  # scalar
            
            # Compute gradient w.r.t. positions (Jacobian @ direction)
            grad_pos = torch.autograd.grad(
                outputs=objective,
                inputs=pos,
                create_graph=True,
                retain_graph=True,
            )[0]  # (B, n_atoms*3)
            
            # Normalize gradient per sample and scale
            grad_norm = torch.linalg.norm(grad_pos, dim=-1, keepdim=True).clamp(min=self.eps)
            step_direction = grad_pos / grad_norm * self.step_scale
            
            # Take a step in position space
            pos_new = pos + step_direction
            
            # Compute new state (pass cell for NPT support)
            h_new = descriptor(pos_new, cell=cell)
            
            # Compute new energy (pass cell for NPT support)
            pos_new_3d = pos_new.view(B, self.n_atoms, 3)
            V_new = self.energy_calculator.total_energy(pos_new_3d, cell=cell)  # (B,)
            
            # Compute changes
            delta_h = torch.linalg.norm(h_new - h, dim=-1)  # (B,)
            delta_V = torch.abs(V_new - V)  # (B,)
            
            # CR loss for this step: |log(|Δh|) - β·ΔV|
            # Add eps inside log for numerical stability
            log_delta_h = torch.log(delta_h + self.eps)
            step_loss = torch.abs(log_delta_h - self.beta * delta_V)
            
            # Clamp to prevent exploding gradients
            step_loss = torch.clamp(step_loss, max=self.max_loss)
            
            # Add to total with decay
            loss_total = loss_total + (self.decay_alpha ** step) * step_loss
            
            # Update for next iteration
            pos = pos_new
            h = h_new
            V = V_new
        
        # Return mean over batch
        return loss_total.mean()


def compute_gnn_jacobian_frobenius_norm(
    gnn: nn.Module,
    positions: torch.Tensor,
    n_atoms: int,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Compute the average Frobenius norm of the GNN Jacobian (d_descriptor/d_positions).
    
    This is used to normalize the GNN output so that different model instances
    trained on the same data produce gradients of similar magnitude.
    For details and theoretical justification, see:

    ////////////////////////////////////////////
    J. Chem. Phys. 163, 141102 (2025)
    https://doi.org/10.1063/5.0287912
    ////////////////////////////////////////////
    
    Parameters
    ----------
    gnn : nn.Module
        The GNN descriptor model (e.g., GNNTransformerDescriptor)
    positions : torch.Tensor
        Dataset of positions, shape (N_samples, n_atoms*3) or (N_samples, n_atoms, 3)
    n_atoms : int
        Number of atoms
    batch_size : int
        Batch size for processing (to manage memory)
    device : torch.device, optional
        Device for computation. If None, uses the GNN's device.
        
    Returns
    -------
    Tuple[float, float]
        (mean_frobenius_norm, std_frobenius_norm) over the dataset
    """
    if device is None:
        device = next(gnn.parameters()).device
    
    gnn = gnn.to(device).eval()
    
    # Ensure positions are (N, n_atoms*3)
    pos, _ = sanitize_positions_shape(positions, n_atoms)
    pos = pos.view(pos.shape[0], -1)  # (N, n_atoms*3)
    
    n_samples = pos.shape[0]
    frobenius_norms = []
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_pos = pos[start:end].to(device).requires_grad_(True)
        
        # Forward pass through GNN
        desc = gnn(batch_pos)  # (B, out_features)
        
        # Compute Jacobian via backward pass for each output dimension
        # Frobenius norm: ||J||_F = sqrt(sum_ij J_ij^2)
        # We compute this as sqrt(sum over outputs of ||grad_i||^2)
        B, D_out = desc.shape
        
        # Accumulate squared gradients
        jacobian_sq_sum = torch.zeros(B, device=device)
        
        for j in range(D_out):
            # Gradient of j-th output w.r.t. all inputs
            grad_j = torch.autograd.grad(
                outputs=desc[:, j].sum(),
                inputs=batch_pos,
                retain_graph=(j < D_out - 1),  # Keep graph for all but last
                create_graph=False
            )[0]  # (B, D_in)
            
            # Sum of squared gradients for this output dimension
            jacobian_sq_sum += (grad_j ** 2).sum(dim=1)  # (B,)
        
        # Frobenius norm for each sample in batch
        batch_frob_norms = torch.sqrt(jacobian_sq_sum).detach().cpu()
        frobenius_norms.append(batch_frob_norms)
        
        # Clear gradients
        batch_pos.grad = None
    
    all_norms = torch.cat(frobenius_norms)
    mean_norm = all_norms.mean().item()
    std_norm = all_norms.std().item()
    
    return mean_norm, std_norm

class JacobianNormalizedGNN(nn.Module):
    """Wrapper that normalizes the GNN output so that the Jacobian has unit Frobenius norm on average.
    
    The scaling is: output = gnn(x) / jacobian_scale
    
    This means: d(output)/d(x) = d(gnn(x))/d(x) / jacobian_scale
    
    If jacobian_scale = mean(||J||_F), then the normalized Jacobian has average Frobenius norm ~1.

    For details and theoretical justification, see:

    ////////////////////////////////////////////
    J. Chem. Phys. 163, 141102 (2025)
    https://doi.org/10.1063/5.0287912
    ////////////////////////////////////////////
    
    Parameters
    ----------
    gnn : nn.Module
        The GNN descriptor model to wrap
    jacobian_scale : float
        The normalization constant (typically the mean Frobenius norm computed on training data)
    """
    
    def __init__(self, gnn: nn.Module, jacobian_scale: float):
        super().__init__()
        self.gnn = gnn
        # Store as buffer so it gets saved/loaded with the model
        self.register_buffer('jacobian_scale', torch.tensor(jacobian_scale, dtype=torch.float32))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional pre-computed edge_index and runtime cell.
        
        Parameters
        ----------
        x : torch.Tensor
            Input positions
        edge_index : torch.Tensor, optional
            Pre-computed edge indices for Verlet list support
        cell : torch.Tensor, optional
            Runtime cell for NPT simulations
        
        Returns
        -------
        torch.Tensor
            Normalized GNN output
        """
        return self.gnn(x, edge_index=edge_index, cell=cell) / self.jacobian_scale
    
    # Delegate attribute access to wrapped GNN for compatibility
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.gnn, name)


def create_normalized_gnn(
    gnn: nn.Module,
    positions: torch.Tensor,
    n_atoms: int,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[JacobianNormalizedGNN, float, float]:
    """Create a Jacobian-normalized GNN wrapper.
    
    Computes the mean Frobenius norm of the Jacobian over the provided positions
    and returns a wrapped GNN that divides its output by this scale factor.

    For details and theoretical justification, see:

    ////////////////////////////////////////////
    J. Chem. Phys. 163, 141102 (2025)
    https://doi.org/10.1063/5.0287912
    ////////////////////////////////////////////

    Example usage:
    >>> n_samples_for_norm = min(1000, len(ds_unbiased['data']))
    >>> positions_for_norm = ds_unbiased['data'][:n_samples_for_norm]
    >>> normalized_gnn, mean_jac_norm, std_jac_norm = create_normalized_gnn(
    >>>     gnn=model_train,
    >>>     positions=positions_for_norm,
    >>>     n_atoms=n_at,
    >>>     batch_size=32,
    >>>     device=next(model_train.parameters()).device,
    >>> )
    
    Parameters
    ----------
    gnn : nn.Module
        The GNN descriptor model
    positions : torch.Tensor
        Training positions to compute the normalization constant from
    n_atoms : int
        Number of atoms
    batch_size : int
        Batch size for Jacobian computation
    device : torch.device, optional
        Device for computation
        
    Returns
    -------
    Tuple[JacobianNormalizedGNN, float, float]
        (normalized_gnn, mean_jacobian_norm, std_jacobian_norm)
    """
    print("Computing GNN Jacobian Frobenius norms over dataset...")
    mean_norm, std_norm = compute_gnn_jacobian_frobenius_norm(
        gnn=gnn,
        positions=positions,
        n_atoms=n_atoms,
        batch_size=batch_size,
        device=device,
    )
    print(f"  Mean ||J||_F = {mean_norm:.6f}")
    print(f"  Std  ||J||_F = {std_norm:.6f}")
    print(f"  Coefficient of variation = {std_norm/mean_norm:.2%}")
    
    normalized_gnn = JacobianNormalizedGNN(gnn, jacobian_scale=mean_norm)
    
    return normalized_gnn, mean_norm, std_norm